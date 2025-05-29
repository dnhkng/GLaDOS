import queue
import threading
import time
from typing import Any

from loguru import logger

from .audio_message import AudioMessage


class SpeechPlayer:
    def __init__(
        self,
        audio_io: Any,  # Replace with actual type if known
        audio_output_queue: queue.Queue[AudioMessage],
        conversation_history: list[dict[str, str]],
        tts_sample_rate: int,
        shutdown_event: threading.Event,
        currently_speaking_event: threading.Event,
        processing_active_event: threading.Event,  # To check if we should interrupt
        pause_time: float,
    ) -> None:
        self.audio_io = audio_io
        self.audio_output_queue = audio_output_queue
        self.conversation_history = conversation_history  # Shared list
        self.tts_sample_rate = tts_sample_rate
        self.shutdown_event = shutdown_event
        self.currently_speaking_event = currently_speaking_event
        self.processing_active_event = processing_active_event  # Used to know when to stop
        self.pause_time = pause_time

    def run(self) -> None:
        """Main loop for the AudioPlayer thread."""
        assistant_text_accumulator: list[str] = []

        logger.info("AudioPlayer thread started.")
        while not self.shutdown_event.is_set():
            try:
                audio_msg = self.audio_output_queue.get(timeout=self.pause_time)

                audio_len = len(audio_msg.audio) if audio_msg.audio is not None else 0

                if audio_msg.is_eos:
                    logger.debug("AudioPlayer: Processing end of stream token.")
                    self.conversation_history.append(
                        {"role": "assistant", "content": " ".join(assistant_text_accumulator)}
                    )
                    assistant_text_accumulator = []
                    self.currently_speaking_event.clear()
                    continue

                if audio_len and audio_msg.text:  # Ensure there's audio and text
                    self.currently_speaking_event.set()  # We are about to speak

                    self.audio_io.start_speaking(audio_msg.audio, self.tts_sample_rate)

                    logger.success(f"TTS text: {audio_msg.text}")

                    # Wait for the audio to finish playing or be interrupted
                    interrupted, percentage_played = self.audio_io.measure_percentage_spoken(audio_len)

                    if interrupted:
                        clipped_text = self.clip_interrupted_sentence(audio_msg.text, percentage_played)
                        logger.success(f"TTS interrupted at {percentage_played}%: {clipped_text}")

                        assistant_text_accumulator.append(clipped_text)
                        self.conversation_history.append(
                            {"role": "assistant", "content": " ".join(assistant_text_accumulator)}
                        )
                        self.conversation_history.append(
                            {
                                "role": "user",
                                "content": (
                                    f"[SYSTEM: User interrupted mid-response! Full intended output: '{audio_msg.text}']"
                                ),
                            }
                        )
                        assistant_text_accumulator = []  # Reset accumulator
                        self._clear_audio_queue()

                    else:  # Playback completed normally
                        logger.success(f"AudioPlayer: Playback completed for: '{audio_msg.text}'")
                        assistant_text_accumulator.append(audio_msg.text)
                else:
                    logger.warning(f"AudioPlayer: Received empty audio message or no text: {audio_len, audio_msg}")

            except queue.Empty:
                pass  # No audio to play right now

            except Exception as e:
                logger.exception(f"AudioPlayer: Unexpected error in run loop: {e}")
                # Potentially add a small sleep here to prevent tight loop on persistent error
                time.sleep(0.1)
        logger.info("VoicePlayer thread finished.")

    def _clear_audio_queue(self) -> None:
        """Clears the audio output queue and resets the speaking event."""

        logger.debug("AudioPlayer: Clearing audio queue due to interruption.")
        self.currently_speaking_event.clear()
        with self.audio_output_queue.mutex:
            self.audio_output_queue.queue.clear()

    def clip_interrupted_sentence(self, generated_text: str, percentage_played: float) -> str:
        """
        Clips the generated text based on the percentage of audio played before interruption.
        """
        tokens = generated_text.split()
        percentage_played = max(0.0, min(100.0, float(percentage_played)))  # Ensure percentage_played is within 0-100
        words_to_print = round((percentage_played / 100) * len(tokens))
        text = " ".join(tokens[:words_to_print])
        return text
