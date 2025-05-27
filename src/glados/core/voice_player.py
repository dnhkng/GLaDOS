import queue
import threading
import time
from typing import Any

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd  # type: ignore

from .audio_message import AudioMessage


class VoicePlayer:
    def __init__(
        self,
        audio_output_queue: queue.Queue[AudioMessage],
        conversation_history: list[dict[str, str]],
        tts_sample_rate: int,
        shutdown_event: threading.Event,
        currently_speaking_event: threading.Event,
        processing_active_event: threading.Event,  # To check if we should interrupt
        pause_time: float,
    ) -> None:
        self.audio_output_queue = audio_output_queue
        self.conversation_history = conversation_history  # Shared list
        self.tts_sample_rate = tts_sample_rate
        self.shutdown_event = shutdown_event
        self.currently_speaking_event = currently_speaking_event
        self.processing_active_event = processing_active_event  # Used to know when to stop
        self.pause_time = pause_time

    def percentage_played(self, total_samples: int) -> tuple[bool, int]:
        """Calculates the percentage of audio played and checks if playback was interrupted.
        Returns a tuple of (interrupted: bool, percentage_played: int).
        This method uses a callback to track the number of frames played and checks if the playback was interrupted.

        Args:
            total_samples (int): The total number of samples in the audio to be played.
        Returns:
            tuple[bool, int]: A tuple where the first element indicates if playback was interrupted,
                              and the second element is the percentage of audio played.
        """
        interrupted = False
        progress = 0
        completion_event = threading.Event()

        def stream_callback(
            outdata: NDArray[np.float32], frames: int, time: dict[str, Any], status: sd.CallbackFlags
        ) -> tuple[NDArray[np.float32], sd.CallbackStop | None]:
            nonlocal progress, interrupted
            progress += frames
            if not self.processing_active_event.is_set() or self.shutdown_event.is_set():
                interrupted = True
                completion_event.set()
                return outdata, sd.CallbackStop
            if progress >= total_samples:
                completion_event.set()
            return outdata, None

        try:
            stream = sd.OutputStream(
                callback=stream_callback,
                samplerate=self.tts_sample_rate,
                channels=1,
                finished_callback=completion_event.set,
            )
            with stream:
                # Wait with timeout to allow for interruption
                completion_event.wait(timeout=total_samples / self.tts_sample_rate + 1)

        except (sd.PortAudioError, RuntimeError):
            logger.debug("Audio stream already closed or invalid")

        logger.debug(f"played {progress} frames out of {total_samples} total frames.")
        percentage_played = min(int(progress / total_samples * 100), 100)
        return interrupted, percentage_played

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
                    sd.play(audio_msg.audio, self.tts_sample_rate)
                    logger.success(f"TTS text: {audio_msg.text}")

                    # Wait for the audio to finish playing or be interrupted
                    interrupted, percentage_played = self.percentage_played(audio_len)

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
                                    "[SYSTEM: User interrupted mid-response! "
                                   f"Full intended output: '{audio_msg.text}']"
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
