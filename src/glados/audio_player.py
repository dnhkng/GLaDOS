# --- audio_player.py ---
import queue
import threading
import time

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from .glados_config import AudioMessage  # Assuming AudioMessage is in glados_config.py or a shared types.py

# from .utils import spoken_text_converter as stc # Not needed here, stc is for TTS part

class AudioPlayer:
    def __init__(self,
                 audio_output_queue: queue.Queue[AudioMessage],
                 conversation_history: list[dict[str, str]],
                 tts_sample_rate: int,
                 interruptible: bool,
                 shutdown_event: threading.Event,
                 currently_speaking_event: threading.Event,
                 processing_active_event: threading.Event, # To check if we should interrupt
                 pause_time: float = 0.05):
        self.audio_output_queue = audio_output_queue
        self.conversation_history = conversation_history # Shared list
        self.tts_sample_rate = tts_sample_rate
        self.interruptible = interruptible
        self.shutdown_event = shutdown_event
        self.currently_speaking_event = currently_speaking_event
        self.processing_active_event = processing_active_event # Used to know when to stop
        self.pause_time = pause_time

        # For managing the output stream if refactoring playback
        self.output_stream: sd.OutputStream | None = None
        self.playback_lock = threading.Lock()
        self.audio_playback_finished_event = threading.Event()
        self.played_samples_count = 0
        self.current_audio_total_samples = 0
        self.playback_interrupted_flag = False
        self.PLAYBACK_TIMEOUT_BUFFER_SECONDS = 2.0


    def _percentage_played_refactored(self, audio_data: NDArray[np.float32]) -> tuple[bool, int]:
        """
        Plays audio using an OutputStream and manages interruption.
        Returns (interrupted, percentage_played).
        """
        self.current_audio_total_samples = len(audio_data)
        self.played_samples_count = 0
        self.playback_interrupted_flag = False # Reset for current audio
        self.audio_playback_finished_event.clear()

        interrupted_by_logic = False
        percentage = 0

        try:
            with self.playback_lock: # Ensure exclusive access to output_stream
                if self.output_stream and self.output_stream.active:
                    logger.debug("Stopping existing audio stream for new playback.")
                    self.output_stream.stop(ignore_errors=True)
                    self.output_stream.close() # Ensure it's closed before reopening

                self.output_stream = sd.OutputStream(
                    samplerate=self.tts_sample_rate,
                    channels=1, # Assuming mono
                    callback=self._audio_playback_callback_factory(audio_data),
                    finished_callback=self.audio_playback_finished_event.set
                )
            
            with self.output_stream: # Starts the stream
                timeout_duration = (self.current_audio_total_samples / self.tts_sample_rate) + self.PLAYBACK_TIMEOUT_BUFFER_SECONDS
                # Wait for playback to finish or timeout
                if not self.audio_playback_finished_event.wait(timeout=timeout_duration):
                    logger.warning("Audio playback timed out or explicitly stopped.")
                    # If it timed out, ensure it's marked as interrupted if not already by the callback
                    if not self.playback_interrupted_flag:
                         self.playback_interrupted_flag = True
                    if self.output_stream.active: # Check if active before stopping
                        self.output_stream.stop(ignore_errors=True) # Ensure stream is stopped

        except sd.PortAudioError as e:
            logger.error(f"PortAudioError during playback: {e}")
            self.playback_interrupted_flag = True # Consider this an interruption
        except RuntimeError as e: # Catches "RuntimeError: Error opening OutputStream: PortAudioError"
            logger.error(f"RuntimeError during playback (possibly stream already closed/invalid): {e}")
            self.playback_interrupted_flag = True
        except Exception as e:
            logger.exception(f"Unexpected error in _percentage_played_refactored: {e}")
            self.playback_interrupted_flag = True
        finally:
            with self.playback_lock:
                if self.output_stream:
                    if self.output_stream.active:
                         self.output_stream.stop(ignore_errors=True)
                    self.output_stream.close(ignore_errors=True)
                    self.output_stream = None # Clear the stream instance


        interrupted_by_logic = self.playback_interrupted_flag
        if self.current_audio_total_samples > 0:
            percentage = min(int((self.played_samples_count / self.current_audio_total_samples) * 100), 100)
        else: # No audio data, so 0% played, not interrupted.
            percentage = 0
            interrupted_by_logic = False
            
        return interrupted_by_logic, percentage

    def _audio_playback_callback_factory(self, audio_data: NDArray[np.float32]) -> callable:
        def callback(outdata: NDArray[np.float32], frames: int, time_info: sd.CallbackTimeInfo, status: sd.CallbackFlags) -> None:
            if status:
                logger.warning(f"Audio playback status: {status}")
                if status.output_underflow: # Or other critical statuses
                    self.playback_interrupted_flag = True # Mark as interrupted
                    # No need to raise CallbackStop here, let it try to continue or finish naturally

            # Check for interruption:
            # 1. interruptible is True
            # 2. processing_active_event is NOT set (meaning a new user input is overriding)
            # 3. shutdown_event is set
            should_interrupt = self.interruptible and \
                               (not self.processing_active_event.is_set() or self.shutdown_event.is_set())

            if should_interrupt:
                if not self.playback_interrupted_flag: # Log only once
                    logger.info("Audio playback interruption condition met in callback.")
                self.playback_interrupted_flag = True
                outdata.fill(0)  # Fill with silence
                raise sd.CallbackStop("Playback interrupted by external signal")

            remaining_samples_in_chunk = self.current_audio_total_samples - self.played_samples_count
            chunk_size = min(remaining_samples_in_chunk, frames)

            if chunk_size > 0:
                outdata[:chunk_size] = audio_data[self.played_samples_count : self.played_samples_count + chunk_size]
            if chunk_size < frames:
                outdata[chunk_size:] = 0  # Zero-pad if audio_data is exhausted or chunk_size is 0

            self.played_samples_count += chunk_size

            if self.played_samples_count >= self.current_audio_total_samples:
                # No specific error, just completed the data
                if frames > chunk_size : # Ensure we signal stop if we provided all data but more was requested
                     pass # Let the finished_callback handle this
                # No need to raise CallbackStop here, finished_callback will be called.
        return callback

    def run(self) -> None:
        """Main loop for the AudioPlayer thread."""
        assistant_text_accumulator: list[str] = []
        # system_text was for copying assistant_text before appending clipped text.
        # Re-evaluate if this deepcopy logic is still needed.

        logger.info("AudioPlayer thread started.")
        while not self.shutdown_event.is_set():
            try:
                audio_msg = self.audio_output_queue.get(timeout=self.pause_time)

                if not self.processing_active_event.is_set() and self.interruptible:
                    # If we were interrupted (processing_active_event cleared by AudioInputHandler)
                    # and there are items in the queue, clear them.
                    logger.info("AudioPlayer: Interruption signal received, clearing pending audio messages.")
                    if assistant_text_accumulator: # Log any accumulated text before clearing
                         self.conversation_history.append({"role": "assistant", "content": " ".join(assistant_text_accumulator) + "<INTERRUPTED_BEFORE_PLAY>"})
                         assistant_text_accumulator = []
                    with self.audio_output_queue.mutex:
                        self.audio_output_queue.queue.clear()
                    self.currently_speaking_event.clear()
                    continue


                if audio_msg.is_eos:
                    logger.debug("AudioPlayer: Processing end of stream token.")
                    if assistant_text_accumulator:
                        self.conversation_history.append({"role": "assistant", "content": " ".join(assistant_text_accumulator)})
                        assistant_text_accumulator = []
                    self.currently_speaking_event.clear()
                    # The processing_active_event should naturally be set again by the
                    # AudioInputHandler/Orchestrator when a new cycle starts.
                    # Or, if the state machine is in place, it would transition to IDLE here.
                    logger.debug("AudioPlayer: End of stream processed, speaking event cleared.")
                    continue

                if len(audio_msg.audio) > 0 and audio_msg.text: # Ensure there's audio and text
                    self.currently_speaking_event.set() # We are about to speak
                    logger.success(f"AudioPlayer: Playing TTS text: '{audio_msg.text}'")
                    
                    interrupted, percentage_played = self._percentage_played_refactored(audio_msg.audio)

                    if interrupted:
                        logger.info(f"AudioPlayer: Playback interrupted at {percentage_played}%.")
                        clipped_text = self.clip_interrupted_sentence(audio_msg.text, percentage_played)
                        
                        # Combine accumulated text with the clipped part of the current one
                        full_interrupted_message_parts = assistant_text_accumulator + [clipped_text]
                        self.conversation_history.append({"role": "assistant", "content": " ".join(full_interrupted_message_parts)})
                        assistant_text_accumulator = [] # Reset accumulator

                        self.currently_speaking_event.clear()
                        # self.processing_active_event is already cleared by the interrupter (AudioInputHandler)
                        
                        # Clear remaining audio queue as TTS was interrupted
                        logger.debug("AudioPlayer: Clearing audio queue due to interruption.")
                        with self.audio_output_queue.mutex:
                            self.audio_output_queue.queue.clear()
                    else: # Playback completed normally
                        logger.debug(f"AudioPlayer: Playback completed for: '{audio_msg.text}'")
                        assistant_text_accumulator.append(audio_msg.text)
                elif len(audio_msg.audio) == 0 and audio_msg.text:
                    # Case: Text provided but no audio (e.g. from a pure text response LLM?)
                    # This shouldn't happen with current TTS flow but good to acknowledge
                    logger.warning(f"AudioPlayer: Received text '{audio_msg.text}' but no audio data.")
                    assistant_text_accumulator.append(audio_msg.text + "<NO_AUDIO_DATA>")


            except queue.Empty:
                # This is normal, just means no audio to play right now
                pass
            except Exception as e:
                logger.exception(f"AudioPlayer: Unexpected error in run loop: {e}")
                # Potentially add a small sleep here to prevent tight loop on persistent error
                time.sleep(0.1)
        
        # Cleanup stream if it was left open on shutdown
        with self.playback_lock:
            if self.output_stream:
                if self.output_stream.active:
                    self.output_stream.stop(ignore_errors=True)
                self.output_stream.close(ignore_errors=True)
                self.output_stream = None
        logger.info("AudioPlayer thread finished.")

    def clip_interrupted_sentence(self, generated_text: str, percentage_played: float) -> str:
        """
        Clips the generated text based on the percentage of audio played before interruption.
        """
        # This method can be copied directly from the old Glados class
        tokens = generated_text.split()
        # Ensure percentage_played is within 0-100
        percentage_played = max(0.0, min(100.0, float(percentage_played)))
        words_to_print = round((percentage_played / 100) * len(tokens))
        text = " ".join(tokens[:words_to_print])

        if words_to_print < len(tokens) and percentage_played < 100: # Add marker only if actually cut short
            text = text + "<INTERRUPTED>"
        return text