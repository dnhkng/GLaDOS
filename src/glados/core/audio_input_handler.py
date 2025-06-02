# --- audio_input_handler.py ---
import queue
import threading
import time

from Levenshtein import distance  # type: ignore
from loguru import logger
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from glados.ASR import VAD, TranscriberProtocol  # Adjust path

from .glados_config import GladosConfig  # For constants, or pass them individually


class AudioInputHandler:
    def __init__(self,
                 llm_input_queue: queue.Queue[str],
                 vad_model: VAD,
                 asr_model: TranscriberProtocol,
                 config: GladosConfig, # Pass the full config or specific parts
                 shutdown_event: threading.Event,
                 currently_speaking_event: threading.Event, # To know if assistant is speaking
                 processing_active_event: threading.Event # To signal interruption
                ):
        self.llm_input_queue = llm_input_queue
        self.vad_model = vad_model
        self.asr_model = asr_model
        
        # Extract constants from config or pass individually
        self.sample_rate = config.SAMPLE_RATE # Assuming GladosConfig has these as class vars or Glados instance has them
        self.vad_sample_size_ms = config.VAD_SIZE
        self.vad_threshold = config.VAD_THRESHOLD
        self.buffer_size_ms = config.BUFFER_SIZE
        self.pause_limit_ms = config.PAUSE_LIMIT
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        self.wake_word = config.wake_word
        self.interruptible = config.interruptible

        self.shutdown_event = shutdown_event
        self.currently_speaking_event = currently_speaking_event
        self.processing_active_event = processing_active_event

        self._samples_for_asr: list[NDArray[np.float32]] = []
        self._raw_sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue() # From mic callback
        
        # Calculate buffer maxsize based on VAD sample size
        buffer_max_vad_chunks = self.buffer_size_ms // self.vad_sample_size_ms
        self._pre_activation_buffer: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=buffer_max_vad_chunks)
        
        self._recording_activated = False # Renamed from _recording_started for clarity
        self._gap_counter_vad_chunks = 0 # Counts VAD-sized chunks of silence

        self.input_stream: sd.InputStream | None = None
        
        # Warm up ASR
        try:
            from ..utils.resources import resource_path  # Adjust path
            self.asr_model.transcribe_file(resource_path("data/0.wav"))
            logger.info("AudioInputHandler: ASR model warmed up.")
        except Exception as e:
            logger.warning(f"AudioInputHandler: Could not warm up ASR model: {e}")


    def _audio_callback_for_sd_input_stream(
        self, indata: NDArray[np.float32], frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        if status:
            logger.warning(f"AudioInputHandler: Microphone callback status: {status}")
        data = np.array(indata).copy().squeeze()
        try:
            vad_confidence_val = self.vad_model(np.expand_dims(data, 0)) # VAD model expects batch
            is_speech = bool(vad_confidence_val > self.vad_threshold)
            self._raw_sample_queue.put((data, is_speech))
        except Exception as e:
            logger.exception(f"AudioInputHandler: Error in audio callback: {e}")


    def _wakeword_detected(self, text: str) -> bool:
        # Copied from Glados._wakeword_detected
        if not self.wake_word: # Should not happen if called, but defensive
            return True # No wake word configured, so always "detected"
        
        words = text.lower().split() # Ensure lowercase for comparison
        # Levenshtein distance is case sensitive by default, ensure wake_word is also lower
        wake_word_lower = self.wake_word.lower()
        
        # Check if any word is close enough
        for word in words:
            if distance(word, wake_word_lower) < self.similarity_threshold:
                return True
        return False

    def _reset_recording_state(self) -> None:
        logger.debug("AudioInputHandler: Resetting recording state...")
        self._recording_activated = False
        self._samples_for_asr.clear()
        self._gap_counter_vad_chunks = 0
        with self._pre_activation_buffer.mutex:
            self._pre_activation_buffer.queue.clear()

    def _perform_asr(self) -> str:
        # Copied from Glados.asr
        if not self._samples_for_asr:
            return ""
        
        audio_concat = np.concatenate(self._samples_for_asr)
        # Normalize audio to prevent clipping and ensure consistent levels for ASR
        max_abs = np.max(np.abs(audio_concat))
        if max_abs > 0: # Avoid division by zero if audio is pure silence
            audio_normalized = audio_concat / max_abs / 2.0 # Normalize to [-0.5, 0.5]
        else:
            audio_normalized = audio_concat # Already zero or very small

        try:
            detected_text = self.asr_model.transcribe(audio_normalized)
            return detected_text
        except Exception as e:
            logger.exception(f"AudioInputHandler: Error during ASR transcription: {e}")
            return ""


    def _process_detected_audio(self) -> None:
        """Processes the accumulated audio, performs ASR, and sends to LLM if valid."""
        logger.debug("AudioInputHandler: Pause detected after speech. Processing audio...")
        
        detected_text = self._perform_asr()

        if detected_text.strip():
            logger.success(f"AudioInputHandler: ASR result: '{detected_text}'")

            if self.wake_word and not self._wakeword_detected(detected_text):
                logger.info(f"AudioInputHandler: Wake word '{self.wake_word}' not detected in '{detected_text}'. Ignoring.")
            else:
                if not self.wake_word:
                    logger.info("AudioInputHandler: No wake word, processing detected speech.")
                else:
                    logger.info(f"AudioInputHandler: Wake word detected in '{detected_text}'.")
                
                # Signal that a new LLM/TTS cycle is starting
                self.processing_active_event.set()
                self.llm_input_queue.put(detected_text)
                # The currently_speaking_event will be set by AudioPlayer when it starts playing
        else:
            logger.info("AudioInputHandler: ASR produced empty or whitespace-only text.")
            # If ASR is empty, we don't start a new processing cycle.
            # Ensure processing_active_event remains cleared if it was cleared by an interruption.
            # If it was already set (e.g. from a previous successful command that just finished),
            # it should stay set to allow the system to naturally go idle or handle next input.
            # This part needs care with the state machine later. For now, if ASR is empty,
            # we effectively go back to listening without explicitly clearing processing_active_event
            # unless an interruption *just* happened.

        self._reset_recording_state()


    def _handle_audio_sample(self, sample: NDArray[np.float32], is_speech: bool) -> None:
        # Logic from Glados._handle_audio_sample, _manage_pre_activation_buffer, _process_activated_audio
        if not self._recording_activated: # ---- Managing pre-activation buffer ----
            if self._pre_activation_buffer.full():
                self._pre_activation_buffer.get_nowait() # Discard oldest
            self._pre_activation_buffer.put(sample)

            if is_speech:
                # Voice activity detected for the first time (or after reset)
                if self.interruptible and self.currently_speaking_event.is_set():
                    logger.info("AudioInputHandler: User speech detected while assistant is speaking. Interrupting!")
                    # Signal other components to stop. AudioPlayer's callback will see this.
                    # LLMProcessor's loop will see this.
                    self.processing_active_event.clear()
                    # The AudioPlayer should also stop its stream if this happens.
                    # This might involve the orchestrator explicitly calling stop on AudioPlayer if needed,
                    # but processing_active_event.clear() should be the primary signal.

                # Start "recording" phase
                self._samples_for_asr = list(self._pre_activation_buffer.queue)
                self._recording_activated = True
                self._gap_counter_vad_chunks = 0 # Reset gap counter for the new utterance
                logger.debug("AudioInputHandler: VAD activated recording.")

        else: # ---- Processing activated audio ----
            self._samples_for_asr.append(sample)

            if not is_speech:
                self._gap_counter_vad_chunks += 1
                # Calculate pause limit in terms of VAD chunks
                pause_limit_vad_chunks = self.pause_limit_ms // self.vad_sample_size_ms
                if self._gap_counter_vad_chunks >= pause_limit_vad_chunks:
                    self._process_detected_audio() # This also resets recording state
            else: # Speech continues
                self._gap_counter_vad_chunks = 0


    def run(self) -> None:
        """Main loop for the AudioInputHandler thread."""
        logger.info("AudioInputHandler thread started. Initializing microphone...")
        try:
            block_size_frames = int(self.sample_rate * self.vad_sample_size_ms / 1000)
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback_for_sd_input_stream,
                blocksize=block_size_frames,
                device=None # Default device
            )
            with self.input_stream: # Starts the stream
                logger.success("AudioInputHandler: Microphone initialized and listening.")
                while not self.shutdown_event.is_set():
                    try:
                        sample_data, is_speech_confidence = self._raw_sample_queue.get(timeout=0.1) # Timeout to check shutdown
                        self._handle_audio_sample(sample_data, is_speech_confidence)
                    except queue.Empty:
                        # This is normal, allows checking shutdown_event
                        if self._recording_activated and not self._samples_for_asr:
                            # Edge case: recording was activated but then immediately an error or
                            # quick reset happened, leaving it activated with no samples. Reset.
                            logger.debug("AudioInputHandler: Recording was active with no samples, resetting.")
                            self._reset_recording_state()
                        continue
                    except Exception as e:
                        logger.exception(f"AudioInputHandler: Error processing raw sample: {e}")
                        self._reset_recording_state() # Reset on error to be safe
                        time.sleep(0.1)


        except sd.PortAudioError as e:
            logger.error(f"AudioInputHandler: PortAudioError initializing microphone: {e}. This component will not function.")
        except Exception as e:
            logger.exception(f"AudioInputHandler: Failed to start or manage microphone stream: {e}")
        finally:
            if self.input_stream and self.input_stream.active:
                self.input_stream.stop(ignore_errors=True)
                self.input_stream.close(ignore_errors=True)
            logger.info("AudioInputHandler thread finished.")