"""
Speech listener module for the Glados voice assistant.

This module provides the SpeechListener class that handles audio input streaming,
voice activity detection, speech recognition, and wake word detection.
"""

from collections import deque
import queue
import threading
import time
import uuid
from typing import Any, Callable

from Levenshtein import distance
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from ..autonomy.interaction_state import InteractionState
from ..ASR import TranscriberProtocol
from ..audio_io import AudioProtocol
from ..observability import ObservabilityBus, trim_message
from .audio_state import AudioState
from .native_audio import NativeAudioInput

# Callback signature: (event_type: str) -> None
InterruptCallback = Callable[[str], None]


class SpeechListener:
    """
    Manages audio input and speech processing for a voice assistant.

    This class handles capturing audio, performing Voice Activity Detection (VAD),
    buffering pre-activation audio, triggering Automatic Speech Recognition (ASR),
    and coordinating with Language Model (LLM) and Text-to-Speech (TTS) components
    via shared events and queues. It supports optional wake word detection.
    """

    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    BUFFER_SIZE: int = 800  # Milliseconds of buffer BEFORE VAD detection
    PAUSE_LIMIT: int = 416  # 13 consecutive silent VAD chunks of 32 ms
    SIMILARITY_THRESHOLD: int = 2  # Threshold for wake word similarity

    def __init__(
        self,
        audio_io: AudioProtocol,  # Replace with actual type if known
        llm_queue: queue.Queue[dict[str, Any]],
        shutdown_event: threading.Event,
        currently_speaking_event: threading.Event,
        processing_active_event: threading.Event,
        asr_model: TranscriberProtocol | None,
        wake_word: str | None,
        pause_time: float,
        interruptible: bool = True,
        interaction_state: "InteractionState | None" = None,
        observability_bus: ObservabilityBus | None = None,
        asr_muted_event: threading.Event | None = None,
        audio_state: AudioState | None = None,
        on_interrupt: InterruptCallback | None = None,
        native_audio: NativeAudioInput | None = None,
        begin_user_turn: Callable[[], int] | None = None,
        end_user_turn: Callable[[int, str], None] = lambda generation, reason: None,
        turn_is_current: Callable[[int], bool] = lambda generation: True,
    ) -> None:
        """
        Initializes the SpeechListener with audio I/O, inter-thread communication, and ASR model.

        Args:
            audio_io: An instance conforming to `AudioProtocol` for audio input/output.
            llm_queue: A queue for sending transcribed text to the language model.
            shutdown_event: A threading.Event to signal the application to shut down.
            currently_speaking_event: A threading.Event indicating if the assistant is currently speaking.
            processing_active_event: A threading.Event indicating if processing is active (e.g., for LLM/TTS).
            asr_model: An instance conforming to `TranscriberProtocol` for speech recognition.
            wake_word: Optional wake word string to activate the assistant. Defaults to None.
            interruptible: If True, allows new speech input to interrupt ongoing assistant speech.
        """
        self.audio_io = audio_io
        self.llm_queue = llm_queue
        self.asr_model = asr_model
        self.wake_word = wake_word.lower() if wake_word else None
        self.pause_time = pause_time
        self.interruptible = interruptible

        # Circular buffer to hold pre-activation samples
        self._buffer: deque[NDArray[np.float32]] = deque(maxlen=self.BUFFER_SIZE // self.VAD_SIZE)
        self._sample_queue = self.audio_io.get_sample_queue()

        # Internal state variables
        self._recording_started = False
        self._samples: list[NDArray[np.float32]] = []
        self._gap_counter = 0
        self._native_audio_overflow = False

        self.shutdown_event = shutdown_event
        self.currently_speaking_event = currently_speaking_event
        self.processing_active_event = processing_active_event
        self._interaction_state = interaction_state
        self._observability_bus = observability_bus
        self._asr_muted_event = asr_muted_event
        self._audio_state = audio_state
        self._on_interrupt = on_interrupt
        self._native_audio = native_audio
        self._last_capture_check = 0.0
        self._begin_user_turn = begin_user_turn
        self._end_user_turn = end_user_turn
        self._turn_generation: int | None = None
        self._speech_onset = 0
        self._turn_is_current = turn_is_current
        self._continuation_lock = threading.RLock()
        self._pending_voice: tuple[list[NDArray[np.float32]], int | None, str, float] | None = None
        self._voice_turn_id: str | None = None
        self._voice_continuation = False
        if native_audio and wake_word:
            raise ValueError("Native audio does not support transcript-based wake words; use Parakeet mode")

    def run(self) -> None:
        """
        Starts the main listening event loop, continuously processing audio input.

        This method initializes the audio input stream and enters a loop that
        listens for incoming audio samples and their Voice Activity Detection (VAD) confidence.
        It retrieves samples from an internal queue and processes them via `_handle_audio_sample`.
        The loop runs until the `shutdown_event` is set. It also handles brief pauses
        in audio input using a timeout.

        Raises:
            Exception: Catches and logs general exceptions encountered during the listening loop,
                       without stopping the loop unless `shutdown_event` is set.
        """
        logger.success("SpeechListener ready")

        # Loop forever, but is 'paused' when new samples are not available
        try:
            while not self.shutdown_event.is_set():  # Check event BEFORE blocking get
                try:
                    now = time.monotonic()
                    if now - self._last_capture_check >= 1:
                        self._last_capture_check = now
                        ensure = getattr(self.audio_io, "ensure_listening", None)
                        if ensure and ensure():
                            self.reset()
                            if self._observability_bus:
                                self._observability_bus.emit("audio", "recovered", "Microphone capture resumed")
                    # Use a timeout for the queue get
                    sample, vad_confidence = self._sample_queue.get(timeout=self.pause_time)
                    discontinuity = getattr(self.audio_io, "consume_capture_discontinuity", None)
                    if discontinuity and discontinuity() is True:
                        self.reset()
                        if self._observability_bus:
                            self._observability_bus.emit("audio", "gap", "Microphone gap; discarded incomplete speech", level="warning")
                        continue
                    if self._asr_muted_event and self._asr_muted_event.is_set():
                        if self._recording_started or self._samples or self._buffer or self._pending_voice:
                            self.reset()
                        continue
                    self._handle_audio_sample(sample, vad_confidence)
                except queue.Empty:
                    # Timeout occurred, loop again to check shutdown_event
                    continue
                except (OSError, RuntimeError) as e:  # More specific exceptions
                    if not self.shutdown_event.is_set():  # Only log if not shutting down
                        logger.error(f"Error in listen loop ({type(e).__name__}): {e}")
                    continue

            logger.info("Shutdown event detected in listen loop, exiting loop.")

        finally:
            self.reset()
            self.audio_io.stop_listening()
            logger.info("Listen event loop is stopping/exiting.")

        logger.info("Speech Listener thread finished.")

    def _handle_audio_sample(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Routes the processing of an individual audio sample based on the current recording state.

        If recording has not started, the sample contributes to the pre-activation buffer.
        Once recording is active, the sample is added to the main speech segment
        and contributes to the voice activity gap detection.

        Args:
            sample: The audio sample (numpy array) to process.
            vad_confidence: True if voice activity is detected in the sample, False otherwise.
        """
        if self._audio_state is not None:
            if sample.size:
                rms = float(np.sqrt(np.mean(sample * sample)))
            else:
                rms = 0.0
            self._audio_state.update(rms, vad_confidence)
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Manages the pre-activation circular buffer and handles voice activity detection.

        Samples are continuously added to a circular buffer until voice activity is detected.
        Upon VAD detection:
        - It checks for interruptibility if the assistant is currently speaking.
        - The assistant's speaking is stopped (`audio_io.stop_speaking()`).
        - The `processing_active_event` is cleared, pausing LLM/TTS activity.
        - The buffered samples are moved to `_samples`, and `_recording_started` is set to True.

        Args:
            sample: The current audio sample (numpy array) to be added to the buffer.
            vad_confidence: True if voice activity is detected in the sample, False otherwise.
        """
        with self._continuation_lock:
            if self._pending_voice:
                _, generation, _, submitted = self._pending_voice
                if (time.monotonic() - submitted > 30 or self.currently_speaking_event.is_set()
                        or (generation is not None and not self._turn_is_current(generation))):
                    self._pending_voice = None
        self._buffer.append(sample)  # Automatically handles overflow
        self._speech_onset = self._speech_onset + 1 if vad_confidence else 0

        # A single noisy 32 ms frame must not stop a reply. Keep pre-roll so
        # confirming sustained speech does not lose the beginning of a word.
        onset_frames = 5 if self.currently_speaking_event.is_set() else 3
        if self._speech_onset >= onset_frames:
            if not self.interruptible and self.currently_speaking_event.is_set():
                logger.debug(f"Detected voice activity but interruptibility is disabled: {self.interruptible=}, {self.currently_speaking_event.is_set()=}")
                return

            # Check if this is an interrupt (user speaking while GLaDOS was speaking)
            was_speaking = self.currently_speaking_event.is_set()

            self.audio_io.stop_speaking()
            self.processing_active_event.clear()
            # Invalidate the old reply before claiming its unanswered audio.
            self._turn_generation = self._begin_user_turn() if self._begin_user_turn else None
            with self._continuation_lock:
                pending, self._pending_voice = self._pending_voice, None
                self._voice_continuation = pending is not None and not was_speaking
                self._voice_turn_id = pending[2] if self._voice_continuation else uuid.uuid4().hex
                self._samples = (list(pending[0]) if self._voice_continuation else []) + list(self._buffer)
            if self._voice_continuation and self._observability_bus:
                self._observability_bus.emit("audio", "continuation",
                                            "Resumed speech; replaced unanswered turn with combined audio")
            self._recording_started = True

            if was_speaking and self._on_interrupt:
                self._on_interrupt("user_interrupt")

    def _process_activated_audio(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Accumulates audio samples and tracks pauses after voice activation.

        This method appends incoming audio samples to `self._samples`. It increments
        `_gap_counter` when no voice activity is detected. If the `_gap_counter`
        exceeds `PAUSE_LIMIT`, it signifies the end of a speech segment, triggering
        `_process_detected_audio`. Otherwise, if voice is detected, the gap counter is reset.

        Args:
            sample: A single audio sample (numpy array) from the input stream.
            vad_confidence: True if voice activity is currently detected, False otherwise.
        """
        if self._native_audio_overflow:
            self._gap_counter = 0 if vad_confidence else self._gap_counter + 1
            if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                self.reset()
            return
        if self._native_audio:
            limit = self._native_audio.config.max_duration_s * self._native_audio.SAMPLE_RATE
            if sum(len(chunk) for chunk in self._samples) + len(sample) > limit:
                # Do not answer a truncated request or accumulate unbounded audio.
                self._native_audio_overflow = True
                self._samples.clear()
                self._gap_counter = 0 if vad_confidence else self._gap_counter + 1
                warning = f"Voice input exceeded {self._native_audio.config.max_duration_s:g}s; use a shorter turn."
                logger.warning(warning)
                if self._observability_bus:
                    self._observability_bus.emit("audio", "input_too_long", warning, level="warning")
                if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                    self.reset()
                return

        self._samples.append(sample)

        if not vad_confidence:
            self._gap_counter += 1
            if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                self._process_detected_audio()
        else:
            self._gap_counter = 0

    def _wakeword_detected(self, text: str) -> bool:
        """
        Checks if the transcribed text contains a sufficiently similar match to the configured wake word.

        This method iterates through words in the `text` and calculates the Levenshtein distance
        (edit distance) between each word (converted to lowercase) and the `wake_word`.
        A match is considered found if the `closest_distance` is less than `SIMILARITY_THRESHOLD`.
        This helps account for minor misrecognitions of the wake word.

        Args:
            text: The transcribed text string to check for wake word similarity.

        Returns:
            True if a word in the text matches the wake word within the similarity threshold, False otherwise.

        Raises:
            AssertionError: If `self.wake_word` is None.
        """
        if self.wake_word is None:
            raise ValueError("Wake word should not be None")

        words = text.split()
        if not words:
            return False
        closest_distance = min(distance(word.lower(), self.wake_word) for word in words)
        return closest_distance < self.SIMILARITY_THRESHOLD

    def reset(self, *, preserve_pending: bool = False) -> None:
        """
        Resets the internal state of the speech listener, clearing all audio buffers and counters.

        This prepares the listener for a new speech segment by:
        - Setting `_recording_started` to False.
        - Clearing the accumulated `_samples`.
        - Resetting the `_gap_counter`.
        - Emptying the pre-activation circular buffer (`_buffer.queue`), safely using its internal mutex.
        """
        if not preserve_pending:
            if self._turn_generation is not None:
                self._end_user_turn(self._turn_generation, "recording_reset")
            with self._continuation_lock:
                self._pending_voice = None
        self._voice_turn_id = None
        self._voice_continuation = False
        logger.debug("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        self._buffer.clear()
        self._native_audio_overflow = False
        self._speech_onset = 0
        self._turn_generation = None
        if self._audio_state is not None:
            self._audio_state.reset()

    def response_started(self, generation: int) -> None:
        """Playback (or a text-only delivered response) seals the pending voice turn."""
        with self._continuation_lock:
            if self._pending_voice and self._pending_voice[1] == generation:
                self._pending_voice = None

    def _remember_unanswered_audio(self) -> dict[str, Any]:
        """Retain bounded PCM only in RAM; never place it in conversation history."""
        self._voice_turn_id = self._voice_turn_id or uuid.uuid4().hex
        max_samples = int((self._native_audio.config.max_duration_s if self._native_audio else 30) * 16000)
        with self._continuation_lock:
            self._pending_voice = ((list(self._samples), self._turn_generation, self._voice_turn_id, time.monotonic())
                                   if sum(len(s) for s in self._samples) <= max_samples else None)
        return {"_voice_turn_id": self._voice_turn_id, "_voice_continuation": self._voice_continuation}

    def _process_detected_audio(self) -> None:
        generation = self._turn_generation
        try:
            self._submit_detected_audio()
        except Exception:
            if generation is not None:
                self._end_user_turn(generation, "input_error")
            self.reset()
            raise
        finally:
            with self._continuation_lock:
                submitted = self._pending_voice is not None and self._pending_voice[1] == generation
            if generation is not None and not submitted:
                self._end_user_turn(generation, "no_input")

    def _submit_detected_audio(self) -> None:
        """
        Processes the accumulated audio samples once a speech pause is detected.

        This method performs the following steps:
        1. Transcribes the collected audio samples using the ASR model.
        2. If transcription is successful:
            a. Checks for the `wake_word` (if configured).
            b. If the wake word is detected (or not required), the transcribed text is
               placed into the `llm_queue`, and `processing_active_event` is set.
        3. Resets the listener's internal state using `self.reset()`, preparing for the next input.
        """
        logger.debug("Detected pause after speech. Processing...")

        if self._native_audio:
            try:
                message = self._native_audio.message(self._samples)
                if message:
                    message.update(self._remember_unanswered_audio())
                    message.update({"_enqueued_at": time.time(), "_lane": "priority", "_spoken": True})
                    if self._turn_generation is not None:
                        message["_quiet_generation"] = self._turn_generation
                    self.processing_active_event.set()
                    self.llm_queue.put(message)
                    if self._interaction_state:
                        self._interaction_state.mark_user()
                    if self._observability_bus:
                        self._observability_bus.emit("audio", "user_input", "Voice input received")
            finally:
                self.reset(preserve_pending=True)
            return

        detected_text = self.asr(self._samples)

        if detected_text:
            logger.success(f"ASR text: '{detected_text}'")

            if self.wake_word and not self._wakeword_detected(detected_text):
                logger.info(f"Required wake word {self.wake_word=} not detected.")
            else:
                if self._observability_bus:
                    self._observability_bus.emit(
                        source="asr",
                        kind="user_input",
                        message=trim_message(detected_text),
                    )
                self.processing_active_event.set()
                self.llm_queue.put(
                    {
                        **self._remember_unanswered_audio(),
                        "role": "user",
                        "content": detected_text,
                        "_enqueued_at": time.time(),
                        "_spoken": True,
                        "_lane": "priority",
                        **({"_quiet_generation": self._turn_generation} if self._turn_generation is not None else {}),
                    }
                )
                if self._interaction_state:
                    self._interaction_state.mark_user()
                self.processing_active_event.set()

        self.reset(preserve_pending=True)

    def asr(self, samples: list[NDArray[np.float32]]) -> str:
        """
        Performs Automatic Speech Recognition (ASR) on a list of audio samples.

        The samples are first concatenated into a single audio array. This combined
        audio is then normalized to a range of [-1.0, 1.0] to ensure consistent
        volume levels before being passed to the ASR model for transcription.

        Args:
            samples: A list of numpy arrays (float32) containing audio sample chunks.

        Returns:
            The transcribed text as a string.
        """
        if not samples:
            logger.warning("ASR received empty sample list")
            return ""

        audio = np.concatenate(samples)

        # Check for silent audio
        max_abs_val = np.max(np.abs(audio))
        if max_abs_val < 1e-10:  # Threshold for effectively silent audio
            logger.warning("ASR received effectively silent audio")
            return ""

        # Normalize to full range [-1.0, 1.0]
        audio = audio / max_abs_val

        if self.asr_model is None:
            raise RuntimeError("No speech recognizer configured")
        detected_text = self.asr_model.transcribe(audio)
        return detected_text
