"""
Speech listener module for the Glados voice assistant.

This module provides the SpeechListener class that handles audio input streaming,
voice activity detection, speech recognition, and wake word detection.
"""

from collections import deque
import queue
import threading

from Levenshtein import distance
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from ..ASR import TranscriberProtocol
from ..audio_io import AudioProtocol


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
    PAUSE_LIMIT: int = 640  # Milliseconds of pause allowed before processing
    SIMILARITY_THRESHOLD: int = 2  # Threshold for wake word similarity

    def __init__(
        self,
        audio_io: AudioProtocol,  # Replace with actual type if known
        llm_queue: queue.Queue[str],
        shutdown_event: threading.Event,
        currently_speaking_event: threading.Event,
        processing_active_event: threading.Event,
        asr_model: TranscriberProtocol,
        wake_word: str | None,
        pause_time: float,
        interruptible: bool = True,
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

        self.shutdown_event = shutdown_event
        self.currently_speaking_event = currently_speaking_event
        self.processing_active_event = processing_active_event

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
        logger.success("Audio Modules Operational")

        # Loop forever, but is 'paused' when new samples are not available
        try:
            while not self.shutdown_event.is_set():  # Check event BEFORE blocking get
                try:
                    # Use a timeout for the queue get
                    sample, vad_confidence = self._sample_queue.get(timeout=self.pause_time)
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
        self._buffer.append(sample)  # Automatically handles overflow

        if vad_confidence:
            if not self.interruptible and self.currently_speaking_event.is_set():
                logger.info("Interruption is disabled, and the assistant is currently speaking, ignoring new input.")
                return

            self.audio_io.stop_speaking()
            self.processing_active_event.clear()
            self._samples = list(self._buffer)  # Clean conversion
            self._recording_started = True

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
        closest_distance = min(distance(word.lower(), self.wake_word) for word in words)
        return closest_distance < self.SIMILARITY_THRESHOLD

    def reset(self) -> None:
        """
        Resets the internal state of the speech listener, clearing all audio buffers and counters.

        This prepares the listener for a new speech segment by:
        - Setting `_recording_started` to False.
        - Clearing the accumulated `_samples`.
        - Resetting the `_gap_counter`.
        - Emptying the pre-activation circular buffer (`_buffer.queue`), safely using its internal mutex.
        """
        logger.debug("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        self._buffer.clear()

    def _process_detected_audio(self) -> None:
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

        detected_text = self.asr(self._samples)

        if detected_text:
            logger.success(f"ASR text: '{detected_text}'")

            if self.wake_word and not self._wakeword_detected(detected_text):
                logger.info(f"Required wake word {self.wake_word=} not detected.")
            else:
                self.llm_queue.put(detected_text)
                self.processing_active_event.set()

        self.reset()

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

        detected_text = self.asr_model.transcribe(audio)
        return detected_text
