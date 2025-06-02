import queue
import threading
from typing import Any

from Levenshtein import distance
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from ..ASR import TranscriberProtocol


class SpeechListener:
    """
    A class to handle speech input for a voice assistant, managing audio input and processing.
    This class is responsible for capturing audio samples, detecting voice activity,
    and processing speech input using automatic speech recognition (ASR) and language models (LLM).
    It provides methods to start and stop the listening event loop, manage audio samples,
    and handle wake word detection.
    """
    PAUSE_TIME: float = 0.05  # Time to wait between processing loops
    SAMPLE_RATE: int = 16000  # Sample rate for input stream
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection
    BUFFER_SIZE: int = 800  # Milliseconds of buffer BEFORE VAD detection
    PAUSE_LIMIT: int = 640  # Milliseconds of pause allowed before processing
    SIMILARITY_THRESHOLD: int = 2  # Threshold for wake word similarity
    PUNCTUATION_SET: tuple[str, ...] = (".", "!", "?", ":", ";", "?!", "\n", "\n\n")  # Sentence splitting punctuation
    NEUROTOXIN_RELEASE_ALLOWED: bool = False  # preparation for function calling, see issue #13
    DEFAULT_PERSONALITY_PREPROMPT: tuple[dict[str, str], ...] = (
        {
            "role": "system",
            "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
        },
    )

    def __init__(
        self,
        audio_io: Any,  # Replace with actual type if known
        llm_queue: queue.Queue[str],
        wake_word: str | None = None,
        asr_model: TranscriberProtocol | None = None,  # Replace with actual ASR model type
        interruptible: bool = True,
        pause_time: float = 0.1,
        pause_limit: int = 10,
        vad_size: int = 1024,
        similarity_threshold: int = 3,
        sample_rate: int = 16000,
    ) -> None:
        """
        Initialize the SpeechListener with audio input/output, LLM queue, and configuration parameters.

        Parameters:
            audio_io: An instance of an audio input/output interface for capturing and playing audio.
            llm_queue: A queue for sending transcribed text to the language model.
            wake_word: Optional wake word to trigger voice assistant activation.
            asr_model: The automatic speech recognition model used for transcribing audio.
            interruptible: Whether the assistant can be interrupted while speaking.
            pause_time: Time in seconds to wait for new audio samples before checking for shutdown.
            pause_limit: Number of silent samples before processing detected audio.
            vad_size: Size of each audio sample chunk for voice activity detection.
            similarity_threshold: Threshold for wake word similarity using Levenshtein distance.
            sample_rate: Sample rate for audio processing (default is 16000 Hz).
        """
        self.audio_io = audio_io
        self.llm_queue = llm_queue
        self.wake_word = wake_word
        self._asr_model = asr_model
        self.interruptible = interruptible

        self.PAUSE_TIME = pause_time
        self.PAUSE_LIMIT = pause_limit
        self.VAD_SIZE = vad_size
        self.SIMILARITY_THRESHOLD = similarity_threshold

        # Circular buffer to hold pre-activation samples
        self._buffer: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=self.BUFFER_SIZE // self.VAD_SIZE)
        self._sample_queue = self.audio_io._get_sample_queue()


        # Internal state variables
        self._recording_started = False
        self._samples: list[NDArray[np.float32]] = []
        self._gap_counter = 0

        # Event flags for controlling processing state
        self.processing_active_event = threading.Event()
        self.currently_speaking_event = threading.Event()
        
        # Shutdown event to stop the listening loop gracefully
        self.shutdown_event = threading.Event()


    def run(self) -> None:
        """
        Start the voice assistant's listening event loop, continuously processing audio input.

        This method initializes the audio input stream and enters an infinite loop to handle incoming audio samples.
        The loop retrieves audio samples and their voice activity detection (VAD) confidence from a queue and processes
        each sample using the `_handle_audio_sample` method.

        Behavior:
        - Starts the audio input stream
        - Logs successful initialization of audio modules
        - Enters an infinite listening loop
        - Retrieves audio samples from a queue
        - Processes each audio sample with VAD confidence
        - Handles keyboard interrupts by stopping the input stream and setting a shutdown event

        Raises:
            KeyboardInterrupt: Allows graceful termination of the listening loop
        """

        # self.input_stream.start()
        self.audio_io.start_listening()
        
        logger.success("Audio Modules Operational")
        logger.success("Listening...")

        logger.info(f"Shutdown event: {self.shutdown_event.is_set()}")
        # Loop forever, but is 'paused' when new samples are not available
        try:
            while not self.shutdown_event.is_set():  # Check event BEFORE blocking get
                try:
                    # Use a timeout for the queue get
                    sample, vad_confidence = self._sample_queue.get(timeout=self.PAUSE_TIME)
                    self._handle_audio_sample(sample, vad_confidence)
                except queue.Empty:
                    # Timeout occurred, loop again to check shutdown_event
                    continue
                except Exception as e:  # Catch other potential errors during get or handle
                    if not self.shutdown_event.is_set():  # Only log if not shutting down
                        logger.error(f"Error in listen loop: {e}")
                    continue

            logger.info("Shutdown event detected in listen loop, exiting loop.")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt in listen loop.")
        finally:
            logger.info("Listen event loop is stopping/exiting.")
            self.stop_listen_event_loop()

    def stop_listen_event_loop(self) -> None:
        """
        Stop the voice assistant's listening event loop and clean up resources.

        This method stops the audio input stream and sets the shutdown event to terminate
        any ongoing processing threads. It ensures that all resources are released properly.

        Raises:
            No explicit exceptions raised; handles cleanup gracefully.
        """
        logger.info("Setting Shutdown event")
        self.shutdown_event.set()  # Set shutdown event first

        logger.info("Calling global sd.stop() to halt all sounddevice activity.")

        # sd.stop()
        self.audio_io.stop_listening()

        logger.info("Glados engine stop sequence initiated. Threads should terminate.")

    def _handle_audio_sample(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Handles the processing of each audio sample.

        If the recording has not started, the sample is added to the circular buffer.

        If the recording has started, the sample is added to the samples list, and the pause
        limit is checked to determine when to process the detected audio.

        Args:
            sample (np.ndarray): The audio sample to process.
            vad_confidence (bool): Whether voice activity is detected in the sample.
        """
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Manages the pre-activation audio buffer and handles voice activity detection.

        This method maintains a circular buffer of audio samples before voice activation,
        discarding the oldest sample when the buffer is full. When voice activity is detected,
        it stops the audio stream and prepares for audio processing.

        Args:
            sample (np.ndarray): The current audio sample to be added to the buffer.
            vad_confidence (bool): Indicates whether voice activity is detected in the sample.

        Side Effects:
            - Modifies the internal circular buffer
            - Stops the audio stream when voice is detected
            - Disables processing on LLM and TTS threads
            - Prepares samples for recording when voice is detected
        """
        if self._buffer.full():
            self._buffer.get()  # Discard the oldest sample to make room for new ones
        self._buffer.put(sample)

        if vad_confidence:  # Voice activity detected
            if not self.interruptible and self.currently_speaking_event.is_set():
                logger.info("Interruption is disabled, and the assistant is currently speaking, ignoring new input.")
                return

            # sd.stop()  # Stop the audio stream to prevent overlap
            self.audio_io.stop_speaking()

            self.processing_active_event.clear()  # Turns off processing on threads for the LLM and TTS!!!
            self._samples = list(self._buffer.queue)
            self._recording_started = True

    def _process_activated_audio(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Process audio samples, tracking speech pauses to capture complete utterances.

        This method accumulates audio samples and monitors voice activity detection (VAD) confidence to determine
        when a complete speech segment has been captured. It appends incoming samples to the internal buffer and
        tracks silent gaps to trigger audio processing.

        Parameters:
            sample (np.ndarray): A single audio sample from the input stream
            vad_confidence (bool): Indicates whether voice activity is currently detected

        Side Effects:
            - Appends audio samples to self._samples
            - Increments or resets self._gap_counter
            - Triggers audio processing via self._process_detected_audio() when pause limit is reached
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
        Check if the detected text contains a close match to the wake word using Levenshtein distance.

        This method helps handle variations in wake word detection by calculating the minimum edit distance
        between detected words and the configured wake word. It accounts for potential misheard
        variations during speech recognition.

        Parameters:
            text (str): The transcribed text to check for wake word similarity

        Returns:
            bool: True if a word in the text is sufficiently similar to the wake word, False otherwise

        Raises:
            AssertionError: If the wake word is not configured (None)

        Notes:
            - Uses Levenshtein distance to measure text similarity
            - Compares each word in the text against the wake word
            - Considers a match if the distance is below a predefined similarity threshold
        """
        assert self.wake_word is not None, "Wake word should not be None"

        words = text.split()
        closest_distance = min([distance(word.lower(), self.wake_word) for word in words])
        return bool(closest_distance < self.SIMILARITY_THRESHOLD)

    def reset(self) -> None:
        """
        Reset the voice recording state and clear all audio buffers.

        This method performs the following actions:
        - Logs a debug message indicating the reset process
        - Stops the current recording by setting `_recording_started` to False
        - Clears the collected audio samples
        - Resets the gap counter used for detecting speech pauses
        - Empties the thread-safe audio buffer queue

        Note:
            Uses a mutex lock to safely clear the shared buffer queue to prevent
            potential race conditions in multi-threaded audio processing.
        """
        logger.debug("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        with self._buffer.mutex:
            self._buffer.queue.clear()

    def _process_detected_audio(self) -> None:
        """
        Process detected audio and generate a response after speech pause.

        Transcribes audio samples and handles wake word detection and LLM processing. Manages the
        audio input stream and processing state throughout the interaction.

        Args:
            None

        Returns:
            None

        Side Effects:
            - Stops the input audio stream
            - Performs automatic speech recognition (ASR)
            - Potentially sends text to LLM queue
            - Resets audio recording state
            - Restarts input audio stream

        Raises:
            No explicit exceptions raised
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
                self.currently_speaking_event.set()

        self.reset()

    def asr(self, samples: list[NDArray[np.float32]]) -> str:
        """
        Perform automatic speech recognition (ASR) on the provided audio samples.

        Parameters:
            samples (list[np.dtype[np.float32]]): A list of numpy arrays containing audio samples to be transcribed.

        Returns:
            str: The transcribed text from the input audio samples.

        Notes:
            - Concatenates multiple audio samples into a single continuous audio array
            - Uses the pre-configured ASR model to transcribe the audio
        """
        audio = np.concatenate(samples)

        # Normalize audio to [-0.5, 0.5] range to prevent clipping and ensure consistent levels
        audio = audio / np.max(np.abs(audio)) / 2

        detected_text = self._asr_model.transcribe(audio)
        return detected_text