import queue
import threading
from typing import Any

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd  # type: ignore

from . import VAD


class SoundDeviceAudioIO:
    """Audio I/O implementation using sounddevice for both input and output.

    This class provides an implementation of the AudioIO interface using the
    sounddevice library to interact with system audio devices. It handles
    real-time audio capture with voice activity detection and audio playback.
    """

    SAMPLE_RATE: int = 16000  # Sample rate for input stream
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection

    def __init__(self, vad_threshold: float | None = None) -> None:
        """Initialize the sounddevice audio I/O.

        Args:
            vad_threshold: Threshold for VAD detection (default: 0.8)

        Raises:
            ImportError: If the sounddevice module is not available
            ValueError: If invalid parameters are provided
        """
        if vad_threshold is None:
            self.vad_threshold = self.VAD_THRESHOLD
        else:
            self.vad_threshold = vad_threshold

        if not 0 <= self.vad_threshold <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")

        self._vad_model = VAD()

        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()
        self.input_stream: sd.InputStream | None = None
        self._is_playing = False
        self._playback_thread = None
        self._stop_event = threading.Event()

    def start_listening(self) -> None:
        """Start capturing audio from the system microphone.

        Creates and starts a sounddevice InputStream that continuously captures
        audio from the default input device. Each audio chunk is processed with
        the VAD model and placed in the sample queue.

        Raises:
            RuntimeError: If the audio input stream cannot be started
            sd.PortAudioError: If there's an issue with the audio hardware
        """
        if self.input_stream is not None:
            self.stop_listening()

        def audio_callback(
            indata: NDArray[np.float32],
            frames: int,
            time: sd.CallbackStop,
            status: sd.CallbackFlags,
        ) -> None:
            """Process incoming audio data and put it in the queue with VAD confidence.

            Parameters:
                indata: Input audio data from the sounddevice stream
                frames: Number of audio frames in the current chunk
                time: Timing information for the audio callback
                status: Status flags for the audio callback

            Notes:
                - Copies and squeezes the input data to ensure single-channel processing
                - Applies voice activity detection to determine speech presence
                - Puts processed audio samples and VAD confidence into a thread-safe queue
            """
            if status:
                # Log any errors for debugging
                logger.debug(f"Audio callback status: {status}")

            data = np.array(indata).copy().squeeze()  # Reduce to single channel if necessary
            vad_value = self._vad_model(np.expand_dims(data, 0))
            vad_confidence = vad_value > self.vad_threshold
            self._sample_queue.put((data, bool(vad_confidence)))

        try:
            self.input_stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                callback=audio_callback,
                blocksize=int(self.SAMPLE_RATE * self.VAD_SIZE / 1000),
            )
            self.input_stream.start()
        except sd.PortAudioError as e:
            raise RuntimeError(f"Failed to start audio input stream: {e}") from e

    def stop_listening(self) -> None:
        """Stop capturing audio and clean up resources.

        Stops the input stream if it's active and releases associated resources.
        This method should be called when audio input is no longer needed or
        before application shutdown.
        """
        if self.input_stream is not None:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception as e:
                logger.error(f"Error stopping input stream: {e}")
            finally:
                self.input_stream = None

    def start_speaking(self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = "") -> None:
        """Play audio through the system speakers.

        Parameters:
            audio_data: The audio data to play as a numpy float32 array
            sample_rate: The sample rate of the audio data in Hz
            text: Optional text associated with the audio (not used by this implementation)

        Raises:
            RuntimeError: If audio playback cannot be initiated
            ValueError: If audio_data is empty or not a valid numpy array
        """
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            raise ValueError("Invalid audio data")

        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        # Stop any existing playback
        self.stop_speaking()

        # Reset the stop event
        self._stop_event.clear()

        logger.debug(f"Playing audio with sample rate: {sample_rate} Hz, length: {len(audio_data)} samples")
        self._is_playing = True
        sd.play(audio_data, sample_rate)

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """
        Monitor audio playback progress and return completion status with interrupt detection.

        Streams audio samples through PortAudio and actively tracks the number of samples
        that have been played. The playback can be interrupted by setting self.processing
        to False or self.shutdown_event. Uses a non-blocking callback system with a completion event for
        synchronization.

        Args:
            total_samples (int): Total number of samples in the audio data being played.
        Returns:
            tuple[bool, int]: A tuple containing:
                - bool: True if playback was interrupted, False if completed normally
                - int: Percentage of audio played (0-100)
        """
        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        interrupted = False
        progress = 0
        completion_event = threading.Event()

        def stream_callback(
            outdata: NDArray[np.float32], frames: int, time: dict[str, Any], status: sd.CallbackFlags
        ) -> None:
            nonlocal progress, interrupted
            progress += frames
            if self._is_playing is False:
                interrupted = True
                completion_event.set()
            if progress >= total_samples:
                completion_event.set()
            outdata.fill(0)

        try:
            logger.debug(f"Using sample rate: {sample_rate} Hz, total samples: {total_samples}")
            stream = sd.OutputStream(
                callback=stream_callback,
                samplerate=sample_rate,
                channels=1,
                finished_callback=completion_event.set,
            )
            with stream:
                # Add a reasonable maximum timeout to prevent indefinite blocking
                max_timeout = total_samples / sample_rate
                completed = completion_event.wait(max_timeout + 1)  # Add a small buffer to ensure completion
                if not completed:
                    # If the event timed out, force interruption
                    self._is_playing = False
                    interrupted = True
                    logger.debug("Audio playback timed out, forcing interruption")

        except (sd.PortAudioError, RuntimeError):
            logger.debug("Audio stream already closed or invalid")

        percentage_played = min(int(progress / total_samples * 100), 100)
        return interrupted, percentage_played

    def check_if_speaking(self) -> bool:
        """Check if audio is currently being played.

        Returns:
            bool: True if audio is currently playing, False otherwise
        """
        return self._is_playing

    def stop_speaking(self) -> None:
        """Stop audio playback and clean up resources.

        Interrupts any ongoing audio playback and waits for the playback thread
        to terminate. This ensures clean resource management and prevents
        multiple overlapping playbacks.
        """
        if self._is_playing:
            self._stop_event.set()
            sd.stop()

            self._is_playing = False

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence.

        Returns:
            queue.Queue: A thread-safe queue containing tuples of
                        (audio_sample, vad_confidence)
        """
        return self._sample_queue
