import queue
import threading
from typing import Any

import soxr

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
        self._output_stream: sd.OutputStream | None = None
        self._is_playing = False
        self._playback_position = 0
        self._playback_audio: NDArray[np.float32] = np.array([], dtype=np.float32)
        self._playback_done = threading.Event()
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
        self._stop_event.clear()

        # Ensure mono float32
        audio = np.asarray(audio_data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Resample to device native rate if needed to avoid low-quality SRC in PortAudio
        device_rate = int(sd.query_devices(kind="output")["default_samplerate"])
        if sample_rate != device_rate:
            audio = soxr.resample(audio, sample_rate, device_rate, quality="HQ")
            sample_rate = device_rate

        self._playback_audio = audio
        self._playback_sample_rate = sample_rate
        self._playback_position = 0
        self._playback_done = threading.Event()
        self._is_playing = True

        def _callback(outdata: NDArray[np.float32], frames: int, t: Any, status: sd.CallbackFlags) -> None:
            if status:
                logger.debug(f"Playback callback status: {status}")
            pos = self._playback_position
            chunk = self._playback_audio[pos : pos + frames]
            if len(chunk) < frames:
                outdata[: len(chunk), 0] = chunk
                outdata[len(chunk) :, 0] = 0
                self._playback_position += len(chunk)
                self._playback_done.set()
                raise sd.CallbackStop
            else:
                outdata[:, 0] = chunk
                self._playback_position += frames

        self._output_stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=_callback,
            finished_callback=self._playback_done.set,
        )
        logger.debug(f"Playing audio with sample rate: {sample_rate} Hz, length: {len(audio)} samples")
        self._output_stream.start()

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """
        Wait for playback to complete or be interrupted, returning progress.

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

        try:
            poll_interval = 0.01

            while True:
                if self._is_playing is False:
                    interrupted = True
                    break
                done = self._playback_done.wait(timeout=poll_interval)
                if done:
                    break

            # Wait a tiny bit to let the stream finish cleanly
            if not interrupted and hasattr(self, "_output_stream") and self._output_stream is not None:
                self._output_stream.stop()

        except Exception as e:
            logger.debug(f"measure_percentage_spoken error: {e}")

        progress = getattr(self, "_playback_position", total_samples)
        self._is_playing = False
        percentage_played = min(int(progress / total_samples * 100), 100) if total_samples > 0 else 100
        return interrupted, percentage_played

    def check_if_speaking(self) -> bool:
        """Check if audio is currently being played.

        Returns:
            bool: True if audio is currently playing, False otherwise
        """
        return self._is_playing

    def stop_speaking(self) -> None:
        """Stop audio playback and clean up resources."""
        if self._is_playing:
            self._is_playing = False
            self._stop_event.set()
            if hasattr(self, "_playback_done"):
                self._playback_done.set()
        if hasattr(self, "_output_stream") and self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence.

        Returns:
            queue.Queue: A thread-safe queue containing tuples of
                        (audio_sample, vad_confidence)
        """
        return self._sample_queue
