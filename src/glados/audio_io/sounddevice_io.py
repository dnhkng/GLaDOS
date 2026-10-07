"""Local microphone and speaker backend implemented with sounddevice."""

import queue
import sys
import threading
import time
from typing import Any

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd  # type: ignore

from . import VAD
from .base import AudioIO
from .resample import resample as resample_audio


class SoundDeviceAudioIO(AudioIO):
    """Audio I/O implementation using sounddevice for both input and output.

    This class provides an implementation of the AudioIO interface using the
    sounddevice library to interact with system audio devices. It handles
    real-time audio capture with voice activity detection and audio playback.
    """

    SAMPLE_RATE: int = 16000  # Sample rate for input stream
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection

    def __init__(self, vad_threshold: float | None = None, input_device: int | str | None = None,
                 output_device: int | str | None = None) -> None:
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

        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue(maxsize=32)
        self.input_stream: sd.InputStream | None = None
        self._is_playing = False
        self._playback_thread = None
        self._stop_event = threading.Event()
        self._pending_audio: NDArray[np.float32] | None = None
        self._pending_sample_rate: int = self.SAMPLE_RATE
        self.input_device, self.output_device = input_device, output_device
        self._pending_output_device: int | str | None = None
        self._device_lock = threading.RLock()
        self._listening_enabled = False
        self._last_callback = 0.0
        self._last_recovery_attempt = 0.0
        self._capture_error: str | None = None
        self._capture_recoveries = 0
        self._output_underflows = 0
        self._input_overflows = 0
        self._capture_discontinuity = threading.Event()

    def _resolve_device(self, device: int | str | None, direction: str) -> int | str | None:
        if device is not None or not sys.platform.startswith("linux"):
            return device
        # Follow the desktop's default source/sink through its routing service.
        # ALSA's raw default can retain a dead USB handle after a device change.
        devices = sd.query_devices()
        for name in ("pulse", "pipewire"):
            for index, row in enumerate(devices):
                if row["name"] == name and row[f"max_{direction}_channels"] > 0:
                    return index
        return None

    def capture_health(self) -> dict[str, Any]:
        age = time.monotonic() - self._last_callback if self._last_callback else None
        active = self.input_stream is not None and self.input_stream.active is not False
        return {"enabled": self._listening_enabled, "connected": active and age is not None and age < 2,
                "last_frame_age_s": round(age, 3) if age is not None else None,
                "error": self._capture_error, "recoveries": self._capture_recoveries,
                "pending_frames": self._sample_queue.qsize(), "overflows": self._input_overflows}

    def consume_capture_discontinuity(self) -> bool:
        if not self._capture_discontinuity.is_set():
            return False
        self._capture_discontinuity.clear()
        with self._sample_queue.mutex:
            self._sample_queue.queue.clear()
        return True

    def _queue_sample(self, chunk: NDArray[np.float32], speech: bool) -> None:
        try:
            self._sample_queue.put_nowait((chunk, speech))
        except queue.Full:
            self._input_overflows += 1
            self._capture_discontinuity.set()
            # Never block a real-time callback on a consumer doing inference.
            try:
                self._sample_queue.get_nowait()
                self._sample_queue.put_nowait((chunk, speech))
            except (queue.Empty, queue.Full):
                pass

    def ensure_listening(self) -> bool:
        """Called by the listener while draining/waiting; never revives an intentionally stopped input."""
        if not self._listening_enabled or self.capture_health()["connected"]:
            return False
        now = time.monotonic()
        if now - self._last_recovery_attempt < 3 or not self._device_lock.acquire(blocking=False):
            return False
        try:
            if not self._listening_enabled:
                return False
            self._last_recovery_attempt = now
            logger.warning("Microphone stream stalled; reopening selected input")
            try:
                self.start_listening()
                with self._sample_queue.mutex:
                    self._sample_queue.queue.clear()
                self._capture_recoveries += 1
                return True
            except (sd.PortAudioError, RuntimeError, ValueError) as exc:
                self._capture_error = str(exc)
                self._listening_enabled = True  # Retry when a disconnected device returns.
                logger.warning("Microphone recovery deferred: {}", exc)
                return False
        finally:
            self._device_lock.release()

    def device_snapshot(self) -> dict[str, Any]:
        devices, apis = sd.query_devices(), sd.query_hostapis()
        def choices(direction: str) -> list[dict[str, Any]]:
            return [{"id": i, "name": device["name"], "host_api": apis[device["hostapi"]]["name"]}
                    for i, device in enumerate(devices) if device[f"max_{direction}_channels"] > 0]
        with self._device_lock:
            return {"available": True, "input": choices("input"), "output": choices("output"),
                    "selected_input": self.input_device, "selected_output": self.output_device,
                    "input_health": self.capture_health(), "output_health": {"underflows": self._output_underflows}}

    def select_device(self, kind: str, device: int | None) -> None:
        if kind not in {"microphone", "speaker"} or (device is not None and (type(device) is not int or device < 0)):
            raise ValueError("Choose a listed audio device or the system default")
        with self._device_lock:
            try:
                if kind == "microphone":
                    self._input_settings(device)
                    old = self.input_device
                    if device == old and self.capture_health()["connected"]:
                        return
                    listening = self.input_stream is not None
                    self.stop_listening()
                    self.input_device = device
                    try:
                        if listening:
                            self.start_listening()
                    except Exception:
                        self.stop_listening()
                        self.input_device = old
                        if listening:
                            self.start_listening()
                        raise
                    with self._sample_queue.mutex:
                        self._sample_queue.queue.clear()
                else:
                    effective = self._resolve_device(device, "output")
                    rate = sd.query_devices(device=effective, kind="output")["default_samplerate"]
                    sd.check_output_settings(device=effective, channels=1, samplerate=rate)
                    if device != self.output_device:
                        self.stop_speaking()
                        self.output_device = device
            except (sd.PortAudioError, RuntimeError) as exc:
                raise ValueError(f"Cannot open the selected {kind}: {exc}") from exc

    def _input_settings(self, device: int | str | None) -> tuple[int, int]:
        device = self._resolve_device(device, "input")
        info = sd.query_devices(device=device, kind="input")
        native_rate = int(info["default_samplerate"])
        candidates = [(self.SAMPLE_RATE, 1), (native_rate, 1)]
        if info["max_input_channels"] >= 2:
            candidates.append((native_rate, 2))
        for rate, channels in candidates:
            try:
                sd.check_input_settings(device=device, channels=channels, samplerate=rate)
                return rate, channels
            except sd.PortAudioError:
                continue
        raise ValueError("This microphone does not support a usable mono or stereo capture format")

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
        self._listening_enabled = True
        input_rate, input_channels = self._input_settings(self.input_device)
        buffered = np.empty(0, dtype=np.float32)
        self._vad_model.reset_states()

        def audio_callback(
            indata: NDArray[np.float32],
            frames: int,
            time_info: Any,
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
            nonlocal buffered
            self._last_callback = time.monotonic()
            if status and status.input_overflow:
                self._input_overflows += 1
                self._capture_discontinuity.set()

            data = np.asarray(indata, dtype=np.float32).mean(axis=1)
            if input_rate != self.SAMPLE_RATE:
                data = resample_audio(data, input_rate, self.SAMPLE_RATE)
            buffered = np.concatenate((buffered, data))
            # Native-rate devices still feed exactly 512 mono samples to Silero.
            while len(buffered) >= 512:
                chunk, buffered = buffered[:512].copy(), buffered[512:]
                vad_value = self._vad_model(np.expand_dims(chunk, 0))
                self._queue_sample(chunk, bool(vad_value > self.vad_threshold))

        try:
            self.input_stream = sd.InputStream(
                device=self._resolve_device(self.input_device, "input"),
                samplerate=input_rate,
                channels=input_channels,
                callback=audio_callback,
                blocksize=int(input_rate * (self.VAD_SIZE / 1000 if input_rate == self.SAMPLE_RATE else 0.04)),
            )
            self.input_stream.start()
            self._last_callback = time.monotonic()
            self._capture_error = None
        except sd.PortAudioError as e:
            raise RuntimeError(f"Failed to start audio input stream: {e}") from e

    def stop_listening(self) -> None:
        """Stop capturing audio and clean up resources.

        Stops the input stream if it's active and releases associated resources.
        This method should be called when audio input is no longer needed or
        before application shutdown.
        """
        self._listening_enabled = False
        if self.input_stream is not None:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception as e:
                logger.error(f"Error stopping input stream: {e}")
            finally:
                self.input_stream = None

    def start_speaking(self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = "") -> None:
        with self._device_lock:
            self._start_speaking(audio_data, sample_rate, text)

    def _start_speaking(self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = "") -> None:
        """Queue audio for playback through the system speakers.

        Stores audio data for playback via measure_percentage_spoken(), which
        uses a single OutputStream to both play and monitor progress. This avoids
        the race condition that occurs when sd.play() and a monitoring OutputStream
        run concurrently.

        Parameters:
            audio_data: The audio data to play as a numpy float32 array
            sample_rate: The sample rate of the audio data in Hz
            text: Optional text associated with the audio (not used by this implementation)

        Raises:
            ValueError: If audio_data is empty or not a valid numpy array
        """
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            raise ValueError("Invalid audio data")

        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        # Stop any existing playback and create a fresh stop event for this session
        self.stop_speaking()
        self._stop_event = threading.Event()

        # Resample to the output device's native sample rate so PortAudio's
        # low-quality built-in sample-rate converter is never used. This avoids
        # the audible crackling/distortion that occurs when the TTS rate differs
        # from the device rate (e.g. 22050 Hz TTS out, 44100 Hz device).
        output_device = self.output_device
        try:
            output_device = self._resolve_device(self.output_device, "output")
            device_rate = int(sd.query_devices(device=output_device, kind="output")["default_samplerate"])
        except Exception as e:
            device_rate = 0
            logger.debug(f"Could not query output device sample rate: {e}")

        if device_rate > 0 and sample_rate != device_rate:
            logger.debug(f"Resampling audio {sample_rate} Hz -> {device_rate} Hz")
            audio_data = resample_audio(audio_data, sample_rate, device_rate)
            sample_rate = device_rate

        logger.debug(f"Playing audio with sample rate: {sample_rate} Hz, length: {len(audio_data)} samples")
        self._is_playing = True
        self._pending_audio = audio_data
        self._pending_sample_rate = sample_rate
        self._pending_output_device = output_device

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """
        Play queued audio and monitor playback progress with interrupt detection.

        Uses a single OutputStream to both play the audio stored by start_speaking()
        and track progress, avoiding the race condition from running sd.play() and a
        separate monitoring stream concurrently.

        Args:
            total_samples (int): Total number of samples in the audio data being played.
            sample_rate (int | None): Sample rate override; uses the value from start_speaking() if None.
        Returns:
            tuple[bool, int]: A tuple containing:
                - bool: True if playback was interrupted, False if completed normally
                - int: Percentage of audio played (0-100)
        """
        audio_data = self._pending_audio
        output_device = self._pending_output_device
        if audio_data is None:
            return False, 100

        # Prefer the sample rate stored by start_speaking() -- that reflects any
        # device-rate resampling that happened, so the stream opens at the true
        # playback rate and the timeout arithmetic stays correct.
        if self._pending_sample_rate and self._pending_sample_rate > 0:
            sample_rate = self._pending_sample_rate
        elif sample_rate is None:
            sample_rate = self._pending_sample_rate

        if sample_rate is None or sample_rate <= 0:
            logger.warning(f"Invalid sample rate {sample_rate}; skipping playback")
            if self._pending_audio is audio_data:
                self._pending_audio = None
                self._is_playing = False
            return False, 100

        # Derive playback length from the actual buffer so a wrong caller-supplied
        # total_samples can't break the timeout or percentage math.
        effective_total = len(audio_data)
        if effective_total <= 0:
            if self._pending_audio is audio_data:
                self._pending_audio = None
                self._is_playing = False
            return False, 100

        position = 0
        interrupted = False
        underflows_before = self._output_underflows
        completion_event = threading.Event()
        # Capture current stop_event so a new start_speaking() call doesn't affect this session
        stop_event = self._stop_event

        def stream_callback(
            outdata: NDArray[np.float32], frames: int, time_info: object, status: sd.CallbackFlags
        ) -> None:
            """Fill the next output block and track completion or interruption."""
            nonlocal position, interrupted
            if status and status.output_underflow:
                self._output_underflows += 1

            if stop_event.is_set():
                outdata.fill(0)
                interrupted = True
                completion_event.set()
                raise sd.CallbackStop

            remaining = effective_total - position
            chunk_size = min(frames, remaining)

            if chunk_size > 0:
                outdata[:chunk_size, 0] = audio_data[position : position + chunk_size]
                if chunk_size < frames:
                    outdata[chunk_size:].fill(0)
                position += chunk_size
            else:
                outdata.fill(0)

            if position >= effective_total:
                completion_event.set()
                raise sd.CallbackStop

        try:
            logger.debug(f"Using sample rate: {sample_rate} Hz, total samples: {effective_total}")
            max_timeout = effective_total / sample_rate + 1
            with sd.OutputStream(
                device=output_device,
                callback=stream_callback,
                samplerate=sample_rate,
                channels=1,
                latency="high",
            ):
                completed = completion_event.wait(max_timeout)
                if not completed:
                    # Timeout: signal stop and mark as interrupted
                    stop_event.set()
                    interrupted = True
                    logger.debug("Audio playback timed out, forcing interruption")

        except (sd.PortAudioError, RuntimeError):
            logger.debug("Audio stream already closed or invalid")

        # Identity-checked teardown: only clear shared state if it still belongs to this
        # session, otherwise a new start_speaking() that ran concurrently could be wiped out.
        if self._pending_audio is audio_data:
            self._pending_audio = None
        if self._stop_event is stop_event:
            self._is_playing = False
        percentage_played = min(int(position / effective_total * 100), 100)
        if self._output_underflows > underflows_before:
            logger.warning("Speech playback had {} buffer underruns", self._output_underflows - underflows_before)
        return interrupted, percentage_played

    def check_if_speaking(self) -> bool:
        """Check if audio is currently being played.

        Returns:
            bool: True if audio is currently playing, False otherwise
        """
        return self._is_playing

    def stop_speaking(self) -> None:
        """Stop audio playback and clean up resources.

        Signals the current playback session to stop by setting the stop event.
        The active OutputStream callback will detect this on its next invocation
        and raise CallbackStop to cleanly terminate the stream.
        """
        if self._is_playing:
            self._stop_event.set()
            self._is_playing = False

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence.

        Returns:
            queue.Queue: A thread-safe queue containing tuples of
                        (audio_sample, vad_confidence)
        """
        return self._sample_queue

    def close(self) -> None:
        """Release local input and output resources."""
        self.stop_speaking()
        self.stop_listening()
