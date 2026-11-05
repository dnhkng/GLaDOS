"""Hand microphone and speaker over to a remote browser (e.g. a phone) while it is connected.

``RemoteSwitchAudioIO`` wraps the local backend. With no remote client attached it is a
transparent pass-through. When a browser attaches through the webapp, its microphone feeds
the engine and GLaDOS speaks through it; local capture is discarded until it disconnects.
"""

from __future__ import annotations

from collections.abc import Callable
import queue
import threading
import time
from typing import Any

from loguru import logger
import numpy as np
from numpy.typing import NDArray

from .base import AudioIO
from .vad import VAD

SAMPLE_RATE = 16000
VAD_WINDOW = SAMPLE_RATE * 32 // 1000  # 512 samples, as the local backends use
QUEUE_MAX_CHUNKS = 256


class RemoteClient:
    """One attached browser: how to send it frames, plus its own VAD context."""

    def __init__(self, label: str, send_text: Callable[[str], None], send_binary: Callable[[bytes], None]) -> None:
        self.label = label
        self.send_text = send_text
        self.send_binary = send_binary
        self.since = time.time()
        self.vad = VAD()
        self.pending = np.empty((0,), dtype=np.float32)


class RemoteSwitchAudioIO(AudioIO):
    """Route audio to the local backend, or to the most recently attached remote browser."""

    def __init__(self, local: AudioIO, vad_threshold: float = 0.8) -> None:
        self.local = local
        self.vad_threshold = vad_threshold
        self._queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue(maxsize=QUEUE_MAX_CHUNKS)
        self._lock = threading.Lock()
        self._client: RemoteClient | None = None
        self._listening = False
        self._discontinuity = threading.Event()
        # Remote playback state for the current track.
        self._track = 0
        self._remote_track: int | None = None
        self._remote_started = 0.0
        self._remote_duration = 0.0
        self._remote_interrupted = False
        self._remote_finished = threading.Event()
        self._remote_finished.set()
        self._closed = threading.Event()
        self._pump = threading.Thread(target=self._pump_local, name="RemoteSwitchPump", daemon=True)
        self._pump.start()

    # ------------------------------------------------------------ remote side
    def attach(
        self, label: str, send_text: Callable[[str], None], send_binary: Callable[[bytes], None]
    ) -> RemoteClient:
        """Make this browser the microphone and speaker; a previously attached one is released."""
        client = RemoteClient(label, send_text, send_binary)
        with self._lock:
            previous, self._client = self._client, client
            self._switched()
        self.local.stop_speaking()
        if previous is not None:
            try:
                previous.send_text("replaced")
            except OSError:
                pass
        logger.info("Remote audio attached: {}", label)
        return client

    def detach(self, client: RemoteClient) -> None:
        """Return audio to the local backend if ``client`` is still the active one."""
        with self._lock:
            if self._client is not client:
                return
            self._client = None
            self._switched()
            self._finish_remote(interrupted=True)
        logger.info("Remote audio detached: {}", client.label)

    def feed(self, client: RemoteClient, pcm16: bytes) -> None:
        """Queue 16 kHz mono int16 microphone audio from ``client``."""
        if len(pcm16) % 2:
            return
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        client.pending = np.concatenate((client.pending, samples))
        while len(client.pending) >= VAD_WINDOW:
            window, client.pending = client.pending[:VAD_WINDOW], client.pending[VAD_WINDOW:]
            voice = bool(client.vad(np.expand_dims(window, 0)) > self.vad_threshold)
            with self._lock:
                if self._client is not client or not self._listening:
                    continue
                # Half duplex: the phone's own speaker must not read as the user talking.
                if not self._remote_finished.is_set():
                    voice = False
            self._enqueue(window, voice)

    def played(self, client: RemoteClient, track: int) -> None:
        """The browser finished playing ``track``."""
        with self._lock:
            if self._client is client and self._remote_track == track:
                self._finish_remote(interrupted=False)

    def remote_status(self) -> dict[str, Any]:
        with self._lock:
            client = self._client
            return {
                "active": client is not None,
                "label": client.label if client else None,
                "since": client.since if client else None,
            }

    # ------------------------------------------------------------- AudioIO API
    def start_listening(self) -> None:
        self._listening = True
        self.local.start_listening()

    def stop_listening(self) -> None:
        self._listening = False
        self.local.stop_listening()

    def start_speaking(
        self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = "", **kwargs: object
    ) -> None:
        with self._lock:
            client = self._client
            if client is None:
                self._remote_track = None
            else:
                self._track += 1
                self._remote_track = self._track
                rate = sample_rate or SAMPLE_RATE
                self._remote_started = time.monotonic()
                self._remote_duration = len(audio_data) / rate
                self._remote_interrupted = False
                self._remote_finished.clear()
                track = self._track
        if client is None:
            self.local.start_speaking(audio_data, sample_rate, text, **kwargs)
            return
        try:
            # int16 halves the Wi-Fi traffic of float32; the new verb keeps stale pages from misreading it.
            client.send_text(f"play16:{track}:{rate}")
            pcm = np.clip(np.asarray(audio_data, dtype=np.float32), -1.0, 1.0) * 32767.0
            client.send_binary(pcm.astype("<i2").tobytes())
        except OSError:
            self.detach(client)

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        with self._lock:
            remote = self._remote_track is not None
            duration = self._remote_duration
        if not remote:
            return self.local.measure_percentage_spoken(total_samples, sample_rate)
        # Generous margin for network latency; a silent browser is treated as having finished.
        if not self._remote_finished.wait(duration + 3.0):
            with self._lock:
                self._finish_remote(interrupted=False)
        with self._lock:
            interrupted = self._remote_interrupted
            elapsed = time.monotonic() - self._remote_started
            self._remote_track = None
        if not interrupted:
            return False, 100
        return True, min(100, int(elapsed * 100 / duration)) if duration else 0

    def check_if_speaking(self) -> bool:
        return not self._remote_finished.is_set() or self.local.check_if_speaking()

    def stop_speaking(self) -> None:
        with self._lock:
            client = self._client if not self._remote_finished.is_set() else None
            self._finish_remote(interrupted=True)
        if client is not None:
            try:
                client.send_text("stop")
            except OSError:
                pass
        self.local.stop_speaking()

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        return self._queue

    def consume_capture_discontinuity(self) -> bool:
        if self._discontinuity.is_set():
            self._discontinuity.clear()
            return True
        consume = getattr(self.local, "consume_capture_discontinuity", None)
        return bool(consume and self._client is None and consume() is True)

    def ensure_listening(self) -> bool:
        ensure = getattr(self.local, "ensure_listening", None)
        return bool(ensure and ensure())

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            client, self._client = self._client, None
            self._finish_remote(interrupted=True)
        if client is not None:
            try:
                client.send_text("closed")
            except OSError:
                pass
        self.local.close()

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - forwards arbitrary backend extras
        # Device selection, capture health, etc. still describe the local hardware.
        if name == "local":
            raise AttributeError(name)
        return getattr(self.local, name)

    # --------------------------------------------------------------- internals
    def _switched(self) -> None:
        """Never splice audio from two sources into one utterance. Caller holds the lock."""
        self._discontinuity.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _finish_remote(self, interrupted: bool) -> None:
        """Caller holds the lock."""
        if not self._remote_finished.is_set():
            self._remote_interrupted = interrupted
            self._remote_finished.set()

    def _enqueue(self, window: NDArray[np.float32], voice: bool) -> None:
        try:
            self._queue.put_nowait((window, voice))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._discontinuity.set()
            try:
                self._queue.put_nowait((window, voice))
            except queue.Full:
                pass

    def _pump_local(self) -> None:
        source = self.local.get_sample_queue()
        while not self._closed.is_set():
            try:
                item = source.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._client is None:
                self._enqueue(*item)
