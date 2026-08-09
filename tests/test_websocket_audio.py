"""Regression tests for WebSocket audio protocol and lifecycle behavior."""

import asyncio
import queue
import socket
import threading
import time
from unittest.mock import MagicMock
import uuid

import numpy as np
from numpy.typing import NDArray
import pytest
import websockets

from glados.audio_io.websocket_io import AudioData, MicState, WebsocketAudioIO


def _unused_port() -> int:
    """Reserve and return a currently unused loopback TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _playback_backend(*, completed: bool, interrupted: bool) -> WebsocketAudioIO:
    """Build a server-free backend fixture with controlled playback state."""
    backend = object.__new__(WebsocketAudioIO)
    backend._speaker_sync_delay_ms = 0
    backend._audio_lock = threading.Lock()
    backend._audio_data = AudioData(
        data=np.ones(100, dtype=np.float32),
        sample_rate=100,
        play_time=time.time(),
        play_time_monotonic=time.monotonic() - 0.5,
        track_id=uuid.uuid4(),
    )
    backend._is_playing = True
    backend._stop_playback = False
    backend._playback_was_interrupted = interrupted
    backend._playback_finished_event = MagicMock()
    backend._playback_finished_event.wait.return_value = completed
    return backend


def test_playback_timeout_clears_published_state() -> None:
    """A missing speaker acknowledgement must clear every active-track flag."""
    backend = _playback_backend(completed=False, interrupted=False)

    assert backend.measure_percentage_spoken(100, 100) == (True, 0)
    assert not backend.check_if_speaking()
    assert backend._stop_playback
    assert backend._audio_data is not None
    assert backend._audio_data.track_id is None


def test_interrupted_percentage_starts_at_scheduled_play_time() -> None:
    """Interrupted progress must exclude time spent waiting for synchronized start."""
    backend = _playback_backend(completed=True, interrupted=True)

    interrupted, percentage = backend.measure_percentage_spoken(100, 100)

    assert interrupted
    assert 49 <= percentage <= 55


def test_close_releases_port_for_next_backend() -> None:
    """Closing a backend must release its socket and terminate its thread."""
    port = _unused_port()
    first = WebsocketAudioIO(options={"port": port})
    try:
        assert first._server_thread.is_alive()
    finally:
        first.close()
    assert not first._server_thread.is_alive()

    second = WebsocketAudioIO(options={"port": port})
    second.close()
    assert not second._server_thread.is_alive()


def test_speaker_acknowledgement_completes_track() -> None:
    """A real speaker client can receive and acknowledge a complete track."""
    port = _unused_port()
    backend = WebsocketAudioIO(options={"port": port, "speaker_sync_delay_ms": 0})

    async def play_track() -> tuple[bool, int]:
        """Receive one track, acknowledge it, and return measured progress."""
        async with websockets.connect(f"ws://127.0.0.1:{port}/speaker") as websocket:
            await asyncio.sleep(0.06)
            samples = np.ones(100, dtype=np.float32)
            backend.start_speaking(samples, sample_rate=1000)

            assert str(await websocket.recv()).startswith("time:")
            assert await websocket.recv() == "sampleRate:1000"
            payload = await websocket.recv()
            assert isinstance(payload, bytes)
            assert np.array_equal(np.frombuffer(payload, dtype=np.float32), samples)

            await asyncio.sleep(0.1)
            await websocket.send("played")
            return await asyncio.to_thread(backend.measure_percentage_spoken, len(samples), 1000)

    try:
        assert asyncio.run(play_track()) == (False, 100)
    finally:
        backend.close()


def test_startup_propagates_non_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server startup must propagate configuration errors without a timeout."""

    async def fail_to_serve(*_args: object, **_kwargs: object) -> None:
        """Simulate a non-network startup failure."""
        raise RuntimeError("bad WebSocket configuration")

    monkeypatch.setattr("glados.audio_io.websocket_io.websockets.serve", fail_to_serve)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="bad WebSocket configuration"):
        WebsocketAudioIO()
    assert time.monotonic() - started < 1


def test_malformed_microphone_frame_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed float32 frames must not enter the sample queue."""

    class FakeWebsocket:
        """Provide one malformed frame and then cancel the handler."""

        def __init__(self) -> None:
            """Initialize the fake connection with one malformed frame."""
            self.sent: list[str] = []
            self.messages: list[str | bytes] = [b"not-float32"]

        async def send(self, message: str) -> None:
            """Record a server control message."""
            self.sent.append(message)

        async def recv(self) -> str | bytes:
            """Return the malformed frame, then cancel the handler."""
            if self.messages:
                return self.messages.pop(0)
            raise asyncio.CancelledError

    backend = object.__new__(WebsocketAudioIO)
    backend._default_room_tag = "office"
    backend._mic_max_silence_chunks = 10
    backend._rooms = False
    backend._is_listening = True
    backend._sample_queue = queue.Queue()
    backend._mic_state = MicState(room="office")
    backend.vad_threshold = 0.8
    websocket = FakeWebsocket()
    monkeypatch.setattr("glados.audio_io.websocket_io.VAD", MagicMock)

    async def run_handler() -> None:
        """Run the microphone handler until the fake connection cancels it."""
        backend._mic_state_lock = asyncio.Lock()
        with pytest.raises(asyncio.CancelledError):
            await backend._server_microphone(websocket)  # type: ignore[arg-type]

    asyncio.run(run_handler())

    assert websocket.sent == ["sampleRate:16000"]
    assert backend.get_sample_queue().empty()


def test_default_mode_accepts_only_one_microphone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default mode must never mix samples from concurrent microphones."""

    class FakeVAD:
        """Return deterministic speech confidence without loading ONNX."""

        def __call__(self, _samples: NDArray[np.float32]) -> float:
            """Classify every frame as speech."""
            return 1.0

        def reset_states(self) -> None:
            """Satisfy the VAD reset interface."""
            return None

    class FakeWebsocket:
        """Hold one microphone connection open after its first frame."""

        def __init__(self, samples: NDArray[np.float32], release: asyncio.Event) -> None:
            """Store one sample frame and a shared disconnect event."""
            self.samples = samples.tobytes()
            self.release = release
            self.sent_sample = False

        async def send(self, _message: str) -> None:
            """Accept the server sample-rate announcement."""
            return None

        async def recv(self) -> bytes:
            """Return one frame, then wait for the simulated disconnect."""
            if not self.sent_sample:
                self.sent_sample = True
                return self.samples
            await self.release.wait()
            raise websockets.exceptions.ConnectionClosedOK(None, None)

    backend = object.__new__(WebsocketAudioIO)
    backend._default_room_tag = "office"
    backend._mic_max_silence_chunks = 10
    backend._rooms = False
    backend._is_listening = True
    backend._sample_queue = queue.Queue()
    backend._mic_state = MicState(room="office")
    backend.vad_threshold = 0.8
    monkeypatch.setattr("glados.audio_io.websocket_io.VAD", FakeVAD)

    async def run_handlers() -> None:
        """Run two clients concurrently while the first retains ownership."""
        backend._mic_state_lock = asyncio.Lock()
        release = asyncio.Event()
        first = FakeWebsocket(np.ones(512, dtype=np.float32), release)
        second = FakeWebsocket(np.full(512, 2.0, dtype=np.float32), release)

        first_task = asyncio.create_task(backend._server_microphone(first))  # type: ignore[arg-type]
        while backend.get_sample_queue().qsize() < 1:
            await asyncio.sleep(0)

        second_task = asyncio.create_task(backend._server_microphone(second))  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        assert backend.get_sample_queue().qsize() == 1

        release.set()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run_handlers())

    samples, confidence = backend.get_sample_queue().get_nowait()
    assert np.array_equal(samples, np.ones(512, dtype=np.float32))
    assert confidence
    assert backend._mic_state.current_id is None
