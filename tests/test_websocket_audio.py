import asyncio
import queue
import socket
import threading
import time
from unittest.mock import MagicMock
import uuid

import numpy as np
import pytest
import websockets

from glados.audio_io.websocket_io import AudioData, MicState, WebsocketAudioIO


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _playback_backend(*, completed: bool, interrupted: bool) -> WebsocketAudioIO:
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
    backend = _playback_backend(completed=False, interrupted=False)

    assert backend.measure_percentage_spoken(100, 100) == (True, 0)
    assert not backend.check_if_speaking()
    assert backend._stop_playback
    assert backend._audio_data is not None
    assert backend._audio_data.track_id is None
    backend._playback_finished_event.set.assert_called_once()


def test_interrupted_percentage_starts_at_scheduled_play_time() -> None:
    backend = _playback_backend(completed=True, interrupted=True)

    interrupted, percentage = backend.measure_percentage_spoken(100, 100)

    assert interrupted
    assert 49 <= percentage <= 55


def test_close_releases_port_for_next_backend() -> None:
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
    port = _unused_port()
    backend = WebsocketAudioIO(options={"port": port, "speaker_sync_delay_ms": 0})

    async def play_track() -> tuple[bool, int]:
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
    async def fail_to_serve(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("bad WebSocket configuration")

    monkeypatch.setattr("glados.audio_io.websocket_io.websockets.serve", fail_to_serve)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="bad WebSocket configuration"):
        WebsocketAudioIO()
    assert time.monotonic() - started < 1


def test_malformed_microphone_frame_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWebsocket:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.messages: list[str | bytes] = [b"not-float32"]

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def recv(self) -> str | bytes:
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
        backend._mic_state_lock = asyncio.Lock()
        with pytest.raises(asyncio.CancelledError):
            await backend._server_microphone(websocket)  # type: ignore[arg-type]

    asyncio.run(run_handler())

    assert websocket.sent == ["sampleRate:16000"]
    assert backend.get_sample_queue().empty()
