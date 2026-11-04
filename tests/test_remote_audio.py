"""A browser (phone) taking over microphone and speaker through the HTTPS console."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import queue
import ssl
import threading
import time
from typing import Any

import numpy as np
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from glados.audio_io.remote_io import VAD_WINDOW, RemoteSwitchAudioIO
from glados.webapp.config import console_hosts
from glados.webapp.server import WebappServer
from glados.webapp.tls import ensure_certificate, server_context


class _LocalAudio:
    def __init__(self) -> None:
        self.queue: queue.Queue[tuple[np.ndarray, bool]] = queue.Queue()
        self.spoken: list[int] = []
        self.stops = 0

    def start_listening(self) -> None: ...
    def stop_listening(self) -> None: ...

    def start_speaking(self, audio_data: np.ndarray, sample_rate: int | None = None, text: str = "") -> None:
        self.spoken.append(len(audio_data))

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        return False, 100

    def check_if_speaking(self) -> bool:
        return False

    def stop_speaking(self) -> None:
        self.stops += 1

    def get_sample_queue(self) -> queue.Queue[tuple[np.ndarray, bool]]:
        return self.queue

    def device_snapshot(self) -> dict[str, Any]:
        return {"available": True}

    def close(self) -> None: ...


class _Sink:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.binary: list[bytes] = []


def _client(audio: RemoteSwitchAudioIO) -> tuple[Any, _Sink]:
    sink = _Sink()
    return audio.attach("Phone · test", sink.text.append, sink.binary.append), sink


def _drain(q: queue.Queue) -> list[tuple[np.ndarray, bool]]:
    items = []
    while True:
        try:
            items.append(q.get(timeout=0.4))
        except queue.Empty:
            return items




def test_remote_client_replaces_local_microphone_and_speaker() -> None:
    local = _LocalAudio()
    audio = RemoteSwitchAudioIO(local)
    audio.start_listening()
    local.queue.put((np.zeros(VAD_WINDOW, np.float32), False))
    assert len(_drain(audio.get_sample_queue())) == 1, "Without a phone the PC microphone passes through"
    assert audio.device_snapshot() == {"available": True}

    client, sink = _client(audio)
    assert audio.remote_status()["active"] and audio.consume_capture_discontinuity()
    local.queue.put((np.zeros(VAD_WINDOW, np.float32), False))
    audio.feed(client, b"\0\0" * (VAD_WINDOW * 2 + 100))
    samples = _drain(audio.get_sample_queue())
    assert len(samples) == 2 and all(len(s) == VAD_WINDOW for s, _ in samples), "Only phone audio, in VAD windows"
    assert len(client.pending) == 100

    speech = np.full(1600, 0.1, np.float32)
    audio.start_speaking(speech, 16000)
    assert sink.text[-1] == "play16:1:16000" and len(sink.binary[-1]) == 1600 * 2
    assert audio.check_if_speaking() and not local.spoken
    audio.played(client, 1)
    assert audio.measure_percentage_spoken(1600, 16000) == (False, 100)

    audio.start_speaking(speech, 16000)
    audio.stop_speaking()
    assert sink.text[-1] == "stop"
    interrupted, _ = audio.measure_percentage_spoken(1600, 16000)
    assert interrupted and not audio.check_if_speaking()

    audio.detach(client)
    assert not audio.remote_status()["active"]
    audio.start_speaking(speech, 16000)
    assert local.spoken == [1600], "After the phone leaves, GLaDOS speaks on the PC again"
    audio.close()


def test_phone_voice_is_detected_but_muted_while_glados_speaks_on_it() -> None:
    audio = RemoteSwitchAudioIO(_LocalAudio())
    audio.start_listening()
    client, _ = _client(audio)
    client.vad = lambda window: np.float32(0.99)  # Every window is speech.
    speech = b"\1\0" * VAD_WINDOW * 4
    audio.feed(client, speech)
    assert all(voice for _, voice in _drain(audio.get_sample_queue())), "Phone speech reaches the listener"
    audio.start_speaking(np.zeros(16000, np.float32), 16000)
    audio.feed(client, speech)
    assert not any(voice for _, voice in _drain(audio.get_sample_queue())), "No self-hearing through the phone"
    audio.close()


def test_newest_device_wins_and_silent_browser_counts_as_played() -> None:
    audio = RemoteSwitchAudioIO(_LocalAudio())
    first, first_sink = _client(audio)
    second, _ = _client(audio)
    assert first_sink.text == ["replaced"]
    audio.detach(first)
    assert audio.remote_status()["active"], "A stale device cannot hand audio back"
    audio.start_speaking(np.zeros(160, np.float32), 16000)
    started = time.monotonic()
    assert audio.measure_percentage_spoken(160, 16000) == (False, 100)
    assert time.monotonic() - started < 4
    audio.detach(second)
    audio.close()


class _Engine:
    def __init__(self, audio: RemoteSwitchAudioIO) -> None:
        self.audio_io = audio
        self.shutdown_event = threading.Event()


@pytest.fixture
def https_console(tmp_path: Path) -> Iterator[tuple[RemoteSwitchAudioIO, str, ssl.SSLContext, int]]:
    hosts = console_hosts("127.0.0.1", [])
    cert, key = ensure_certificate(hosts, tmp_path)
    assert ensure_certificate(hosts, tmp_path) == (cert, key)
    audio = RemoteSwitchAudioIO(_LocalAudio())
    audio.start_listening()
    server = WebappServer(_Engine(audio), host="127.0.0.1", port=0, tls=server_context(cert, key))
    server.start()
    assert server.url.startswith("https://")
    client_tls = ssl.create_default_context(cafile=cert)
    yield audio, f"wss://127.0.0.1:{server.bound_port}/api/audio/remote", client_tls, server.bound_port
    server.shutdown()
    audio.close()


def test_phone_attaches_over_https_websocket_and_detaches_on_close(https_console: tuple) -> None:
    audio, url, client_tls, port = https_console
    with connect(url, ssl=client_tls, origin=f"https://127.0.0.1:{port}",
                 user_agent_header="Mozilla/5.0 (Linux; Android 15) Mobile") as ws:
        deadline = time.monotonic() + 3
        while not audio.remote_status()["active"] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert audio.remote_status()["label"] == "Phone · 127.0.0.1"
        ws.send(b"\0\0" * VAD_WINDOW)
        assert len(audio.get_sample_queue().get(timeout=2)[0]) == VAD_WINDOW
        audio.start_speaking(np.full(320, 0.2, np.float32), 16000)
        assert ws.recv(timeout=2) == "play16:1:16000"
        pcm = np.frombuffer(ws.recv(timeout=2), "<i2")
        assert len(pcm) == 320 and abs(pcm[0] / 32768 - 0.2) < 1e-3
        ws.send("played:1")
        assert audio.measure_percentage_spoken(320, 16000) == (False, 100)
    deadline = time.monotonic() + 3
    while audio.remote_status()["active"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not audio.remote_status()["active"], "Closing the page returns audio to the PC"


def test_remote_audio_rejects_cross_origin_pages(https_console: tuple) -> None:
    audio, url, client_tls, _ = https_console
    with pytest.raises(InvalidStatus, match="403"):
        connect(url, ssl=client_tls, origin="https://evil.example")
    assert not audio.remote_status()["active"]


def test_auto_host_follows_this_machines_current_address() -> None:
    import http.client

    from glados.webapp.config import local_addresses

    addresses = sorted(a for a in local_addresses() if ":" not in a)
    if not addresses:
        pytest.skip("No IPv4 network route on this machine")
    server = WebappServer(_Engine(RemoteSwitchAudioIO(_LocalAudio())), host="0.0.0.0", port=0, allowed_hosts=["auto"])
    server.start()
    try:
        for host, expected in ((addresses[0], 404), ("203.0.113.9", 421)):
            connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
            connection.request("GET", "/missing", headers={"Host": f"{host}:{server.bound_port}"})
            assert connection.getresponse().status == expected, host
            connection.close()
    finally:
        server.shutdown()
        server.engine.audio_io.close()
