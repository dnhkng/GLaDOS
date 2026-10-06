"""Device selectors change real backends and preserve working devices on failure."""

import http.client
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import sounddevice as sd

from glados.audio_io.sounddevice_io import SoundDeviceAudioIO
from glados.vision.frame_picker import CameraSampler
from glados.webapp import devices
from glados.webapp.server import WebappServer
from tests.test_webapp import _FakeEngine


@pytest.fixture
def audio(monkeypatch):
    vad = Mock(return_value=np.array([0.9], dtype=np.float32))
    monkeypatch.setattr("glados.audio_io.sounddevice_io.VAD", lambda: vad)
    rows = [dict(name="Microphone", hostapi=0, max_input_channels=2, max_output_channels=0, default_samplerate=44100),
            dict(name="Speakers", hostapi=0, max_input_channels=0, max_output_channels=2, default_samplerate=48000),
            dict(name="USB microphone", hostapi=0, max_input_channels=2, max_output_channels=0, default_samplerate=44100)]
    def query(device=None, kind=None):
        return rows if device is None and kind is None else rows[device if device is not None else (0 if kind == "input" else 1)]
    monkeypatch.setattr(sd, "query_devices", query)
    monkeypatch.setattr(sd, "query_hostapis", lambda: [{"name": "ALSA"}])
    monkeypatch.setattr(sd, "check_input_settings", Mock())
    monkeypatch.setattr(sd, "check_output_settings", Mock())
    return SoundDeviceAudioIO()


def test_audio_inventory_filters_directions_and_switches_capture_with_rollback(audio, monkeypatch):
    streams = []
    fail = False
    def stream(**kwargs):
        value = Mock()
        if fail and kwargs["device"] == 2:
            value.start.side_effect = sd.PortAudioError("Device disconnected")
        streams.append((kwargs, value))
        return value
    monkeypatch.setattr(sd, "InputStream", stream)
    state = audio.device_snapshot()
    assert [d["id"] for d in state["input"]] == [0, 2]
    assert [d["id"] for d in state["output"]] == [1]
    audio.start_listening()
    old = audio.input_stream
    audio._sample_queue.put((np.zeros(512, np.float32), False))
    audio.select_device("microphone", 0)
    assert audio.input_device == 0 and streams[-1][0]["device"] == 0
    old.close.assert_called_once()
    assert audio._sample_queue.empty()
    fail = True
    with pytest.raises(ValueError, match="disconnected"):
        audio.select_device("microphone", 2)
    assert audio.input_device == 0 and streams[-1][0]["device"] == 0
    assert audio.input_stream is streams[-1][1]
    assert audio.device_snapshot()["selected_input"] == 0


def test_native_rate_stereo_microphone_feeds_exact_mono_vad_chunks(audio, monkeypatch):
    def check(**kwargs):
        if kwargs["samplerate"] != 44100 or kwargs["channels"] != 2:
            raise sd.PortAudioError("Unsupported rate/channel count")
    monkeypatch.setattr(sd, "check_input_settings", check)
    parameters = {}
    monkeypatch.setattr(sd, "InputStream", lambda **kw: parameters.update(kw) or Mock())
    audio.select_device("microphone", 2)
    audio.start_listening()
    assert parameters["samplerate"] == 44100 and parameters["channels"] == 2
    samples = np.column_stack((np.ones(1764, np.float32)*0.2, np.ones(1764, np.float32)*0.4))
    for _ in range(4):
        parameters["callback"](samples, 1764, None, False)
    chunks = [audio._sample_queue.get_nowait() for _ in range(audio._sample_queue.qsize())]
    assert len(chunks) == 5 and all(chunk.shape == (512,) and voice for chunk, voice in chunks)
    assert all(abs(float(np.mean(chunk)) - 0.3) < 0.01 for chunk, _ in chunks)


def test_selected_speaker_controls_resampling_and_output_stream(audio, monkeypatch):
    audio.select_device("speaker", 1)
    data = np.ones(1600, np.float32)
    audio.start_speaking(data, 16000)
    assert audio._pending_sample_rate == 48000 and len(audio._pending_audio) == 4800
    parameters = {}
    class Output:
        def __init__(self, **kwargs):
            parameters.update(kwargs)
        def __enter__(self):
            try:
                parameters["callback"](np.zeros((4800, 1), np.float32), 4800, None, False)
            except sd.CallbackStop:
                pass
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(sd, "OutputStream", Output)
    assert audio.measure_percentage_spoken(1600) == (False, 100)
    assert parameters["device"] == 1 and parameters["samplerate"] == 48000
    assert parameters["latency"] == "high", "Speech playback uses driver buffering to absorb scheduler jitter"
    audio.start_speaking(data, 16000)
    audio.select_device("speaker", None)
    assert audio._stop_event.is_set() and audio.output_device is None
    with pytest.raises(ValueError):
        audio.select_device("microphone", True)


def test_stalled_or_inactive_capture_recovers_but_intentional_stop_stays_off(audio, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("glados.audio_io.sounddevice_io.time.monotonic", lambda: clock[0])
    streams = []
    def create(**kwargs):
        stream = Mock(active=True)
        streams.append(stream)
        return stream
    monkeypatch.setattr(sd, "InputStream", create)
    audio.start_listening()
    assert audio.capture_health()["connected"] and not audio.ensure_listening()
    clock[0] += 2.5
    assert not audio.capture_health()["connected"]
    assert audio.ensure_listening()
    assert len(streams) == 2 and audio.capture_health()["recoveries"] == 1
    streams[0].close.assert_called_once()
    clock[0] += 3
    audio.input_stream.active = False
    assert audio.ensure_listening() and len(streams) == 3
    audio.stop_listening()
    clock[0] += 10
    assert not audio.ensure_listening() and not audio.capture_health()["enabled"]


def test_microphone_backlog_is_bounded_and_marked_as_a_gap(audio):
    for value in range(200):
        audio._queue_sample(np.full(512, value, dtype=np.float32), True)
    assert audio.get_sample_queue().qsize() == 32
    assert audio.capture_health()["overflows"] == 168
    assert audio.consume_capture_discontinuity()
    assert audio.get_sample_queue().empty()
    assert not audio.consume_capture_discontinuity()


def test_recovery_retries_with_backoff_after_device_returns(audio, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("glados.audio_io.sounddevice_io.time.monotonic", lambda: clock[0])
    factory = Mock(return_value=Mock(active=True))
    monkeypatch.setattr(sd, "InputStream", factory)
    audio.start_listening()
    clock[0] += 3
    factory.side_effect = sd.PortAudioError("Device disconnected")
    assert not audio.ensure_listening()
    assert "disconnected" in audio.capture_health()["error"]
    calls = factory.call_count
    clock[0] += 1
    assert not audio.ensure_listening() and factory.call_count == calls
    clock[0] += 3
    factory.side_effect = None
    assert audio.ensure_listening() and audio.capture_health()["error"] is None


def test_linux_system_default_uses_desktop_routing_for_both_directions(audio, monkeypatch):
    monkeypatch.setattr("glados.audio_io.sounddevice_io.sys.platform", "linux")
    rows = [dict(name="default", max_input_channels=2, max_output_channels=2),
            dict(name="pulse", max_input_channels=2, max_output_channels=2),
            dict(name="pipewire", max_input_channels=2, max_output_channels=2)]
    monkeypatch.setattr(sd, "query_devices", lambda: rows)
    assert audio._resolve_device(None, "input") == 1
    assert audio._resolve_device(None, "output") == 1
    assert audio._resolve_device(0, "input") == 0


def test_camera_switch_releases_capture_and_discards_old_frames(monkeypatch):
    old_released, block_read, resume_read = threading.Event(), threading.Event(), threading.Event()
    old, new = Mock(), Mock()
    old.release.side_effect = old_released.set
    count = 0
    def read_old():
        nonlocal count
        count += 1
        if count > 1:
            block_read.set()
            resume_read.wait(2)
        return True, np.zeros((32, 32, 3), dtype=np.uint8)
    old.read.side_effect = read_old
    new.read.return_value = (True, np.full((32, 32, 3), 200, dtype=np.uint8))
    monkeypatch.setattr("glados.vision.frame_picker.cv2.VideoCapture", lambda spec: old if spec == 0 else new)
    sampler = CameraSampler(face_tracking=False)
    sampler.start()
    try:
        assert sampler.wait_ready(2) and block_read.wait(2)
        sampler.select_camera("/dev/video2")
        assert sampler.frames.pick()[0] is None and sampler.preview() is None
        assert not sampler.snapshot()["connected"]
        resume_read.set()
        assert sampler.wait_ready(2) and old_released.wait(1)
        assert sampler.snapshot()["device"] == "/dev/video2"
        assert sampler.frames.pick()[0].image.mean() == 200
        sampler.set_enabled(False)
        sampler.select_camera(0)
        assert not sampler.snapshot()["enabled"], "Selecting a camera preserves camera OFF"
    finally:
        resume_read.set()
        sampler.stop()


def test_device_api_validates_inventory_and_preserves_microphone_mute(audio, monkeypatch):
    engine = _FakeEngine()
    engine.audio_io = audio
    engine.asr_muted_event = threading.Event()
    engine.set_asr_muted = lambda value: engine.asr_muted_event.set() if value else engine.asr_muted_event.clear()
    camera = SimpleNamespace(snapshot=lambda: {"device": 0})
    engine.vision_agent = SimpleNamespace(camera=camera, settings=SimpleNamespace(camera_spec=0), select_camera=Mock())
    monkeypatch.setattr(devices, "camera_devices", lambda: [{"id": "/dev/video0", "name": "Webcam"}, {"id": "/dev/video2", "name": "Other"}])
    server = WebappServer(engine, port=0)
    server.start()
    def request(method, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
        try:
            conn.request(method, "/api/devices", json.dumps(body) if body else None,
                         headers or {"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()
    try:
        assert request("GET")[1]["camera"]["selected"] == "/dev/video0"
        assert request("POST", {"kind": "camera", "device": "/dev/video2"})[0] == 200
        engine.vision_agent.select_camera.assert_called_once_with("/dev/video2")
        assert request("POST", {"kind": "camera", "device": "http://arbitrary.example"})[0] == 400
        assert request("POST", {"kind": "microphone", "device": 0})[0] == 200
        assert not engine.asr_muted_event.is_set()
        engine.asr_muted_event.set()
        assert request("POST", {"kind": "microphone", "device": 2})[0] == 200
        assert engine.asr_muted_event.is_set()
        for body in ({"kind": "microphone", "device": 999}, {"kind": "microphone", "device": True},
                     {"kind": "camera", "device": 1}, {"kind": "unknown"}):
            assert request("POST", body)[0] == 400
        assert request("POST", {"kind": "speaker", "device": 1}, {"Content-Type": "application/json", "Origin": "https://other.example"})[0] == 403
    finally:
        engine.shutdown_event.set()
        server.shutdown()
