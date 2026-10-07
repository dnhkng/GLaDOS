"""Tests for the in-process webapp observability console."""

import http.client
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from glados.core.speech_animation import SpeechAnimationState
from glados.observability import ObservabilityBus
from glados.webapp.serializers import (
    build_snapshot,
    build_state,
    serialize_event,
)
from glados.webapp.server import WebappServer

# --------------------------------------------------------------------------- bus fan-out


class TestObservabilityBusFanOut:
    def test_subscribe_receives_every_published_event(self) -> None:
        bus = ObservabilityBus()
        sub = bus.subscribe()
        try:
            for i in range(5):
                bus.emit("autonomy", "tick", f"event {i}", meta={"i": i})
            got = [sub.get_nowait().message for _ in range(5)]
            assert got == [f"event {i}" for i in range(5)]
        finally:
            bus.unsubscribe(sub)

    def test_drain_still_works_independently_of_subscribers(self) -> None:
        bus = ObservabilityBus()
        sub = bus.subscribe()
        try:
            bus.emit("llm", "queue", "to drain caller", level="debg")
            # The single-consumer drain() queue must still see the event
            # (mirrors the TUI ObservabilityScreen).
            drained = bus.drain(max_items=10)
            assert [e.message for e in drained] == ["to drain caller"]
            # And the subscriber got its own independent copy.
            assert sub.get_nowait().message == "to drain caller"
        finally:
            bus.unsubscribe(sub)

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = ObservabilityBus()
        sub = bus.subscribe()
        bus.emit("engine", "start", "a")
        bus.unsubscribe(sub)
        bus.emit("engine", "start", "b")
        # subscriber keeps only the first event
        assert sub.get_nowait().message == "a"
        assert sub.empty()


# --------------------------------------------------------------------------- serializers
# Stub engine carrying just the accessors the serializers touch.


class _FakeInteraction:
    def seconds_since_user(self) -> int:
        return 12

    def seconds_since_assistant(self) -> int:
        return 3


class _FakeAudioState:
    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(rms=0.1, vad_active=True)


class _FakeMindRegistry:
    def snapshot(self) -> list:
        return [
            SimpleNamespace(
                mind_id="m1",
                title="Forecast Mind",
                status="running",
                summary="watching the sky",
                role="weather-summarizer",
                updated_at=time.time(),
            )
        ]


class _FakeEngine:
    def __init__(self) -> None:
        self.autonomy_config = SimpleNamespace(enabled=True)
        self.llm_queue_priority = _FakeQueue(1)
        self.llm_queue_autonomy = _FakeQueue(2)
        self._priority_inflight = SimpleNamespace(value=lambda: 0)
        self._autonomy_inflight = SimpleNamespace(value=lambda: 3)
        self.autonomy_llm_processors = [None, None]
        self.audio_state = _FakeAudioState()
        self._emotion_agent = None
        self.mcp_manager = None
        self.autonomy_slots = None
        self.subagent_manager = None
        self.vision_state = None
        self.mind_registry = _FakeMindRegistry()
        self.interaction_state = _FakeInteraction()
        self.currently_speaking_event = SimpleNamespace(is_set=lambda: False)
        self.shutdown_event = threading.Event()
        self.observability_bus = ObservabilityBus()
        self._command_order = ["/mcp", "/tts"]


class _FakeQueue:
    def __init__(self, size: int) -> None:
        self._size = size

    def qsize(self) -> int:
        return self._size


def test_serialize_event() -> None:
    ev = type("E", (), {"timestamp": 1, "source": "llm", "kind": "queue",
                        "level": "info", "message": "hi", "meta": {"slot": "s1"}})()
    out = serialize_event(ev)
    assert out["source"] == "llm"
    assert out["meta"]["slot"] == "s1"


def test_build_snapshot_and_state() -> None:
    engine = _FakeEngine()
    snap = build_snapshot(engine)
    assert snap["lanes"]["priority"]["queue"] == 1
    assert snap["lanes"]["autonomy"]["inflight"] == 3
    assert snap["minds"][0]["mind_id"] == "m1"
    assert snap["speaking"] is False

    state = build_state(engine)
    assert state["lanes"]["autonomy"]["inflight"] == 3
    assert state["interaction"]["seconds_since_user"] == 12


def test_performance_recovers_current_state_without_replaying_history() -> None:
    engine = _FakeEngine()
    engine.speech_animation = SpeechAnimationState(engine.observability_bus)
    engine.speech_animation.set(True, "smug")
    engine.speech_animation.set(False)
    assert build_snapshot(engine)["performance"] == build_state(engine)["performance"]
    server = WebappServer(engine, port=0)
    server.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
    try:
        conn.request("GET", "/api/stream")
        response = conn.getresponse()

        def event() -> tuple[str, dict]:
            kind, payload = "", {}
            while line := response.readline().decode().strip():
                if line.startswith("event: "):
                    kind = line[7:]
                elif line.startswith("data: "):
                    payload = json.loads(line[6:])
            return kind, payload

        initial = [event() for _ in range(4)]
        assert [kind for kind, _ in initial] == ["obs", "obs", "state", "snapshot"]
        assert initial[-1][1]["performance"]["active"] is False
        engine.speech_animation.set(True, "disappointed")
        for _ in range(10):
            kind, payload = event()
            if kind == "performance":
                assert payload["active"] is True
                assert payload["emotion"] == "disappointed"
                assert payload["revision"] == 3
                break
        else:
            pytest.fail("Live performance event was not delivered")
    finally:
        conn.close()
        engine.shutdown_event.set()
        server.shutdown()


# --------------------------------------------------------------------------- HTTP server


def test_webapp_server_serves_snapshot_and_stream() -> None:
    engine = _FakeEngine()
    engine.observability_bus.emit("autonomy", "tick", "hello stream", meta={"x": 1})
    server = WebappServer(engine, host="127.0.0.1", port=0)
    server.start()
    try:
        assert server.is_running
        port = server.bound_port
        assert port

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/snapshot")
        resp = conn.getresponse()
        assert resp.status == 200
        payload = json.loads(resp.read())
        assert "minds" in payload
        conn.close()

        # static console
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        body = resp.read().decode("utf-8")
        assert "GLaDOS Core Console" in body
        conn.close()

        # The avatar's assets must be served from the installed static directory.
        for asset, content_type in (
            ("glados-rig.js", "application/javascript"),
            ("glados-vision.js", "application/javascript"),
            ("glados-routing.js", "application/javascript"),
            ("glados-context.js", "application/javascript"),
            ("glados-devices.js", "application/javascript"),
            ("glados-avatar.js", "application/javascript"),
            ("glados-avatar.css", "text/css"),
        ):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", f"/{asset}")
            resp = conn.getresponse()
            assert resp.status == 200
            assert resp.getheader("Content-Type").startswith(content_type)
            assert resp.read()
            conn.close()

        # SSE stream replays history then pushes state pings
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/stream")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type", "").startswith("text/event-stream")
        head = resp.read(2048).decode("utf-8", "replace")
        assert "hello stream" in head
        assert "event:" in head
        conn.close()
    finally:
        engine.shutdown_event.set()
        server.shutdown()


# --------------------------------------------------------------------------- env override


def test_webapp_env_override_enables_disabled_config(monkeypatch, tmp_path) -> None:
    """GLADOS_WEBAPP_* env vars enable the console even when YAML disables it."""
    from glados.core.engine import GladosConfig

    cfg = tmp_path / "wp_disabled.yaml"
    cfg.write_text(
        "Glados:\n"
        "  llm_model: llama\n"
        "  completion_url: http://localhost:11434/api/chat\n"
        "  api_key: null\n"
        "  interruptible: true\n"
        "  audio_io: sounddevice\n"
        "  input_mode: text\n"
        "  asr_engine: tdt\n"
        "  wake_word: null\n"
        "  voice: glados\n"
        "  announcement: null\n"
        "  webapp:\n"
        "    enabled: false\n"
        "    host: 127.0.0.1\n"
        "    port: 8050\n"
        "  personality_preprompt:\n"
        "    - system: you are a test\n"
    )
    monkeypatch.setenv("GLADOS_WEBAPP_ENABLED", "1")
    monkeypatch.setenv("GLADOS_WEBAPP_PORT", "8085")

    resolved = GladosConfig.from_yaml(cfg)
    assert resolved.webapp is not None
    assert resolved.webapp.enabled is True
    assert resolved.webapp.port == 8085


def test_webapp_env_absent_keeps_yaml_disabled(monkeypatch, tmp_path) -> None:
    """With no env flag, a disabled-by-YAML webapp stays disabled."""
    from glados.core.engine import GladosConfig

    cfg = tmp_path / "wp_disabled.yaml"
    cfg.write_text(
        "Glados:\n"
        "  llm_model: llama\n"
        "  completion_url: http://localhost:11434/api/chat\n"
        "  api_key: null\n"
        "  interruptible: true\n"
        "  audio_io: sounddevice\n"
        "  input_mode: text\n"
        "  asr_engine: tdt\n"
        "  wake_word: null\n"
        "  voice: glados\n"
        "  announcement: null\n"
        "  webapp:\n"
        "    enabled: false\n"
        "  personality_preprompt: []\n"
    )
    monkeypatch.delenv("GLADOS_WEBAPP_ENABLED", raising=False)
    resolved = GladosConfig.from_yaml(cfg)
    assert resolved.webapp is None or resolved.webapp.enabled is False


# --------------------------------------------------------------------------- webapp launcher


class _LauncherEngine:
    """Minimal stand-in for the real engine in launcher tests."""

    def __init__(self) -> None:
        self.announcement = None
        self.ran = False
        self.shutdown_event = threading.Event()
        self.shutdown_called = False

    def run(self) -> None:
        self.ran = True

    def _graceful_shutdown(self) -> None:
        assert self.shutdown_event.is_set()
        self.shutdown_called = True


class _FakeServer:
    """Stub WebappServer that records start/stop without binding a socket."""

    def __init__(self, engine, host: str, port: int, allowed_hosts=None) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.allowed_hosts = allowed_hosts
        self.is_running = False
        self.shutdown_called = False

    def start(self) -> None:
        self.is_running = True

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.is_running = False


def _fake_glados_config(enabled: bool):
    from glados.webapp import WebappConfig

    return SimpleNamespace(webapp=WebappConfig(enabled=enabled, host="127.0.0.1", port=0))


def test_run_webapp_refuses_when_disabled(monkeypatch) -> None:
    """A disabled webapp aborts the launcher without building the engine."""
    from glados import cli

    monkeypatch.setattr(
        cli.GladosConfig, "from_yaml", lambda *a, **k: _fake_glados_config(enabled=False)
    )
    monkeypatch.setattr(
        cli.Glados,
        "from_config",
        lambda c: (_ for _ in ()).throw(AssertionError("engine must not be built")),
    )

    with pytest.raises(SystemExit) as exc:
        cli.run_webapp("x.yaml")
    assert exc.value.code == 1


def test_run_webapp_starts_server_then_shuts_down(monkeypatch) -> None:
    """The launcher builds the engine, starts its server, runs, then shuts down."""
    from glados import cli

    engine = _LauncherEngine()
    created = []

    def _server_factory(*args, **kwargs):
        server = _FakeServer(*args, **kwargs)
        created.append(server)
        return server

    monkeypatch.setattr(
        cli.GladosConfig, "from_yaml", lambda *a, **k: _fake_glados_config(enabled=True)
    )
    monkeypatch.setattr(cli.Glados, "from_config", lambda c: engine)
    monkeypatch.setattr(cli, "WebappServer", _server_factory)

    cli.run_webapp("x.yaml")

    assert engine.ran is True
    assert len(created) == 1
    server = created[0]
    assert server.port == 0
    assert server.shutdown_called is True
    assert server.is_running is False


def test_run_webapp_shuts_down_engine_when_port_is_busy(monkeypatch) -> None:
    from glados import cli

    engine = _LauncherEngine()
    occupied = WebappServer(_FakeEngine(), port=0)
    occupied.start()
    config = _fake_glados_config(enabled=True)
    config.webapp.port = occupied.bound_port
    monkeypatch.setattr(cli.GladosConfig, "from_yaml", lambda *a, **k: config)
    monkeypatch.setattr(cli.Glados, "from_config", lambda c: engine)
    try:
        with pytest.raises(SystemExit) as exc:
            cli.run_webapp("x.yaml")
        assert exc.value.code == 1
        assert not engine.ran
        assert engine.shutdown_event.is_set()
        assert engine.shutdown_called
    finally:
        occupied.shutdown()


def test_browser_renders_server_payload_and_stream_updates() -> None:
    """Run the shipped client against the serializer's actual JSON contract."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the webapp client regression test")
    engine = _FakeEngine()
    engine.llm_queue_priority = _FakeQueue(0)
    engine._priority_inflight = SimpleNamespace(value=lambda: 1)
    engine._autonomy_inflight = SimpleNamespace(value=lambda: 2)
    engine.autonomy_llm_processors = [None] * 4
    result = subprocess.run(
        [node, str(Path(__file__).with_name("webapp_client.cjs"))],
        input=json.dumps(build_snapshot(engine)),
        text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_stream_refreshes_state_and_snapshot_during_continuous_events() -> None:
    engine = _FakeEngine()
    server = WebappServer(engine, port=0)
    stop = threading.Event()

    def publish_continuously() -> None:
        while not stop.wait(0.02):
            engine.observability_bus.emit("test", "tick", "busy")

    producer = threading.Thread(target=publish_continuously, daemon=True)
    server.start()
    producer.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
    try:
        conn.request("GET", "/api/stream")
        response = conn.getresponse()
        states, snapshots, events = [], [], []
        event_type = ""
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and len(snapshots) < 2:
            line = response.readline().decode().strip()
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                if event_type == "state":
                    states.append(payload)
                elif event_type == "obs":
                    events.append(payload)
                elif event_type == "snapshot":
                    snapshots.append(payload)
                    engine.vision_state = SimpleNamespace(snapshot=lambda: "updated scene")
                    engine.mind_registry = SimpleNamespace(snapshot=lambda: [])
        assert len(states) >= 3
        assert len(snapshots) == 2
        assert events
        assert snapshots[0]["minds"]
        assert snapshots[1]["minds"] == []
        assert snapshots[1]["vision"] == "updated scene"
    finally:
        stop.set()
        engine.shutdown_event.set()
        conn.close()
        producer.join(timeout=1)
        server.shutdown()


def test_stream_pushes_fast_camera_updates_without_rebuilding_full_state() -> None:
    engine = _FakeEngine()
    calls = {"snapshot": 0, "tracking": 0}

    def tracking() -> dict:
        calls["tracking"] += 1
        return {"paused": False, "camera": {
            "enabled": True, "connected": True,
            "face": {"observed_at": time.time(), "present": True, "x": 0.25, "y": -0.1},
        }}

    def snapshot() -> dict:
        calls["snapshot"] += 1
        return {"scene": "Stable caption", "revision": 1, **tracking()}

    engine.vision_agent = SimpleNamespace(snapshot=snapshot, tracking_snapshot=tracking)
    server = WebappServer(engine, port=0)
    server.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
    try:
        conn.request("GET", "/api/stream")
        response = conn.getresponse()
        cameras, states = [], []
        event_type = ""
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(cameras) < 20:
            line = response.readline().decode().strip()
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                if event_type == "camera":
                    cameras.append(payload)
                elif event_type == "state":
                    states.append(payload)
        assert len(cameras) == 20
        assert all(c["camera"]["face"]["x"] == 0.25 for c in cameras)
        assert calls["snapshot"] < len(cameras) / 2
        assert states and all(s["vision_mind"]["scene"] == "Stable caption" for s in states)
    finally:
        engine.shutdown_event.set()
        conn.close()
        server.shutdown()


def test_stream_pushes_vad_between_dashboard_refreshes() -> None:
    from glados.core.audio_state import AudioState

    engine = _FakeEngine()
    engine.audio_state = AudioState()
    server = WebappServer(engine, port=0)
    server.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
    try:
        conn.request("GET", "/api/stream")
        response = conn.getresponse()
        event_type = ""
        received = []
        started = time.monotonic()
        while time.monotonic() - started < 2:
            line = response.readline().decode().strip()
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                if event_type == "state" and not received:
                    assert not payload["audio"]["vad_active"]
                    engine.audio_state.update(0.1, True)
                elif event_type == "audio" and payload["vad_active"]:
                    received.append(payload)
                    engine.audio_state.update(0, False)
                elif event_type == "audio" and received and not payload["vad_active"]:
                    received.append(payload)
                    break
        assert len(received) == 2
        assert received[1]["updated_at"] > received[0]["updated_at"]
        assert time.monotonic() - started < 0.4, "VAD must not wait for the 500ms dashboard refresh"
    finally:
        engine.shutdown_event.set()
        conn.close()
        server.shutdown()
