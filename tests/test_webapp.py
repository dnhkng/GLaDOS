"""Tests for the in-process webapp observability console."""

import http.client
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from glados.observability import ObservabilityBus, ObservabilityEvent
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

    def test_slow_subscriber_keeps_newest_events(self) -> None:
        bus = ObservabilityBus(subscriber_max=2)
        sub = bus.subscribe()
        try:
            for i in range(4):
                bus.emit("engine", "tick", str(i))
            assert [sub.get_nowait().message for _ in range(2)] == ["2", "3"]
        finally:
            bus.unsubscribe(sub)


# --------------------------------------------------------------------------- serializers
# Stub engine carrying just the accessors the serializers touch.


class _FakeInteraction:
    def seconds_since_user(self) -> int:
        return 12

    def seconds_since_assistant(self) -> int:
        return 3


class _FakeAudioState:
    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(rms=-30.5, vad_active=True)


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
        self.shutdown_event = SimpleNamespace(is_set=lambda: False)
        self.observability_bus = ObservabilityBus()
        self._command_order = ["/mcp", "/tts"]


class _FakeQueue:
    def __init__(self, size: int) -> None:
        self._size = size

    def qsize(self) -> int:
        return self._size


def test_serialize_event() -> None:
    ev = type(
        "E",
        (),
        {"timestamp": 1, "source": "llm", "kind": "queue", "level": "info", "message": "hi", "meta": {"slot": "s1"}},
    )()
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

        # SSE stream replays history then pushes state pings
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/stream")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type", "").startswith("text/event-stream")
        head = ""
        for _ in range(10):
            head += resp.readline().decode("utf-8", "replace")
            if "hello stream" in head:
                break
        assert "hello stream" in head
        assert "event:" in head
        conn.close()
    finally:
        server.shutdown()


class _SnapshotPublishingBus(ObservabilityBus):
    """Publish once immediately after copying history to expose replay gaps."""

    def __init__(self) -> None:
        super().__init__()
        self._published_during_snapshot = False

    def snapshot(self, limit: int | None = None) -> list[ObservabilityEvent]:
        events = super().snapshot(limit)
        if not self._published_during_snapshot:
            self._published_during_snapshot = True
            self.emit("engine", "race", "between replay and live")
        return events


def test_stream_does_not_lose_event_at_replay_boundary() -> None:
    engine = _FakeEngine()
    engine.observability_bus = _SnapshotPublishingBus()
    server = WebappServer(engine, host="127.0.0.1", port=0)
    server.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
        conn.request("GET", "/api/stream")
        resp = conn.getresponse()
        body = ""
        for _ in range(10):
            body += resp.readline().decode("utf-8", "replace")
            if "between replay and live" in body:
                break
        assert "between replay and live" in body
        conn.close()
    finally:
        server.shutdown()


def test_server_rejects_cross_origin_browser_request() -> None:
    engine = _FakeEngine()
    server = WebappServer(engine, host="127.0.0.1", port=0)
    server.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
        conn.request("GET", "/api/snapshot", headers={"Origin": "https://example.test"})
        resp = conn.getresponse()
        assert resp.status == 403
        assert resp.getheader("Access-Control-Allow-Origin") is None
        conn.close()

        # A forged host without Origin models a DNS-rebinding request.
        conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
        conn.request("GET", "/api/snapshot", headers={"Host": "example.test"})
        resp = conn.getresponse()
        assert resp.status == 403
        conn.close()
    finally:
        server.shutdown()


def test_server_refuses_non_loopback_bind() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        WebappServer(_FakeEngine(), host="0.0.0.0", port=8050)  # noqa: S104


# --------------------------------------------------------------------------- env override


def test_webapp_env_override_enables_disabled_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_webapp_env_absent_keeps_yaml_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    for name in ("GLADOS_WEBAPP_ENABLED", "GLADOS_WEBAPP_HOST", "GLADOS_WEBAPP_PORT"):
        monkeypatch.delenv(name, raising=False)
    resolved = GladosConfig.from_yaml(cfg)
    assert resolved.webapp is None or resolved.webapp.enabled is False


def test_webapp_port_env_preserves_yaml_enabled_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A port-only override must not unexpectedly enable the console."""
    from glados.core.engine import GladosConfig

    cfg = tmp_path / "wp_port.yaml"
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
    monkeypatch.setenv("GLADOS_WEBAPP_PORT", "8099")
    resolved = GladosConfig.from_yaml(cfg)
    assert resolved.webapp is not None
    assert resolved.webapp.enabled is False
    assert resolved.webapp.port == 8099


# --------------------------------------------------------------------------- webapp launcher


class _LauncherEngine:
    """Minimal stand-in for the real engine in launcher tests."""

    def __init__(self) -> None:
        self.announcement = None
        self.ran = False
        self.fail_announcement = False

    def play_announcement(self) -> None:
        """Play the fake announcement or raise for lifecycle testing."""
        if self.fail_announcement:
            raise RuntimeError("announcement failed")

    def run(self) -> None:
        self.ran = True


class _FakeServer:
    """Stub WebappServer that records start/stop without binding a socket."""

    def __init__(self, engine: object, host: str, port: int) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.is_running = False
        self.shutdown_called = False

    def start(self) -> None:
        self.is_running = True

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.is_running = False


def _fake_glados_config(enabled: bool) -> SimpleNamespace:
    from glados.webapp import WebappConfig

    return SimpleNamespace(webapp=WebappConfig(enabled=enabled, host="127.0.0.1", port=0))


def test_run_webapp_refuses_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disabled webapp aborts the launcher without building the engine."""
    from glados import cli

    monkeypatch.setattr(cli.GladosConfig, "from_yaml", lambda *a, **k: _fake_glados_config(enabled=False))
    monkeypatch.setattr(
        cli.Glados,
        "from_config",
        lambda c: (_ for _ in ()).throw(AssertionError("engine must not be built")),
    )

    with pytest.raises(SystemExit) as exc:
        cli.run_webapp("x.yaml")
    assert exc.value.code == 1


def test_run_webapp_starts_server_then_shuts_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher builds the engine, starts its server, runs, then shuts down."""
    from glados import cli

    engine = _LauncherEngine()
    created: list[_FakeServer] = []

    def _server_factory(engine: object, host: str, port: int) -> _FakeServer:
        server = _FakeServer(engine, host, port)
        created.append(server)
        return server

    monkeypatch.setattr(cli.GladosConfig, "from_yaml", lambda *a, **k: _fake_glados_config(enabled=True))
    monkeypatch.setattr(cli.Glados, "from_config", lambda c: engine)
    monkeypatch.setattr(cli, "WebappServer", _server_factory)

    cli.run_webapp("x.yaml")

    assert engine.ran is True
    assert len(created) == 1
    server = created[0]
    assert server.port == 0
    assert server.shutdown_called is True
    assert server.is_running is False


def test_run_webapp_shuts_down_when_announcement_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HTTP socket is closed even if startup audio raises."""
    from glados import cli

    engine = _LauncherEngine()
    engine.announcement = "hello"
    engine.fail_announcement = True
    created: list[_FakeServer] = []

    def _server_factory(engine: object, host: str, port: int) -> _FakeServer:
        server = _FakeServer(engine, host, port)
        created.append(server)
        return server

    monkeypatch.setattr(cli.GladosConfig, "from_yaml", lambda *a, **k: _fake_glados_config(enabled=True))
    monkeypatch.setattr(cli.Glados, "from_config", lambda config: engine)
    monkeypatch.setattr(cli, "WebappServer", _server_factory)

    with pytest.raises(RuntimeError, match="announcement failed"):
        cli.run_webapp("x.yaml")

    assert created[0].shutdown_called is True
