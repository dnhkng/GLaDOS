from unittest.mock import patch

from glados.observability import ObservabilityBus
from glados.observability.latency import LatencyTracker


def _emit_at(bus: ObservabilityBus, t: float, source: str, kind: str, **meta: object) -> None:
    with patch("glados.observability.bus.time.time", return_value=t):
        bus.emit(source, kind, kind, meta=meta)


def test_voice_turn_breaks_down_into_pipeline_stages() -> None:
    bus = ObservabilityBus()
    tracker = LatencyTracker(bus)
    _emit_at(bus, 100.0, "asr", "user_input", asr_ms=250.0)
    _emit_at(bus, 100.2, "tts", "play", autonomy=True)  # background speech does not end the turn
    _emit_at(bus, 101.5, "llm", "first_token", lane="priority", autonomy=False)
    _emit_at(bus, 102.0, "tts", "ready", autonomy=False)
    _emit_at(bus, 102.1, "tts", "ready", autonomy=False)
    _emit_at(bus, 102.25, "tts", "play", autonomy=False)
    [turn] = tracker.snapshot()
    assert (turn["asr_ms"], turn["llm_ms"], turn["tts_ms"], turn["playback_ms"], turn["total_ms"]) == (
        250.0,
        1500.0,
        500.0,
        250.0,
        2500.0,
    )
    assert any(e.source == "latency" and e.meta["total_ms"] == 2500.0 for e in bus.snapshot())


def test_play_without_a_user_turn_is_ignored_and_a_new_input_restarts() -> None:
    bus = ObservabilityBus()
    tracker = LatencyTracker(bus)
    _emit_at(bus, 10.0, "tts", "play")
    _emit_at(bus, 20.0, "text", "user_input")
    _emit_at(bus, 30.0, "text", "user_input")
    _emit_at(bus, 31.0, "tts", "play")
    [turn] = tracker.snapshot()
    assert turn["source"] == "text" and turn["total_ms"] == 1000.0 and turn["llm_ms"] is None


def test_a_failing_listener_does_not_break_publishing() -> None:
    bus = ObservabilityBus()
    bus.add_listener(lambda event: 1 / 0)
    bus.emit("x", "y", "still delivered")
    assert bus.snapshot()[-1].message == "still delivered"
