"""Agents, their outputs, and callable tools have separate contracts."""

from datetime import datetime
import json
import queue
import threading
from unittest.mock import Mock

import pytest

from glados.autonomy.mind_runtime import MindRuntime
from glados.autonomy.mind_schedule import FixedInterval
from glados.autonomy.mind_scheduler import MindScheduler
from glados.autonomy.slots import TaskSlotStore
from glados.autonomy.subagent import Subagent, SubagentConfig, SubagentOutput
from glados.tools.get_time import GetTime, current_time
from glados.webapp.serializers import build_snapshot
from tests.test_speech_markup import make_processor
from tests.test_webapp import _FakeEngine


def test_clock_returns_real_time_and_rejects_bad_zones() -> None:
    before = datetime.now().astimezone().timestamp()
    result = current_time("Europe/Berlin")
    assert before <= datetime.fromisoformat(result["datetime"]).timestamp() <= datetime.now().timestamp()
    assert result["timezone"] == "Europe/Berlin"
    assert current_time("UTC")["utc_offset"] == "+0000"
    with pytest.raises(ValueError):
        current_time("Not/AZone")
    output = queue.Queue()
    GetTime(output).run("clock-1", {"timezone": "Not/AZone"})
    message = output.get_nowait()
    assert message["tool_call_id"] == "clock-1"
    assert "error" in json.loads(message["content"])


def test_minds_and_tools_exclude_engine_components() -> None:
    engine = _FakeEngine()
    engine.llm_processor = make_processor()
    payload = build_snapshot(engine)
    assert [mind["id"] for mind in payload["agent_minds"]] == ["glados"]
    assert payload["minds"][0]["mind_id"] == "m1"
    assert "get_time" in {tool["name"] for tool in payload["tools"]}
    assert "speak" not in {tool["name"] for tool in payload["tools"]}


def test_reply_context_reads_clock_fresh_without_retaining_old_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Mock(side_effect=[{"time": "10:01:00"}, {"time": "10:02:00"}, {"time": "10:03:00"}])
    monkeypatch.setattr("glados.core.llm_processor.current_time", clock)
    processor = make_processor()
    first = json.dumps(processor._build_messages(False))
    second = json.dumps(processor._build_messages(False))
    assert "10:01:00" in first
    assert "10:02:00" in second and "10:01:00" not in second
    autonomous = json.dumps(processor._build_messages(True))
    assert "10:03:00" in autonomous and "10:02:00" not in autonomous


def test_paused_mind_can_run_once_without_stopping_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glados.autonomy.subagent.SubagentMemory", Mock())
    completed = threading.Event()
    shutdown = threading.Event()

    class Mind(Subagent):
        def run(self, runtime: MindRuntime) -> SubagentOutput:
            completed.set()
            return SubagentOutput(status="done", summary="Real result")

    store = TaskSlotStore()
    mind = Mind(SubagentConfig("test", "Test"), store, shutdown_event=shutdown)
    mind.set_paused(True)
    scheduler = MindScheduler(store)
    scheduler.register(mind, FixedInterval(0.1))
    scheduler.start_all()
    try:
        assert not completed.wait(0.2), "Starting a paused mind must not execute work"
        mind.request_tick()
        assert completed.wait(2)
        completed.clear()
        assert not completed.wait(0.3), "Run once must keep the schedule paused"
        assert store.get_slot("test").summary == "Real result"
        assert not shutdown.is_set()
        mind.set_paused(False)
        assert completed.wait(2)
    finally:
        shutdown.set()
        scheduler.shutdown()
    assert not scheduler._thread.is_alive()
