"""One slot publication path for ongoing observations, task results and late recall."""

from collections.abc import Iterator
import json
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from glados.autonomy.agents.compaction_agent import CompactionAgent
from glados.autonomy.event_bus import EventBus
from glados.autonomy.mind_schedule import OnDemand
from glados.autonomy.mind_scheduler import MindScheduler
from glados.autonomy.slots import TaskSlotStore
from glados.autonomy.task_manager import TaskManager, TaskResult
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados
from glados.core.tool_executor import ToolExecutor
from tests.test_autonomy_core import make_loop, wait_until
from tests.test_memory_recall import core, save


def test_publication_is_visible_before_notification_and_timestamp_refresh_is_quiet() -> None:
    bus = EventBus()
    store = TaskSlotStore(event_bus=bus)
    slot = store.update_slot("test", "Test", "done", "Ready", report="First result", update_priority="important")
    event = bus.get(timeout=0.1)
    assert store.get_slot(event.slot_id) is slot
    assert event.update_priority == "important" and event.revision == slot.revision
    store.update_slot("test", "Test", "done", "Ready", report="First result", update_priority="important")
    assert bus._queue.empty()
    store.update_slot("test", "Test", "done", "Ready", report="Changed result", update_priority="important")
    assert bus.get(timeout=0.1).revision > event.revision


def test_regular_update_clears_old_result_and_overrides_legacy_attention_flags() -> None:
    loop = make_loop()
    slots = loop._slot_store
    slots.update_slot("weather", "Weather", "active", "Storm", report="Old storm", update_priority="important")
    loop._remember(loop._event_bus.get(timeout=0.1))
    assert loop.snapshot()["pending_updates"] == 1
    slot = slots.update_slot("weather", "Weather", "active", "Clear", update_priority="regular", notify_user=True)
    loop._remember(loop._event_bus.get(timeout=0.1))
    assert slot.report is None and loop.snapshot()["pending_updates"] == 0


def test_slot_publication_wakes_loop_without_polling_and_report_change_is_new() -> None:
    loop = make_loop()
    worker = threading.Thread(target=loop.run)
    worker.start()
    try:
        loop._slot_store.update_slot("task", "Search", "running", "Checking sources", update_priority="regular")
        wait_until(lambda: loop._event_bus._queue.empty())
        assert loop._llm_queue.empty()
        loop._slot_store.update_slot(
            "task", "Search", "done", "Ready", report="One source", update_priority="important"
        )
        first = loop._llm_queue.get(timeout=2)
        loop.finish_cycle(first["_autonomy_cycle"], "silent")
        loop._slot_store.update_slot(
            "task", "Search", "done", "Ready", report="Corrected source", update_priority="important"
        )
        second = loop._llm_queue.get(timeout=2)
        assert second["_autonomy_cycle"] != first["_autonomy_cycle"]
        assert "Corrected source" in loop._task_summary()  # Reports enter the separate slot context.
    finally:
        loop._shutdown_event.set()
        worker.join(2)


def test_recovered_condition_is_not_restored_after_cancelled_review() -> None:
    loop = make_loop()
    loop._slot_store.update_slot(
        "health", "Health", "active", "Hot", update_priority="important", attention_key="hot:1"
    )
    loop._remember(loop._event_bus.get(timeout=0.1))
    assert loop._dispatch("Review health")
    cycle = loop._llm_queue.get_nowait()["_autonomy_cycle"]
    loop._active_updates = dict(loop._pending_updates)
    loop._pending_updates.clear()
    loop._slot_store.update_slot("health", "Health", "active", "Recovered", update_priority="regular")
    loop._remember(loop._event_bus.get(timeout=0.1))
    loop.finish_cycle(cycle, "cancelled")
    assert loop.snapshot()["pending_updates"] == 0


def test_superseded_queued_publication_cannot_restore_an_old_alert() -> None:
    loop = make_loop()
    loop._slot_store.update_slot("health", "Health", "active", "Hot", update_priority="important")
    loop._slot_store.update_slot("health", "Health", "active", "Normal", update_priority="regular")
    loop._remember(loop._event_bus.get(timeout=0.1))
    assert loop.snapshot()["pending_updates"] == 0


def test_recall_runs_alongside_reply_and_publishes_once_for_its_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save(tmp_path / "facts.jsonl", "My favourite food is steak.")
    agent = core(tmp_path)
    loop = make_loop()
    agent._slot_store = loop._slot_store
    entered, release = threading.Event(), threading.Event()
    retrieve = agent._recall.retrieve

    def delayed(query: str, previous: str) -> dict:
        entered.set()
        assert release.wait(2)
        return retrieve(query, previous)

    monkeypatch.setattr(agent._recall, "retrieve", delayed)
    try:
        agent.request_recall("What should I have for dinner?", turn_id="turn-7")
        assert entered.wait(1)  # request_recall returned while retrieval is still blocked.
        assert loop._slot_store.get_slot("compaction").context is None
        release.set()
        wait_until(lambda: not agent.runtime.executing and agent._pending_recall is None)
        slot = loop._slot_store.get_slot("compaction")
        assert slot.update_priority == "important" and slot.turn_id == "turn-7"
        assert "steak" in slot.context and "Original query" in slot.context and '"source": "user"' in slot.context
        loop._scan_slots()
        event = loop._pending_updates["compaction"]
        loop._seen_updates["compaction"] = loop._signature(event)
        loop._pending_updates.clear()
        agent.write_slot(status="monitoring", summary="Summary maintenance completed", notify_user=False)
        loop._scan_slots()
        assert loop.snapshot()["pending_updates"] == 0
    finally:
        release.set()
        wait_until(lambda: not agent.runtime.executing and agent._pending_recall is None)


def test_new_turn_discards_inflight_recall_and_keeps_only_latest_waiting_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save(tmp_path / "facts.jsonl", "My favourite food is steak.", "My workshop is orange.")
    agent = core(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []
    retrieve = agent._recall.retrieve

    def delayed(query: str, previous: str) -> dict:
        calls.append(query)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return retrieve(query, previous)

    monkeypatch.setattr(agent._recall, "retrieve", delayed)
    try:
        agent.request_recall("dinner", turn_id="1")
        assert entered.wait(1)
        agent.request_recall("tea", turn_id="2")
        agent.request_recall("workshop", turn_id="3")
        release.set()
        wait_until(lambda: not agent.runtime.executing and agent._pending_recall is None)
        slot = agent._slot_store.get_slot("compaction")
        assert calls == ["dinner", "workshop"]
        assert slot.turn_id == "3" and "orange" in slot.context and "steak" not in slot.context
        agent.request_recall("unrelated weather", turn_id="4")
        wait_until(lambda: not agent.runtime.executing and agent._pending_recall is None)
        slot = agent._slot_store.get_slot("compaction")
        assert slot.update_priority == "regular" and slot.attention_key is None and slot.context is None
    finally:
        release.set()
        wait_until(lambda: not agent.runtime.executing and agent._pending_recall is None)


def test_async_recall_searches_historical_summaries(tmp_path: Path) -> None:
    agent = core(tmp_path)
    history = agent._conversation_store
    history.append({"role": "user", "content": "My favourite food is spaghetti."})
    assert history.compact(history.records(), "[summary] The user's favourite food is spaghetti.", 0)
    agent.request_recall("What should I have for dinner?", turn_id="8")
    wait_until(lambda: not agent.runtime.executing and agent._pending_recall is None)
    slot = agent._slot_store.get_slot("compaction")
    assert slot.update_priority == "important" and "spaghetti" in slot.report
    assert "Compacted conversation" in slot.report


def test_engine_clears_old_context_before_routing_and_starts_recall_only_for_accepted_input() -> None:
    engine = Glados.__new__(Glados)
    engine.compaction_agent = Mock()
    engine.search_agent = Mock()
    engine._quiet_generation = 8
    engine.quiet_event = threading.Event()
    engine._emotion_agent = None
    engine._conversation_store = ConversationStore()
    message = {"role": "user", "content": "What should I have for dinner?", "_quiet_generation": 8}
    engine._clear_input_context(message)
    engine.compaction_agent.request_recall.assert_called_once_with("", turn_id="8")
    engine.compaction_agent.reset_mock()
    engine._react_to_input(message)
    engine.compaction_agent.request_recall.assert_called_once_with(message["content"], "", turn_id="8", audio=None)


def test_pausing_memory_discards_a_late_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = core(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def delayed(*args: str) -> dict:
        entered.set()
        assert release.wait(2)
        return {"facts": [{"content": "Steak"}], "context": "Old food preference"}

    monkeypatch.setattr(agent._recall, "retrieve", delayed)
    try:
        agent.request_recall("dinner", turn_id="8")
        assert entered.wait(1)
        agent.set_paused(True)
        release.set()
        wait_until(lambda: not agent.runtime.executing and agent._pending_recall is None)
        slot = agent._slot_store.get_slot("compaction")
        assert slot.context is None and slot.update_priority == "regular"
    finally:
        release.set()
        wait_until(lambda: not agent.runtime.executing and agent._pending_recall is None)


def test_task_progress_is_regular_and_result_publishes_one_important_update() -> None:
    store, bus = TaskSlotStore(), EventBus()
    tasks = TaskManager(store, bus, progress_interval_s=0.01)
    release = threading.Event()

    def work() -> TaskResult:
        assert release.wait(2)
        return TaskResult("partial", "Some sources found", report="Verified evidence", update_priority="important")

    try:
        handle = tasks.submit("search", "Search", work, progress=lambda: "Comparing two sources")
        wait_until(lambda: "elapsed" in store.get_slot("search").summary)
        assert store.get_slot("search").update_priority == "regular"
        release.set()
        handle.future.result(timeout=2)
        tasks.shutdown(wait=True)
        slot = store.get_slot("search")
        assert slot.status == "partial" and slot.report == "Verified evidence"
        events = list(bus._queue.queue)
        assert sum(e.update_priority == "important" for e in events) == 1
        assert events[-1].status == "partial"
    finally:
        release.set()
        handle.future.result(timeout=2)
        tasks.shutdown(wait=True)


@pytest.mark.parametrize("status", ["done", "partial", "cancelled", "error"])
def test_search_outcomes_are_preserved(status: str) -> None:
    assert ToolExecutor._search_status(json.dumps({"status": status})) == status


@pytest.fixture(autouse=True)
def schedule_memory_cores(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import tests.test_memory_recall as helpers
    original = helpers.core
    schedulers = []
    def create(tmp_path: Path, store: ConversationStore | None = None, **kwargs: object) -> CompactionAgent:
        agent = original(tmp_path, store, **kwargs)
        scheduler = MindScheduler(agent._slot_store)
        scheduler.register(agent, OnDemand(), run_on_start=False)
        scheduler.start_all()
        schedulers.append(scheduler)
        return agent
    monkeypatch.setattr(helpers, "core", create)
    # This module imported the helper directly.
    monkeypatch.setattr(__import__(__name__, fromlist=["core"]), "core", create)
    yield
    for scheduler in schedulers:
        scheduler.shutdown()
