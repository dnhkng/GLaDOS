"""Queue, catalogue and evidence contracts shared by independently running cores."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
import pytest

from glados.autonomy.context import slot_evidence
from glados.autonomy.event_bus import EventBus
from glados.autonomy.slots import TaskSlotStore
from glados.autonomy.task_manager import TaskManager, TaskResult
from glados.autonomy.agents.compaction_agent import CompactionAgent
from glados.autonomy.subagent import SubagentConfig
from glados.autonomy.llm_client import LLMConfig
from glados.core.memory_recall import MemoryRecall, RecallConfig
from glados.core.memory_records import append_record, revision
from glados.core.conversation_store import ConversationStore


def test_search_fifo_capacity_cancel_and_group_snapshot():
    slots = TaskSlotStore()
    tasks = TaskManager(slots, EventBus(), progress_interval_s=0.01)
    started, release = threading.Event(), threading.Event()
    order = []

    def run(i):
        order.append(i)
        if i == 0:
            started.set()
            assert release.wait(3)
        return TaskResult("done", str(i), report="result " + str(i))

    try:
        handles = [tasks.submit("task_0", "First", lambda: run(0), group="search")]
        assert started.wait(2)
        handles += [tasks.submit("task_" + str(i), str(i), lambda i=i: run(i), group="search") for i in range(1, 9)]
        assert all("task_" + str(i) in slots.get_slot("search").summary for i in range(9))
        assert slots.get_slot("task_8").queue_position == 8
        assert all(h.started_at is None for h in handles[1:])
        with pytest.raises(ValueError, match="full"):
            tasks.submit("overflow", "Overflow", lambda: "", group="search")
        assert tasks.cancel("task_3")
        assert handles[3].future.result(timeout=1).status == "cancelled"
        assert slots.get_slot("task_8").queue_position == 7
        release.set()
        for handle in handles:
            handle.future.result(timeout=2)
        assert order == [0, 1, 2, 4, 5, 6, 7, 8]
        assert slots.get_slot("search").report is None
        result = slots.get_slot("task_0")
        slots.mark_handled(result.slot_id, result.revision)
        assert all(r.get("slot_id") != result.slot_id for r in slot_evidence(slots.list_slots()))
        assert slots.get_slot(result.slot_id).report == "result 0"
    finally:
        release.set()
        tasks.shutdown(wait=True)


def test_shutdown_signals_running_and_cancels_waiting():
    tasks = TaskManager(TaskSlotStore(), EventBus())
    cancellation, started = threading.Event(), threading.Event()

    def run():
        started.set()
        assert cancellation.wait(2)
        return "finished after cancellation"

    running = tasks.submit("first", "First", run, group="search", cancelled=cancellation)
    assert started.wait(2)
    queued = tasks.submit("second", "Second", lambda: pytest.fail("Queued task ran"), group="search")
    tasks.shutdown(wait=True)
    assert running.future.result().status == queued.future.result().status == "cancelled"


def test_evidence_budget_counts_escaping_and_deduplicates():
    slots = TaskSlotStore()
    for i in range(30):
        slots.update_slot(
            str(i), "Title", "running", "Status", context='"' * 6000, report='"' * 6000, update_priority="important"
        )
    rows = slot_evidence(slots.list_slots(), 8000)
    assert len(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))) <= 8000
    assert len(rows) == 30
    assert all("context" not in row and "report" not in row for row in rows)


@pytest.fixture
def recall_core(tmp_path, monkeypatch):
    monkeypatch.setattr("glados.autonomy.subagent.SubagentMemory", Mock())
    agent = CompactionAgent(
        SubagentConfig("compaction", "Memory"),
        LLMConfig("http://unused"),
        recall_config=RecallConfig(memory_dir=str(tmp_path)),
        slot_store=TaskSlotStore(),
    )
    for i in range(65):
        append_record(
            tmp_path / "facts.jsonl",
            {
                "id": str(i),
                "content": "Steak is my favourite" if i == 64 else "Unrelated item " + str(i),
                "source": "user",
                "created_at": i,
            },
        )
    return agent


def test_semantic_recall_all_pages_stable_prefix_and_originals(recall_core, monkeypatch):
    calls = []

    def infer(config, system, query, **kwargs):
        calls.append((config, system, query))
        return '{"ids":["64"]}' if '"id": "64"' in system else '{"ids":[]}'

    monkeypatch.setattr("glados.autonomy.agents.compaction_agent.llm_call", infer)
    result = recall_core._semantic_recall("What should I have for dinner?", "", None, 0)
    assert len(calls) == 3
    assert [f["content"] for f in result["facts"]] == ["Steak is my favourite"]
    # The HTTP timeout starts after admission; a conversation hold must not
    # consume the memory lookup's inference budget.
    assert all(c[0].lane == "autonomy" and c[0].deadline is None and c[0].timeout > 0 for c in calls)
    prefixes = [c[1] for c in calls]
    calls.clear()
    recall_core._semantic_recall("Something for supper?", "", None, 0)
    assert prefixes == [c[1] for c in calls]


def test_audio_topic_and_edits_inflight_do_not_publish_stale_facts(recall_core, monkeypatch):
    calls = []

    def infer(config, system, query, **kwargs):
        calls.append(query)
        if isinstance(query, list):
            return '{"query":"dinner suggestion"}'
        if '"id": "64"' in system:
            recall_core._memory_store.edit("64", revision("Steak is my favourite"), "I prefer pasta now")
            return '{"ids":["64"]}'
        return '{"ids":[]}'

    monkeypatch.setattr("glados.autonomy.agents.compaction_agent.llm_call", infer)
    audio = [{"type": "input_audio", "input_audio": {"data": "transient", "format": "wav"}}]
    result = recall_core._semantic_recall("", "", audio, 0)
    assert calls[0][1:] == audio and result["query"] == "dinner suggestion"
    assert result["facts"] == []
    assert "transient" not in str(recall_core._memory_store.entries())


def test_memory_atomic_edits_and_revision_conflicts(recall_core, tmp_path):
    entry = recall_core.memory_entry("64")
    with ThreadPoolExecutor(2) as workers:
        append = workers.submit(
            append_record, tmp_path / "facts.jsonl", {"id": "new", "content": "Another fact", "created_at": 100}
        )
        edit = workers.submit(recall_core.mutate_memory, "64", "edit", entry["revision"], "Pasta")
        append.result()
        edit.result()
    assert recall_core.memory_entry("new")["content"] == "Another fact"
    with pytest.raises(ValueError, match="changed"):
        recall_core.mutate_memory("64", "delete", entry["revision"])
    recall_core.mutate_memory("64", "delete", recall_core.memory_entry("64")["revision"])
    with pytest.raises(ValueError, match="not found"):
        recall_core.memory_entry("64")


def test_catalogue_large_escaped_entries_fit_pages():
    memory = MemoryRecall(RecallConfig())
    entries = [{"id": "large", "content": "\x00" * 30000}]
    pages = memory.catalogue_pages(entries)
    assert len(pages) > 1
    assert all(len(page) <= 24000 and len(page.splitlines()) <= 32 for page in pages)
    assert {json.loads(line)["id"] for page in pages for line in page.splitlines()} == {"large"}


def test_conversation_summary_edits_invalidate_compaction_snapshot():
    store = ConversationStore()
    store.append({"role": "user", "content": "Original"})
    assert store.compact(store.records(), "Saved summary", 0)
    old = store.records()
    store.edit_summary(old[0].id, revision("Saved summary"), "Corrected summary")
    assert not store.compact(old, "Stale summary", 1)
    assert store.snapshot()[0]["content"] == "Corrected summary"
    store.edit_summary(old[0].id, revision("Corrected summary"), None)
    assert not store.records()


def test_maintenance_tick_preserves_semantic_result(recall_core, monkeypatch):
    recall_core._compaction_enabled = False
    recall_core._recall_query = "dinner"
    recall_core._recall_result = {"facts": [{"id": "64"}], "context": "Semantic steak finding"}
    monkeypatch.setattr(recall_core, "recall_for", lambda *a: pytest.fail("Maintenance used lexical recall"))
    recall_core.tick()
    assert recall_core._recall_result["context"] == "Semantic steak finding"


def test_memory_http_full_record_edit_delete_and_revision_conflict(recall_core):
    import http.client
    from tests.test_webapp import _FakeEngine
    from glados.webapp.server import WebappServer

    engine = _FakeEngine()
    engine.compaction_agent = recall_core
    server = WebappServer(engine, port=0)
    server.start()

    def request(method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
        try:
            connection.request(method, path, json.dumps(body) if body else None, {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        status, entry = request("GET", "/api/memory?id=64")
        assert status == 200 and entry["content"] == "Steak is my favourite"
        body = {"id": "64", "revision": entry["revision"], "action": "edit", "content": "Pasta"}
        assert request("POST", "/api/memory/edit", body)[0] == 200
        assert request("POST", "/api/memory/edit", body)[0] == 400
        entry = request("GET", "/api/memory?id=64")[1]
        assert (
            request("POST", "/api/memory/edit", {"id": "64", "revision": entry["revision"], "action": "delete"})[0]
            == 200
        )
        assert request("GET", "/api/memory?id=64")[0] == 400
    finally:
        server.shutdown()


def test_preview_cannot_replace_versions_of_submitted_review():
    from tests.test_speech_markup import make_processor
    from glados.autonomy.context import slot_version
    processor = make_processor()
    processor.slot_store = TaskSlotStore()
    slot = processor.slot_store.update_slot("task", "Result", "done", "First", report="Old evidence")
    processor._build_messages(True)
    submitted = dict(processor._autonomy_meta["_evidence_versions"])
    assert submitted["task"] == slot_version(slot)
    processor.slot_store.update_slot("task", "Result", "done", "Second", report="New evidence")
    processor.context_preview(True)
    assert processor._autonomy_meta["_evidence_versions"] == submitted
    processor._build_messages(True)
    assert processor._autonomy_meta["_evidence_versions"] != submitted
