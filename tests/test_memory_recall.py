"""Recall is topic-dependent, bounded and available before the current inference."""

import http.client
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from glados.autonomy.agents.compaction_agent import CompactionAgent
from glados.autonomy.slots import TaskSlotStore
from glados.autonomy.subagent import SubagentConfig
from glados.core.conversation_store import ConversationStore
from glados.core.decision_lists import default_list
from glados.core.engine import Glados
from glados.core.memory_recall import MemoryRecall, RecallConfig
from glados.core.routing_tree import RoutingTree
from glados.tools import tool_definitions
from glados.webapp.server import WebappServer
from tests.test_speech_markup import make_processor
from tests.test_webapp import _FakeEngine


def save(path: Path, *contents: str) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"fact_{i}",
                    "content": content,
                    "source": "user",
                    "importance": i / max(1, len(contents)),
                    "created_at": i + 1,
                }
            )
            for i, content in enumerate(contents)
        )
        + "\n"
    )


@pytest.fixture
def memory(tmp_path: Path) -> MemoryRecall:
    save(
        tmp_path / "facts.jsonl",
        "My workshop paint is orange.",
        "The robot camera uses a USB cable.",
        "My preferred tea is Earl Grey.",
    )
    return MemoryRecall(RecallConfig(memory_dir=str(tmp_path)))


def test_topic_changes_replace_facts_without_importance_filter(memory: MemoryRecall) -> None:
    result = memory.retrieve("What colour is my workshop paint?")
    assert [f["content"] for f in result["facts"]] == ["My workshop paint is orange."]
    assert result["facts"][0]["importance"] == 0
    assert "Earl Grey" not in result["context"]
    result = memory.retrieve("What tea do I prefer?")
    assert [f["content"] for f in result["facts"]] == ["My preferred tea is Earl Grey."]
    assert memory.retrieve("What is the weather?")["context"] is None
    assert memory.retrieve("")["facts"] == []


def test_followup_reuses_topic_but_new_question_does_not(memory: MemoryRecall) -> None:
    assert "USB" in memory.retrieve("What about it?", "Tell me about the robot camera")["context"]
    assert memory.retrieve("What is the weather?", "Tell me about the robot camera")["facts"] == []


def test_changed_deleted_and_malformed_files_do_not_leave_stale_recall(memory: MemoryRecall) -> None:
    file = Path(memory.config.memory_dir) / "facts.jsonl"
    assert memory.retrieve("workshop")["facts"]
    save(file, "My workshop paint is blue.")
    with file.open("a") as stream:
        stream.write('{"unfinished":\n')
        stream.write(json.dumps({"content": "The robot camera is wireless", "created_at": 10}) + "\n")
    assert "blue" in memory.retrieve("workshop")["context"]
    assert "wireless" in memory.retrieve("camera")["context"]
    file.unlink()
    assert memory.retrieve("workshop")["facts"] == []


def test_duplicate_records_keep_newest_and_corrections_have_provenance(tmp_path: Path) -> None:
    rows = [
        {"id": "project", "content": "Project launch is Friday", "created_at": 1},
        {"id": "project", "content": "Project launch moved to Monday", "created_at": 2},
        {"id": "copy", "content": "Project launch moved to Monday", "created_at": 3},
    ]
    (tmp_path / "facts.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = MemoryRecall(RecallConfig(memory_dir=str(tmp_path))).retrieve("project launch")
    assert len(result["facts"]) == 1
    assert "Friday" not in result["context"]
    assert '"id": "copy"' in result["context"] and '"saved_at"' in result["context"]
    assert "later corrections take precedence" in result["context"]


def test_huge_memory_has_bounded_index_and_context(tmp_path: Path) -> None:
    save(tmp_path / "facts.jsonl", *[f"Project {i} detail " + "x" * 1000 for i in range(80)])
    memory = MemoryRecall(
        RecallConfig(
            memory_dir=str(tmp_path), max_file_bytes=4096, max_candidates=16, max_fact_chars=100, max_chars=512
        )
    )
    result = memory.retrieve("project")
    assert result["indexed_facts"] <= 16
    assert result["context"] and len(result["context"]) <= 512
    assert all(len(f["content"]) <= 100 for f in result["facts"])
    assert not memory.browse()["limited"]
    assert memory.browse()["total"] == 80  # Catalogue/browser includes the complete store.
    assert all(f["excerpt"] for f in result["facts"])


def core(tmp_path: Path, store: ConversationStore | None = None, **kwargs: object) -> CompactionAgent:
    return CompactionAgent(
        SubagentConfig("compaction", "Memory Core"),
        conversation_store=store or ConversationStore(),
        slot_store=TaskSlotStore(),
        recall_config=RecallConfig(memory_dir=str(tmp_path)),
        **kwargs,
    )


def test_recall_is_in_memory_core_slot_and_clears_on_pause_and_topic_change(tmp_path: Path) -> None:
    save(tmp_path / "facts.jsonl", "User owns an orange workshop.")
    agent = core(tmp_path)
    agent.recall_for("workshop")
    slot = agent._slot_store.get_slot("compaction")
    assert not slot.notify_user and "orange" in slot.context
    engine = SimpleNamespace(autonomy_slots=agent._slot_store)
    assert "orange" in Glados._format_slots(engine)
    agent.set_paused(True)
    assert agent._slot_store.get_slot("compaction").context is None
    assert Glados._format_slots(engine) is None
    agent.set_paused(False)
    agent._run_recall()
    assert "orange" in agent._slot_store.get_slot("compaction").context
    agent.recall_for("tea")
    assert agent._slot_store.get_slot("compaction").context is None


def test_compaction_update_cannot_overwrite_recall_for_newer_topic(tmp_path: Path) -> None:
    save(tmp_path / "facts.jsonl", "Workshop is orange", "Preferred tea is Earl Grey")
    agent = core(tmp_path)
    agent.recall_for("workshop")
    old_output = agent.run(agent.runtime)  # Represents a pass begun before the topic changed.
    agent.recall_for("tea")
    agent.write_slot(
        status=old_output.status,
        summary=old_output.summary,
        report=old_output.report,
        notify_user=old_output.notify_user,
    )
    context = agent._slot_store.get_slot("compaction").context
    assert "Earl Grey" in context and "orange" not in context


def test_saved_conversation_notes_are_browsable_and_recalled_without_extra_files(tmp_path: Path) -> None:
    store = ConversationStore()
    store.append({"role": "user", "content": "Chosen project: orange robot"}, timestamp=1)
    assert store.compact(store.records(), "[summary] The robot project uses an orange shell.", 0)
    agent = core(tmp_path, store, compaction_enabled=False)
    agent.recall_for("robot project")
    assert "orange shell" in agent._slot_store.get_slot("compaction").context
    page = agent.memory_snapshot(kind="summary")
    assert page["total"] == 1 and page["memories"][0]["source"] == "Compacted conversation"
    assert list(tmp_path.iterdir()) == []
    assert agent.run(agent.runtime).summary == "Recall ready; compaction disabled"


def test_memory_browsing_does_not_change_recall(tmp_path: Path) -> None:
    save(tmp_path / "facts.jsonl", "Workshop is orange", "Preferred tea is Earl Grey")
    agent = core(tmp_path)
    agent.recall_for("workshop")
    page = agent.memory_snapshot(query="tea", limit=1)
    assert page["total"] == 1 and "Earl Grey" in page["memories"][0]["content"]
    assert page["recall"]["query"] == "workshop"
    assert "orange" in agent._slot_store.get_slot("compaction").context


def test_router_uses_supplied_recall_and_keeps_memory_writes(tmp_path: Path) -> None:
    save(tmp_path / "facts.jsonl", "Workshop is orange")
    agent = core(tmp_path)
    agent.recall_for("workshop")
    tree = RoutingTree(default_list(include_commands=True), tool_definitions, [], recalled_topic=agent.recalled_topic)
    root = tree.nodes["area"].decision
    assert "already recalled" in root.instructions
    assert "set_preference" in tree.tools and "get_preferences" in tree.tools
    assert any("remembered facts" in o.description for o in root.options if o.action == "reply")
    agent.set_paused(True)
    assert agent.recalled_topic is None
    tree = RoutingTree(default_list(include_commands=True), tool_definitions, [], recalled_topic=agent.recalled_topic)
    assert "already recalled" not in tree.nodes["area"].decision.instructions


def test_recall_is_prepared_before_speculation_and_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    order = []
    processor._before_context = lambda message: order.append(("recall", message["content"]))
    before = processor._build_messages

    def build(mode: bool) -> list:
        assert order and order[0] == ("recall", "workshop colour")
        order.append(("context", ""))
        return before(mode)

    processor._build_messages = build
    processor.llm_input_queue.put({"role": "user", "content": "workshop colour", "_allow_tools": False})

    def post(*_args: object, **_kwargs: object) -> Mock:
        processor.shutdown_event.set()
        response = Mock(status_code=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_lines.return_value = iter(())
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()
    assert order[0][0] == "recall" and len(order) > 1


def test_memory_api_paging_validation_and_private_read_only_access(tmp_path: Path) -> None:
    save(tmp_path / "facts.jsonl", "Workshop is orange", "Preferred tea is Earl Grey")
    engine = _FakeEngine()
    engine.compaction_agent = core(tmp_path)
    engine.autonomy_slots = engine.compaction_agent._slot_store
    engine.compaction_agent.recall_for("workshop")
    server = WebappServer(engine, port=0)
    server.start()

    def get(path: str, headers: dict | None = None) -> tuple:
        conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
        try:
            conn.request("GET", path, headers=headers or {})
            response = conn.getresponse()
            return response.status, json.loads(response.read()), response.getheader("Cache-Control")
        finally:
            conn.close()

    try:
        status, page, cache = get("/api/memory?limit=1")
        assert status == 200 and len(page["memories"]) == 1 and page["total"] == 2
        assert page["recall"]["query"] == "workshop" and cache == "no-store"
        assert get("/api/memory?offset=1&limit=1")[1]["memories"] != page["memories"]
        assert get("/api/memory?query=tea")[1]["total"] == 1
        assert get("/api/memory?limit=1000")[0] == 400
        assert get("/api/memory?offset=invalid")[0] == 400
        assert get("/api/memory?kind=invalid")[0] == 400
        assert get("/api/memory", {"Origin": "https://other.example"})[0] == 403
        assert get("/api/memory", {"Sec-Fetch-Site": "cross-site"})[0] == 403
        assert engine.compaction_agent.snapshot()["recall"]["query"] == "workshop"
        assert "orange" in get("/api/slots/compaction")[1]["context"]
    finally:
        engine.shutdown_event.set()
        server.shutdown()
