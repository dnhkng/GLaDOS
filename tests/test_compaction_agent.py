"""Rolling compaction preserves recent turns, time ranges and concurrent work."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest

from glados.autonomy.agents.compaction_agent import CompactionAgent, age_band
from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.subagent import SubagentConfig
from glados.core.conversation_store import ConversationStore

AgentFactory = Callable[..., tuple[CompactionAgent, Mock]]

NOW = 1791210000.0


@pytest.fixture
def make_agent(monkeypatch: pytest.MonkeyPatch) -> AgentFactory:
    monkeypatch.setattr("glados.autonomy.subagent.SubagentMemory", Mock())
    model = Mock(return_value="User prefers English. E4B is the default. Camera trails should last 0.6 seconds.")
    monkeypatch.setattr("glados.autonomy.agents.compaction_agent.llm_call", model)

    def make(store: ConversationStore | None = None, **kwargs: object) -> tuple[CompactionAgent, Mock]:
        defaults = {
            "llm_config": LLMConfig("http://unused", model="gemma-4-E4B"),
            "clock": lambda: NOW,
            "slot_store": None,
        }
        defaults.update(kwargs)
        return CompactionAgent(SubagentConfig("compaction", "Compaction"), conversation_store=store, **defaults), model

    return make


def history(old: int = 8, age: int = 10) -> ConversationStore:
    store = ConversationStore([{"role": "system", "content": "Keep the personality prompt."}])
    for i in range(old):
        store.append(
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Old {i}: " + "details " * 100}, NOW - age
        )
    for i in range(8):
        store.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"Recent {i}"}, NOW)
    return store


@pytest.mark.parametrize(
    ("age", "band"),
    [(0, 0), (3599, 0), (3600, 1), (14400, 2), (28800, 3), (86400, 4), (259200, 5), (604800, 6), (2592000, 7)],
)
def test_non_overlapping_bands(age: int, band: int) -> None:
    assert age_band(NOW - age, NOW) == band


def test_idle_and_small_history_do_not_infer(make_agent: AgentFactory) -> None:
    agent, model = make_agent(history(old=0))
    assert agent.tick().status == "monitoring"
    assert model.call_count == 0
    agent, _ = make_agent(None)
    assert "No conversation store" in agent.tick().summary
    agent, _ = make_agent(llm_config=None)
    assert "No LLM" in agent.tick().summary


def test_compaction_preserves_last_eight_and_system_prompt(make_agent: AgentFactory) -> None:
    store = history()
    recent = store.snapshot()[-8:]
    agent, model = make_agent(store)
    result = agent.tick()
    assert result.status == "compacted"
    assert not result.notify_user
    assert store.snapshot()[-8:] == recent
    assert store.snapshot()[0]["content"] == "Keep the personality prompt."
    assert len(store.snapshot()) == 10
    record = store.records()[1]
    assert record.summary_level == 0 and record.end_at == NOW - 10
    assert record.message["role"] == "assistant"  # Quoted memory is not a new system instruction.
    assert model.call_args[0][0].request_options["chat_template_kwargs"] == {"enable_thinking": False}
    assert agent.snapshot()["bands"][0]["summaries"] == 1


def test_tool_exchange_is_not_split_at_recent_boundary(make_agent: AgentFactory) -> None:
    store = history()
    store.append({"role": "user", "content": "What time?"}, NOW)
    store.append({"role": "assistant", "tool_calls": [{"id": "clock", "function": {"name": "get_time"}}]}, NOW)
    store.append({"role": "tool", "tool_call_id": "clock", "content": "12:34"}, NOW)
    for i in range(6):
        store.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"Extra {i}"}, NOW)
    recent = store.snapshot()[-9:]
    agent, _ = make_agent(store)
    assert agent.tick().status == "compacted"
    assert store.snapshot()[-9:] == recent


def test_tool_exchange_is_not_split_across_age_bands(make_agent: AgentFactory) -> None:
    store = ConversationStore()
    store.append({"role": "user", "content": "Check load. " * 80}, NOW - 3601)
    store.append({"role": "assistant", "tool_calls": [{"id": "cpu", "function": {"name": "cpu_load"}}]}, NOW - 3599)
    store.append({"role": "tool", "tool_call_id": "cpu", "content": "Load: 3.0 " * 80}, NOW - 3598)
    for record in history(old=0).records()[1:]:
        store.append(record.message, record.end_at)
    agent, _ = make_agent(store, token_threshold=100)
    assert agent.tick().status == "compacted"
    assert len(store.snapshot()) == 9
    assert store.records()[0].summary_level == 0
    assert store.records()[0].start_at == NOW - 3601
    assert not any(m.get("role") == "tool" for m in store.snapshot())


def test_distinct_age_bands_merge_without_duplication(make_agent: AgentFactory) -> None:
    store = history(old=8, age=20000)
    agent, model = make_agent(store)
    assert agent.tick().status == "compacted"
    assert store.records()[1].summary_level == 2
    assert agent.tick().status == "monitoring"
    model.reset_mock()
    agent._clock = lambda: NOW + 86400
    assert agent.tick().status == "compacted"  # Aging one summary only changes its metadata.
    assert store.records()[1].summary_level == 4
    model.assert_not_called()
    assert len([r for r in store.records() if r.summary_level is not None]) == 1


def test_concurrent_append_survives_compaction(make_agent: AgentFactory) -> None:
    store = history()
    agent, model = make_agent(store)

    def summarize(*args: object) -> str:
        store.append({"role": "user", "content": "New input arrived during inference"}, NOW + 1)
        return "The user prefers English and chose E4B."

    model.side_effect = summarize
    assert agent.tick().status == "compacted"
    assert store.snapshot()[-1]["content"] == "New input arrived during inference"


def test_concurrent_edit_revokes_replacement(make_agent: AgentFactory) -> None:
    store = history()
    agent, model = make_agent(store)

    def summarize(*args: object) -> str:
        store.modify_message(1, {"content": "Corrected while summarizing"})
        return "Old note."

    model.side_effect = summarize
    assert "History changed" in agent.tick().summary
    assert store.snapshot()[1]["content"] == "Corrected while summarizing"
    assert not any(r.summary_level is not None for r in store.records())


def test_failure_preserves_every_record(make_agent: AgentFactory) -> None:
    store = history()
    before = store.records()
    agent, model = make_agent(store)
    model.return_value = None
    assert agent.tick().status == "error"
    assert store.records() == before


def test_busy_interactive_turn_defers_work(make_agent: AgentFactory) -> None:
    agent, model = make_agent(history(), interactive_busy=lambda: True)
    assert "Waiting" in agent.tick().summary
    model.assert_not_called()


def test_large_inputs_are_read_in_full_with_bounded_calls(make_agent: AgentFactory) -> None:
    store = history()
    store.modify_message(1, {"content": "HEAD " + "long information " * 2000 + " TAIL"})
    agent, model = make_agent(store)
    assert agent.tick().status == "compacted"
    prompts = [call.args[2] for call in model.call_args_list]
    assert any("HEAD" in p for p in prompts) and any("TAIL" in p for p in prompts)
    assert all(len(p) < 3700 for p in prompts)
    assert model.call_count > 2


def test_manual_run_can_compact_small_old_batch(make_agent: AgentFactory) -> None:
    store = history(old=1)
    agent, model = make_agent(store)
    assert agent.tick().status == "monitoring"
    agent._running = True
    agent.request_tick()
    assert agent.tick().status == "compacted"
    assert model.called


def test_persistence_restores_summary_ranges_and_recent_history(tmp_path: Path, make_agent: AgentFactory) -> None:
    path = tmp_path / "history.json"
    store = ConversationStore([{"role": "system", "content": "Old prompt"}], path=path)
    for record in history().records()[1:]:
        store.append(record.message, record.end_at)
    agent, _ = make_agent(store)
    assert agent.tick().status == "compacted"
    restored = ConversationStore([{"role": "system", "content": "New prompt"}], path=path)
    assert restored.snapshot()[0]["content"] == "New prompt"
    assert restored.records()[1:] == store.records()[1:]
    assert path.stat().st_mode & 0o777 == 0o600
    assert all("summary_level" not in m and "start_at" not in m for m in restored.snapshot())


def test_corrupt_history_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("invalid JSON")
    store = ConversationStore(path=path)
    store.append({"role": "user", "content": "Fresh session"})
    assert path.read_text() == "invalid JSON"
