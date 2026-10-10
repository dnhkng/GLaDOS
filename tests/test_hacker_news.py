from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from glados.autonomy.agents.hacker_news import HackerNewsSubagent
from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.subagent import SubagentConfig
from glados.autonomy.subagent_memory import SubagentMemory


def test_verdicts_survive_restart_and_only_unpublished_relevant_stories_remain(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "glados.autonomy.subagent.SubagentMemory",
        lambda agent_id, max_entries: SubagentMemory(agent_id, max_entries, tmp_path),
    )
    judge = Mock(
        side_effect=[
            SimpleNamespace(relevant=True, importance=0.8, summary="Good"),
            SimpleNamespace(relevant=False, importance=0.1, summary="Skip"),
            SimpleNamespace(relevant=True, importance=0.7, summary="Also good"),
        ]
    )
    monkeypatch.setattr("glados.autonomy.agents.hacker_news.llm_decide_sync", judge)
    stories = [{"id": i, "title": str(i), "score": 300} for i in range(3)]

    def agent():
        result = HackerNewsSubagent(
            SubagentConfig("hn", "HN"), slot_store=Mock(), top_n=1, llm_config=LLMConfig("http://test")
        )
        monkeypatch.setattr(result, "_fetch_top_stories", lambda: stories)
        return result

    first = (lambda a: a.run(a.runtime))(agent())
    assert first.raw["stories"][0]["id"] == 0
    second = (lambda a: a.run(a.runtime))(agent())
    assert second.raw["stories"][0]["id"] == 2
    assert (lambda a: a.run(a.runtime))(agent()).status == "idle"
    assert judge.call_count == 3


def test_legacy_reported_and_rejected_stories_are_not_judged_again(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "glados.autonomy.subagent.SubagentMemory",
        lambda agent_id, max_entries: SubagentMemory(agent_id, max_entries, tmp_path),
    )
    memory = SubagentMemory("hn", 100, tmp_path)
    memory.set("hn_1", {"id": 1, "title": "Already reported", "_reported": True})
    memory.set("hn_2", {"id": 2, "title": "Already rejected", "_relevant": False})
    judge = Mock()
    monkeypatch.setattr("glados.autonomy.agents.hacker_news.llm_decide_sync", judge)
    agent = HackerNewsSubagent(SubagentConfig("hn", "HN"), slot_store=Mock(), llm_config=LLMConfig("http://test"))
    monkeypatch.setattr(agent, "_fetch_top_stories", lambda: [{"id": 1, "title": "Already reported"}])

    assert agent.run(agent.runtime).status == "idle"
    judge.assert_not_called()
    assert SubagentMemory("hn", 100, tmp_path).list_unshown() == []
