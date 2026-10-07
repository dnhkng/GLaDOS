from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from glados.autonomy.agents.hacker_news import HackerNewsSubagent
from glados.autonomy.subagent import SubagentConfig
from glados.autonomy.subagent_memory import SubagentMemory
from glados.autonomy.llm_client import LLMConfig


def test_verdicts_survive_restart_and_only_unpublished_relevant_stories_remain(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("glados.autonomy.subagent.SubagentMemory",
        lambda agent_id, max_entries: SubagentMemory(agent_id, max_entries, tmp_path))
    judge = Mock(side_effect=[SimpleNamespace(relevant=True, importance=.8, summary="Good"),
        SimpleNamespace(relevant=False, importance=.1, summary="Skip"),
        SimpleNamespace(relevant=True, importance=.7, summary="Also good")])
    monkeypatch.setattr("glados.autonomy.agents.hacker_news.llm_decide_sync", judge)
    stories = [{"id": i, "title": str(i), "score": 300} for i in range(3)]
    def agent():
        result = HackerNewsSubagent(SubagentConfig("hn", "HN"), slot_store=Mock(), top_n=1, llm_config=LLMConfig("http://test"))
        monkeypatch.setattr(result, "_fetch_top_stories", lambda: stories)
        return result
    first = agent().tick()
    assert first.raw["stories"][0]["id"] == 0
    second = agent().tick()
    assert second.raw["stories"][0]["id"] == 2
    assert agent().tick().status == "idle"
    assert judge.call_count == 3
