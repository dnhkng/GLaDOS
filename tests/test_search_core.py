"""Requested research loops, validates evidence and hands source links back to Central Core."""

from collections.abc import Iterator
import json
import queue
import threading
from unittest.mock import MagicMock, Mock

import pytest

from glados.autonomy.agents.search_agent import SearchAgent, SearchConfig, forecast_date_matches, parse_sources
from glados.autonomy.llm_client import LLMConfig, llm_call
from glados.autonomy.slots import TaskSlotStore
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados
from glados.core.tool_executor import ToolExecutor
from glados.mcp.search_results import SEARCH_TOOL
from tests.test_speech_markup import make_processor

PRICE = "The module is available now for $30."
PURPOSE = "It adapts Compute Module 5 for digital signage."


def source(index: int, text: str) -> str:
    return (
        f"Title: Official source {index}\nURL: https://example.org/{index}\nPublished: 2026-10-06\nHighlights:\n{text}"
    )


def review(action: str = "done", findings: list | None = None, **kwargs: object) -> str:
    return json.dumps(
        {"action": action, "findings": findings or [{"source_id": 1, "quote": PRICE}], "gaps": [], **kwargs}
    )


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> tuple[SearchAgent, Mock, Mock]:
    search = Mock(return_value=source(1, PRICE + " " + PURPOSE))
    model = Mock(return_value=review())
    monkeypatch.setattr("glados.autonomy.agents.search_agent.llm_call", model)
    agent = SearchAgent(SearchConfig(), LLMConfig(url="http://unused", model="E4B"), search, slot_store=TaskSlotStore())
    agent._read_page = Mock(return_value=None)
    return agent, search, model


def test_sufficient_evidence_stops_after_one_round_and_supplies_slot(core: tuple) -> None:
    agent, search, model = core
    result = json.loads(agent.research({"query": "module price", "objective": "Find its price"}))
    assert result["status"] == "done"
    assert result["findings"] == [{"source_id": 1, "quote": PRICE, "url": "https://example.org/1"}]
    assert result["sources"][0]["url"] == "https://example.org/1"
    search.assert_called_once()
    model.assert_called_once()
    assert model.call_args.args[0].owner == "Search" and model.call_args.args[0].lane == "autonomy"
    slot = agent._slot_store.get_slot("search")
    assert not slot.notify_user and PRICE in slot.context
    assert "https://example.org/1" in Glados._format_slots(Mock(autonomy_slots=agent._slot_store))


def test_agent_follows_up_on_gaps_and_combines_verified_sources(core: tuple) -> None:
    agent, search, model = core
    search.side_effect = [source(1, PRICE), source(2, PURPOSE)]
    model.side_effect = [
        review("search", gaps=["Purpose missing"], query="official module purpose", objective="Find purpose"),
        review(findings=[{"source_id": 2, "quote": PURPOSE}]),
    ]
    result = json.loads(agent.research({"query": "module", "objective": "Find its price and purpose"}))
    assert result["status"] == "done" and result["rounds"] == 2
    assert len(result["findings"]) == 2 and len(result["sources"]) == 2
    assert search.call_args_list[1].args[0]["query"] == "official module purpose"
    assert search.call_args_list[1].args[0]["objective"] == "Find purpose"
    assert json.loads(model.call_args.args[2])["objective"] == "Find its price and purpose"


def test_favorite_sites_focus_query_and_review_without_restricting_followups(core: tuple) -> None:
    agent, search, model = core
    agent.preferences.update({"weather": [], "news": ["favorite.example.org/news"], "reddit": [], "general": []})
    model.side_effect = [
        review("search", gaps=["Purpose"], query="module official release", objective="Purpose"),
        review(findings=[{"source_id": 1, "quote": PURPOSE}]),
    ]
    report = json.loads(agent.research({"query": "module news", "objective": "Price and purpose"}))
    assert "site:favorite.example.org/news" in search.call_args_list[0].args[0]["query"]
    assert search.call_args_list[1].args[0]["query"] == "module official release"
    assert json.loads(model.call_args.args[2])["preferred_sources"] == {"news": ["favorite.example.org/news"]}
    assert "preferences, not an allowlist" in model.call_args.args[1]
    assert report["status"] == "done"


def test_news_reads_both_favorite_pages_before_review_without_web_search(core: tuple) -> None:
    agent, search, model = core
    agent.preferences.update({"weather": [], "news": ["news.ycombinator.com", "huggingnews.com"],
                              "reddit": [], "general": []})
    agent._read_page.side_effect = [
        {"url": "https://news.ycombinator.com/", "title": "Hacker News", "published": None,
         "kind": "news_page", "retrieved_at": "2026-10-06T16:00:00Z", "excerpt": PRICE},
        {"url": "https://huggingnews.com/", "title": "HuggingNews", "published": None,
         "kind": "news_page", "retrieved_at": "2026-10-06T16:00:00Z", "excerpt": PURPOSE},
    ]
    model.return_value = review(findings=[{"source_id": 1, "quote": PRICE}, {"source_id": 2, "quote": PURPOSE}])
    result = json.loads(agent.research({"query": "check the news today"}))
    assert result["status"] == "done" and len(result["findings"]) == 2
    search.assert_not_called()
    assert [call.args[0] for call in agent._read_page.call_args_list] == ["news.ycombinator.com", "huggingnews.com"]
    assert agent.snapshot()["pages_read"] == 2
    assert {s["kind"] for s in result["sources"]} == {"news_page"}
    assert all(s["published"] is None for s in result["sources"])
    payload = json.loads(model.call_args.args[2])
    assert payload["sources"][1]["retrieved_at"] == "2026-10-06T16:00:00Z"


def test_news_page_gaps_use_search_and_preserve_page_evidence(core: tuple) -> None:
    agent, search, model = core
    agent.preferences.update({"weather": [], "news": ["favorite.example.org/news"], "reddit": [], "general": []})
    agent._read_page.return_value = {"url": "https://favorite.example.org/news", "title": "News",
                                    "published": None, "kind": "news_page", "excerpt": PRICE}
    search.return_value = source(2, PURPOSE)
    model.side_effect = [review("search", gaps=["Purpose missing"], query="module official release", objective="Purpose"),
                         review(findings=[{"source_id": 2, "quote": PURPOSE}])]
    result = json.loads(agent.research({"query": "module news", "objective": "Price and purpose"}))
    assert result["status"] == "done" and len(result["findings"]) == 2
    agent._read_page.assert_called_once()
    search.assert_called_once()
    assert search.call_args.args[0]["query"] == "module official release"


def test_news_page_failure_falls_back_to_search(core: tuple) -> None:
    agent, search, _ = core
    agent._read_page.side_effect = TimeoutError("Page unavailable")
    result = json.loads(agent.research({"query": "latest news"}))
    assert result["status"] == "done"
    search.assert_called_once()
    assert agent.snapshot()["pages_read"] == 0 and agent.snapshot()["page_errors"]


def test_reading_news_pages_does_not_count_as_already_searching_the_query(core: tuple) -> None:
    agent, search, model = core
    agent.preferences.update({"weather": [], "news": ["favorite.example.org/news"], "reddit": [], "general": []})
    agent._read_page.return_value = {"url": "https://favorite.example.org/news", "title": "News",
                                    "published": None, "kind": "news_page", "excerpt": PRICE}
    search.return_value = source(2, PURPOSE)
    model.side_effect = [review("search", gaps=["Purpose missing"], query="module news", objective="Purpose"),
                         review(findings=[{"source_id": 2, "quote": PURPOSE}])]
    result = json.loads(agent.research({"query": "module news", "objective": "Price and purpose"}))
    assert result["status"] == "done" and len(result["findings"]) == 2
    search.assert_called_once()
    assert search.call_args.args[0]["query"] == "module news"
    assert json.loads(model.call_args_list[0].args[2])["previous_queries"] == []


@pytest.fixture
def dated_clock(monkeypatch: pytest.MonkeyPatch) -> dict:
    reading = {
        "date": "2026-10-06",
        "time": "23:59:50",
        "weekday": "Tuesday",
        "timezone": "Europe/Berlin",
        "datetime": "2026-10-06T23:59:50+02:00",
        "utc_offset": "+0200",
        "source": "system clock",
    }
    monkeypatch.setattr("glados.autonomy.agents.search_agent.current_time", lambda: reading)
    return reading


def test_weather_tomorrow_is_pinned_to_clock_and_followups_keep_date(core: tuple, dated_clock: dict) -> None:
    agent, search, model = core
    forecast = "Wednesday October 7, 2026: Sunny, high 16°C, low 8°C, chance of rain 10%."
    search.side_effect = [source(1, "Thursday: Maximum temperature 16°C."), source(2, forecast)]
    model.side_effect = [
        review(
            "search",
            findings=[{"source_id": 1, "quote": "Thursday: Maximum temperature 16°C."}],
            gaps=["No dated forecast"],
            query="Munich official weather forecast",
            objective="Get weather",
        ),
        review(findings=[{"source_id": 2, "quote": forecast}]),
    ]
    report = json.loads(
        agent.research({"query": "Munich weather tomorrow", "objective": "Weather in Munich for the next day"})
    )
    assert report["status"] == "done" and report["target_dates"] == ["2026-10-07"]
    assert report["findings"][0]["date"] == "2026-10-07" and "Thursday" not in json.dumps(report["findings"])
    assert "Wednesday 2026-10-07" in search.call_args_list[0].args[0]["query"]
    assert "2026-10-07" in search.call_args_list[1].args[0]["query"]
    assert json.loads(model.call_args.args[2])["clock"] == dated_clock


def test_undated_weather_never_falls_back_to_wrong_day_excerpts(core: tuple, dated_clock: dict) -> None:
    agent, search, model = core
    misleading = "Tomorrow at 15:00: maximum 24°C, minimum 21°C, wind chill 22°C."
    search.return_value = source(1, misleading)
    model.return_value = review(findings=[{"source_id": 1, "quote": misleading}])
    report = json.loads(agent.research({"query": "Munich weather tomorrow"}))
    assert report["status"] == "partial" and report["findings"] == []
    assert "24°C" not in json.dumps(report)
    assert "2026-10-07" in " ".join(report["gaps"])
    assert search.call_count == 3


@pytest.mark.parametrize(
    "quote,valid",
    [
        ("Wednesday 2026-10-07: High 16°C, low 8°C.", True),
        ("Wednesday, Oct 7: High 16°C, low 8°C.", True),
        ("7 October 2026: High 16°C, low 8°C.", True),
        ("07.10.2026: High 16°C, low 8°C.", True),
        ("October 7,2026: High 16°C, low 8°C.", True),
        ("Thursday 2026-10-07: High 16°C.", False),
        ("Thursday: High 16°C.", False),
        ("October 7, 2025: High 16°C.", False),
        ("Tomorrow: High 16°C.", False),
        ("2026-10-08: High 16°C.", False),
        ("Published: 2026-10-07\nHigh 16°C.", False),
        ("https://weather.example/2026-10-07 High 16°C.", False),
    ],
)
def test_forecast_date_labels_match_requested_day(quote: str, valid: bool) -> None:
    assert forecast_date_matches(quote, "2026-10-07") is valid


def test_invented_quotes_and_unknown_citations_never_become_verified_findings(core: tuple) -> None:
    agent, _, model = core
    model.return_value = review(
        findings=[{"source_id": 1, "quote": "The module costs $300."}, {"source_id": 99, "quote": PRICE}]
    )
    result = json.loads(agent.research({"query": "module price"}))
    assert result["status"] == "partial"
    assert "$300" not in json.dumps(result)
    assert all(f["source_id"] == 1 for f in result["findings"])
    assert any("excerpts" in gap for gap in result["gaps"])


def test_citation_urls_are_attached_and_not_treated_as_missing_evidence(core: tuple) -> None:
    agent, search, model = core
    model.return_value = review(
        "search",
        gaps=["The specific source link for the price is not provided in the excerpt."],
        query="find price source link",
    )
    result = json.loads(agent.research({"query": "module price"}))
    assert result["status"] == "done" and result["gaps"] == []
    assert result["findings"][0]["url"] == "https://example.org/1"
    search.assert_called_once()


@pytest.mark.parametrize("mode", ["repeat", "round_limit", "bad_json", "empty", "failure"])
def test_loop_failure_modes_are_bounded_and_reported(core: tuple, mode: str) -> None:
    agent, search, model = core
    if mode == "repeat":
        model.return_value = review("search", query="module price", gaps=["Missing detail"])
    elif mode == "round_limit":
        model.side_effect = [review("search", query=f"query {i}", gaps=["Missing detail"]) for i in range(3)]
    elif mode == "bad_json":
        model.return_value = "not JSON"
    elif mode == "empty":
        search.return_value = "No results"
    else:
        search.side_effect = RuntimeError("service unavailable")
    result = json.loads(agent.research({"query": "module price"}))
    assert result["status"] in {"partial", "error"}
    assert search.call_count <= 3 and model.call_count <= 3
    assert result["gaps"]
    assert agent.snapshot()["status"] != "researching"


def test_paused_and_cancelled_research_does_not_publish_evidence(core: tuple) -> None:
    agent, search, _ = core
    agent.set_paused(True)
    assert json.loads(agent.research({"query": "module"}))["status"] == "error"
    search.assert_not_called()
    agent.set_paused(False)
    assert json.loads(agent.research({"query": "module"}, cancelled=lambda: True))["status"] == "cancelled"
    assert agent._slot_store.get_slot("search").context is None


def test_cancellation_during_review_is_permanent_even_after_resume(core: tuple) -> None:
    agent, _, model = core

    def pause_during_review(*_args: object, **_kwargs: object) -> str:
        agent.set_paused(True)
        agent.set_paused(False)
        return review()

    model.side_effect = pause_during_review
    result = json.loads(agent.research({"query": "module"}))
    assert result["status"] == "cancelled"
    assert agent._slot_store.get_slot("search").context is None


def test_old_background_result_stays_in_report_but_not_current_context(core: tuple) -> None:
    agent, _, _ = core
    agent.research({"query": "module"}, context_current=lambda: False)
    slot = agent._slot_store.get_slot("search")
    assert PRICE in slot.report and slot.context is None
    agent.research({"query": "module"})
    agent.clear_context()
    assert agent._slot_store.get_slot("search").context is None
    assert PRICE in agent._slot_store.get_slot("search").report


def test_deadline_stops_before_more_model_calls(core: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    agent, search, model = core
    clock = [0.0]
    monkeypatch.setattr("glados.autonomy.agents.search_agent.time.monotonic", lambda: clock[0])

    def slow(*_args: object) -> str:
        clock[0] += 100
        return source(1, PRICE)

    search.side_effect = slow
    result = json.loads(agent.research({"query": "module"}))
    search.assert_called_once()
    model.assert_not_called()
    assert "time limit" in " ".join(result["gaps"])


def test_bounded_review_and_report_preserve_valid_json_and_sources(core: tuple) -> None:
    agent, search, model = core
    agent.settings.max_evidence_chars = 3500
    agent.settings.max_report_chars = 1800
    search.return_value = source(1, PRICE + '\n"\\' * 5000) + "\n\n" + source(2, PURPOSE * 1000)
    result_text = agent.research({"query": "module", "objective": "Price and purpose"})
    assert len(result_text) <= 1800
    assert len(model.call_args.args[2]) <= 3500
    assert "https://example.org/1" in model.call_args.args[2]
    assert json.loads(result_text)["sources"][0]["url"] == "https://example.org/1"


def test_bad_source_urls_are_ignored_and_missing_sources_are_not_invented() -> None:
    assert parse_sources("Title: bad\nURL: javascript:alert(1)\nText") == []
    assert parse_sources("Title: bad\nURL: https://[invalid/\nText") == []
    assert parse_sources("Text without a URL") == []


def test_bounds_include_escaped_metadata_at_maximum_source_count(core: tuple) -> None:
    agent, _, _ = core
    agent.settings.max_evidence_chars = 3500
    agent.settings.max_report_chars = 1800
    agent._state.update(query="\x00" * 1000, objective="\x00" * 2000)
    agent._state["preferred_sources"] = {
        category: [f"site{i}.org/" + "a" * 180 for i in range(8)] for category in ("weather", "general")
    }
    sources = [
        {"source_id": i, "title": "\x00" * 200, "url": f"https://example.org/{i}", "excerpt": PRICE} for i in range(12)
    ]
    assert len(agent._review_input(sources, [], set(), 0)) <= 3500
    report = agent._pack_report("done", sources, [{"source_id": 0, "quote": PRICE}], ["\x00" * 200] * 4)
    assert len(report) <= 1800
    assert json.loads(report)["status"] == "partial"
    assert "size limit" in " ".join(json.loads(report)["gaps"])


def test_report_marks_omitted_evidence_partial(core: tuple) -> None:
    agent, _, _ = core
    agent.settings.max_report_chars = 1800
    agent._state.update(query="module", objective="price and purpose")
    sources = [{"source_id": i, "title": "Official", "url": f"https://example.org/{i}"} for i in range(6)]
    findings = [{"source_id": i, "quote": "Evidence " * 50} for i in range(6)]
    text = agent._pack_report("done", sources, findings, [])
    report = json.loads(text)
    assert len(text) <= 1800 and report["status"] == "partial"
    assert 0 < len(report["findings"]) < 6
    assert "size limit" in " ".join(report["gaps"])


def test_tool_executor_delegates_to_core_and_returns_completed_evidence(core: tuple) -> None:
    agent, search, _ = core
    replies, calls = queue.Queue(), queue.Queue()
    active, shutdown = threading.Event(), threading.Event()
    active.set()
    manager = Mock()
    executor = ToolExecutor(
        replies,
        queue.Queue(),
        calls,
        active,
        shutdown,
        mcp_manager=manager,
        tool_config={"search_agent": agent},
        autonomy_enabled=lambda: False,
    )
    calls.put({"id": "research", "function": {"name": SEARCH_TOOL, "arguments": {"query": "module price"}}})
    worker = threading.Thread(target=executor.run)
    worker.start()
    try:
        reply = replies.get(timeout=3)
        assert json.loads(reply["content"])["status"] == "done"
        assert reply["_tool_reply_context"]["name"] == SEARCH_TOOL
        search.assert_called_once()
        manager.call_tool.assert_not_called()
    finally:
        shutdown.set()
        worker.join(2)


def test_main_reply_gets_specific_research_handoff_instruction(core: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    agent, _, _ = core
    processor = make_processor()
    processor._conversation_store = ConversationStore([{"role": "user", "content": "module price"}])
    processor.llm_input_queue.put(
        {
            "role": "tool",
            "tool_call_id": "research",
            "_allow_tools": False,
            "content": agent.research({"query": "module price"}),
            "_tool_reply_context": {"name": SEARCH_TOOL},
        }
    )
    response = Mock(status_code=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_lines.return_value = iter(())

    def post(*_args: object, **kwargs: object) -> Mock:
        processor.shutdown_event.set()
        prompt = json.dumps(kwargs["json"]["messages"])
        assert "Search Core has finished" in prompt and "source_id" in prompt
        assert "https://example.org/1" in prompt and PRICE in prompt
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()


@pytest.mark.parametrize(
    "citation",
    ["https://example.org/price", "(https://example.org/price)", "[Official price](https://example.org/price)"],
)
def test_citation_urls_survive_response_sentence_cleanup(citation: str) -> None:
    processor = make_processor()
    processor._process_sentence_for_tts([f"The module costs $30. Source: {citation} (an aside)."])
    text = processor.tts_input_queue.get_nowait().text
    assert "https://example.org/price" in text
    assert "an aside" not in text


def test_streamed_citation_is_not_split_at_scheme_or_hostname_punctuation(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    processor.llm_input_queue.put({"role": "user", "content": "Give a source link", "_allow_tools": False})

    def lines(chunk_size: int = 1) -> Iterator[bytes]:
        for char in "Source: https://example.org/price. Complete.":
            yield b"data: " + json.dumps({"choices": [{"delta": {"content": char}}]}).encode()
        processor.shutdown_event.set()

    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.side_effect = lines
    monkeypatch.setattr("glados.core.llm_processor.requests.post", lambda *a, **kw: response)
    processor.run()
    answer = "".join(item.text for item in processor.tts_input_queue.queue)
    assert "https://example.org/price" in answer


def test_background_search_uses_core_and_saves_one_notification(core: tuple) -> None:
    agent, search, _ = core
    replies, calls = queue.Queue(), queue.Queue()
    active, shutdown = threading.Event(), threading.Event()
    active.set()
    tasks, manager = Mock(), Mock()
    executor = ToolExecutor(
        replies,
        queue.Queue(),
        calls,
        active,
        shutdown,
        mcp_manager=manager,
        tool_config={"search_agent": agent, "task_manager": tasks},
        autonomy_enabled=lambda: True,
    )
    calls.put({"id": "background", "function": {"name": SEARCH_TOOL, "arguments": {"query": "module"}}})
    worker = threading.Thread(target=executor.run)
    worker.start()
    try:
        assert json.loads(replies.get(timeout=3)["content"])["status"] in {"running", "queued"}
        result = tasks.submit.call_args.args[2]()
        assert result.status == "done" and result.notify_user
        assert json.loads(result.report)["findings"]
        assert agent._slot_store.get_slot("search") is None  # Managed results belong to their task slot.
        search.assert_called_once()
        manager.call_tool.assert_not_called()
    finally:
        shutdown.set()
        worker.join(2)


@pytest.mark.parametrize("admitted_at,expected_timeout", [(13, 2), (16, None)])
def test_review_deadline_includes_waiting_for_inference_capacity(
    admitted_at: float,
    expected_timeout: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    scheduler = MagicMock()
    scheduler.lease.return_value.__enter__.side_effect = lambda: clock.__setitem__(0, admitted_at)
    monkeypatch.setattr("glados.autonomy.llm_client.time.monotonic", lambda: clock[0])
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "Evidence"}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr("glados.autonomy.llm_client.requests.post", post)
    result = llm_call(LLMConfig("http://test", scheduler=scheduler, timeout=15, deadline=15), "system", "evidence")
    if expected_timeout is None:
        post.assert_not_called()
        assert result is None
    else:
        assert result == "Evidence"
        assert post.call_args.kwargs["timeout"] == expected_timeout
