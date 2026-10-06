"""Large search results must not overflow the local model's reply context."""

import asyncio
from copy import deepcopy
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import requests

from glados.core.context_budget import reduce_request_context
from glados.core.conversation_store import ConversationStore
from glados.mcp import MCPManager
from glados.mcp.manager import MCPToolError
from glados.mcp.search_results import SEARCH_TOOL, compact_search_results
from tests.test_speech_markup import make_processor


def search_response() -> str:
    return "\n\n---\n\n".join(
        f"Title: Source {i}\nURL: https://example.org/{i}\nPublished: 2026-10-06\nHighlights:\n" + "Evidence " * 2000
        for i in range(3)
    )


def test_search_excerpts_retain_all_sources_and_dates() -> None:
    original = search_response()
    result = compact_search_results(original)
    assert len(result) <= 3500
    for i in range(3):
        assert f"Title: Source {i}" in result
        assert f"URL: https://example.org/{i}" in result
    assert result.count("Published: 2026-10-06") == 3
    assert result.count("Evidence") >= 3
    assert "shortened" in result
    assert compact_search_results("Small result") == "Small result"


def test_only_search_output_is_bounded_and_errors_are_preserved() -> None:
    manager = MCPManager([])
    content = search_response()
    session = SimpleNamespace(call_tool=AsyncMock(return_value={"content": [{"type": "text", "text": content}]}))
    manager._sessions["internet_search"] = session
    result = asyncio.run(manager._call_tool_async("internet_search", "web_search_exa", {}))
    assert len(result) <= 3500
    assert asyncio.run(manager._call_tool_async("internet_search", "another_tool", {})) == content.strip()
    session.call_tool.return_value = {"isError": True, "content": [{"type": "text", "text": "Rate limited"}]}
    with pytest.raises(MCPToolError, match="Rate limited"):
        asyncio.run(manager._call_tool_async("internet_search", "web_search_exa", {}))


def history() -> list[dict]:
    return [
        {"role": "system", "content": "Preserve instructions"},
        {"role": "user", "content": "Old question " * 100},
        {"role": "assistant", "tool_calls": [{"id": "old", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": "Old evidence " * 100},
        {"role": "assistant", "content": "Old answer"},
        {"role": "system", "content": "Current emotion and clock"},
        {"role": "user", "content": "Search for Raspberry Pi news"},
        {"role": "assistant", "tool_calls": [{"id": "search", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "search", "content": "New evidence " * 1000},
    ]


def test_context_reduction_preserves_current_exchange_and_original_history() -> None:
    original = history()
    saved = deepcopy(original)
    reduced = reduce_request_context(original, 5000, 4096)
    assert original == saved
    assert reduced[0] == original[0]
    assert original[5] in reduced
    assert original[6] in reduced
    assert original[7] in reduced
    assert reduced[-1]["tool_call_id"] == "search"
    assert len(json.dumps(reduced)) < len(json.dumps(original))
    assert not any(m.get("tool_call_id") == "old" for m in reduced)
    assert not any(c.get("id") == "old" for m in reduced for c in m.get("tool_calls", []))


def overflow() -> requests.Response:
    response = requests.Response()
    response.status_code = 400
    response._content = json.dumps(
        {
            "error": {
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 5000,
                "n_ctx": 4096,
            }
        }
    ).encode()
    response._content_consumed = True
    return response


def test_reply_retries_overflow_with_smaller_context(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    original = history()
    saved = deepcopy(original)
    data = {"model": "test", "messages": deepcopy(original), "stream": True}
    success = requests.Response()
    success.status_code = 200
    calls = []

    def post(*args: object, **kwargs: object) -> requests.Response:
        calls.append(deepcopy(kwargs["json"]))
        return overflow() if len(calls) == 1 else success

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    result = processor._post_with_context_recovery(str(processor.completion_url), data, original, False)
    assert result is success
    assert len(calls) == 2
    assert len(json.dumps(calls[1]["messages"])) < len(json.dumps(calls[0]["messages"]))
    assert original == saved
    assert processor.last_context()["message_count"] == len(calls[1]["messages"])


def test_other_http_400_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    invalid = requests.Response()
    invalid.status_code = 400
    invalid._content = b'{"error":{"type":"invalid_request_error"}}'
    post = Mock(return_value=invalid)
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    data = {"messages": history()}
    assert processor._post_with_context_recovery(str(processor.completion_url), data, history(), False) is invalid
    assert post.call_count == 1


def test_context_recovery_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    post = Mock(side_effect=lambda *a, **kw: overflow())
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    data = {"messages": history()}
    response = processor._post_with_context_recovery(str(processor.completion_url), data, history(), False)
    assert response.status_code == 400
    assert post.call_count <= 3


def test_tool_continuation_streams_answer_after_context_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    original = history()[:-1]
    processor._conversation_store = ConversationStore(original)
    evidence = {"role": "tool", "tool_call_id": "search", "content": "Fresh news " * 1500}
    processor.llm_input_queue.put({**evidence, "_allow_tools": False})
    success = MagicMock(status_code=200)
    success.__enter__.return_value = success
    success.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"Fresh search results are available."}}]}',
        b"data: [DONE]",
    ]
    post = Mock(side_effect=[overflow(), success])
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    worker = threading.Thread(target=processor.run)
    worker.start()
    speech = []
    try:
        while (item := processor.tts_input_queue.get(timeout=3)).text != "<EOS>":
            speech.append(item.text)
    finally:
        processor.shutdown_event.set()
        worker.join(2)
    assert "Fresh search results are available." in "".join(speech)
    assert not any("HTTP status" in text for text in speech)
    assert post.call_count == 2
    assert processor.tool_calls_queue.empty()
    assert processor._conversation_store.snapshot() == [*original, evidence]


@pytest.mark.parametrize("status", ["queued", "running", "done", "partial", "error", None])
def test_search_acknowledgement_distinguishes_pending_work_from_findings(
    status: str | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = make_processor()
    original = history()[:-1]
    processor._conversation_store = ConversationStore(original)
    processor._inference_scheduler = Mock()
    content = json.dumps({"status": status, "task_id": "task_search_demo"}) if status else "Verified source excerpt"
    message = {"role": "tool", "tool_call_id": "search", "content": content}
    processor.llm_input_queue.put({**message,
        "_allow_tools": False, "_tool_reply_context": {"name": "mcp.internet_search.web_search_exa"}})
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.return_value = [b'data: {"choices":[{"delta":{"content":"Search status received."}}]}',
                                        b'data: [DONE]']
    post = Mock(return_value=response)
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    speech = []
    worker = threading.Thread(target=processor.run)
    worker.start()
    try:
        while (text := processor.tts_input_queue.get(timeout=3).text) != "<EOS>":
            speech.append(text)
    finally:
        processor.shutdown_event.set()
        worker.join(2)
    assert processor._conversation_store.snapshot() == [*original, message]
    assert processor.tool_calls_queue.empty()
    if status in {"queued", "running"}:
        assert speech == ["Your search is queued." if status == "queued" else "I'm checking that now."]
        post.assert_not_called()
        processor._inference_scheduler.acquire.assert_not_called()
    else:
        assert "Search status received." in "".join(speech)
        prompt = "\n".join(m.get("content", "") for m in post.call_args.kwargs["json"]["messages"])
        assert "internet search has completed" in prompt


@pytest.mark.parametrize("calls_tool", [True, False])
def test_required_search_planning_does_not_speak_date_questions(
    calls_tool: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = make_processor()
    processor._conversation_store = ConversationStore([])
    processor.router = Mock()
    processor.router.score.return_value = {"action": "plan", "strategy": "hierarchical", "list_id": "speech",
                                          "revision": 1, "settings_revision": 1, "tool_scope": [SEARCH_TOOL]}
    processor._build_tools = lambda _: [{"type": "function", "function": {"name": SEARCH_TOOL}}]
    processor.llm_input_queue.put({"role": "user", "content": "Weather in Munich tomorrow?"})
    chunks = [{"content": "What is the exact date for tomorrow? I will search."}]
    if calls_tool:
        chunks.append({"tool_calls": [{"index": 0, "id": "weather", "type": "function", "function": {
            "name": SEARCH_TOOL, "arguments": '{"query":"Munich weather 2026-10-07"}'}}]})
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.return_value = [b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode()
                                        for delta in chunks] + [b"data: [DONE]"]
    post = Mock(return_value=response)
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    worker = threading.Thread(target=processor.run)
    worker.start()
    speech = []
    try:
        while (item := processor.tts_input_queue.get(timeout=3)).text != "<EOS>":
            speech.append(item.text)
    finally:
        processor.shutdown_event.set()
        worker.join(2)
    assert post.call_args.kwargs["json"]["tool_choice"] == "required"
    if calls_tool:
        assert speech == []
        assert processor.tool_calls_queue.get_nowait()["function"]["name"] == SEARCH_TOOL
    else:
        assert "couldn't start" in "".join(speech)
        assert processor.tool_calls_queue.empty()
