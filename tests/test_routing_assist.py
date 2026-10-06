"""Uncertain classification delegates understanding without authorizing writes."""

import json
import queue
import threading
from unittest.mock import MagicMock, Mock

import pytest
import requests

from glados.core.conversation_store import ConversationStore
from glados.core.inference import InferenceScheduler
from glados.core.llm_processor import INTERNET_SEARCH_TOOL, LanguageModelProcessor
from glados.core.tool_executor import _ToolResultQueue
from glados.vision.vision_state import VisionState
from tests.test_speech_markup import make_processor


@pytest.mark.parametrize("action", ["assist", "clarify", "unavailable"])
@pytest.mark.parametrize("search_enabled", [False, True])
def test_full_assistant_receives_original_audio_and_limited_tools(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    search_enabled: bool,
) -> None:
    processor = make_processor()
    processor._conversation_store = ConversationStore([])
    if search_enabled:
        processor.mcp_manager = Mock()
        processor.mcp_manager.get_context_messages.return_value = []
        processor.mcp_manager.get_tool_definitions.return_value = [
            {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object"}}}
            for name in (INTERNET_SEARCH_TOOL, "mcp.other.write")
        ]
    store = Mock()
    store.snapshot.return_value = {"speculative": False}
    processor.router = Mock(store=store)
    processor.router.score.return_value = {"action": action}
    if action == "unavailable":
        processor.router.score.side_effect = requests.Timeout()
    audio = [{"type": "input_audio", "input_audio": {"data": "original-audio", "format": "wav"}}]
    processor.llm_input_queue.put({"role": "user", "content": "[Voice input]", "_native_audio": audio})
    captured = []

    def post(*args: object, **kwargs: object) -> MagicMock:
        captured.append(kwargs["json"])
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            b"data: " + json.dumps({"choices": [{"delta": {"content": "Hello. How are you today?"}}]}).encode(),
            b"data: [DONE]",
        ]
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    thread = threading.Thread(target=processor.run)
    thread.start()
    speech = []
    try:
        while (item := processor.tts_input_queue.get(timeout=3)).text != "<EOS>":
            speech.append(item.text)
    finally:
        processor.shutdown_event.set()
        thread.join(2)
    assert "Hello" in " ".join(speech)
    assert captured[0]["messages"][-1]["content"] == audio
    assert "original-audio" not in json.dumps(processor._conversation_store.snapshot())
    names = {tool["function"]["name"] for tool in captured[0].get("tools", [])}
    assert names == (
        set()
        if action == "clarify"
        else {
            "get_time",
            "run_safe_command",
            "get_report",
            "get_preferences",
        } | ({INTERNET_SEARCH_TOOL} if search_enabled else set())
    )


def test_search_is_available_in_replies_and_without_routing() -> None:
    processor = make_processor()
    search = {"type": "function", "function": {"name": INTERNET_SEARCH_TOOL}}
    system = {"type": "function", "function": {"name": "mcp.system_info.cpu_load"}}
    write = {"type": "function", "function": {"name": "mcp.other.write"}}
    processor.mcp_manager = Mock()
    processor.mcp_manager.get_tool_definitions.return_value = [search, system, write]
    assert processor._reply_tools() == [search]
    assert LanguageModelProcessor._filter_tools_for_message(
        [search, system, write], "Search for recent astronomy news",
    ) == [search]


def test_generated_write_tool_cannot_escape_read_only_fallback() -> None:
    processor = make_processor()
    calls = [
        {"id": "read", "function": {"name": "run_safe_command", "arguments": '{"task":"cpu_load"}'}},
        {"id": "write", "function": {"name": "set_preference", "arguments": "{}"}},
    ]
    processor._process_tool_call(calls, False, {"run_safe_command"}, read_only=True)
    call = processor.tool_calls_queue.get_nowait()
    assert call["id"] == "read" and call["_read_only_tools"]
    assert processor.tool_calls_queue.empty()
    target = queue.Queue()
    _ToolResultQueue(target, call, bound=True).put({"role": "tool", "content": "load = 1"})
    assert target.get_nowait()["_allow_tools"] is False


@pytest.mark.parametrize("native_audio", [False, True])
def test_reply_routing_allows_a_fresh_visual_question_and_preserves_audio(
    monkeypatch: pytest.MonkeyPatch,
    native_audio: bool,
) -> None:
    processor = make_processor()
    processor._conversation_store = ConversationStore([])
    processor.vision_state = VisionState()
    processor.vision_state.update("A person in a jacket.")
    processor._inference_scheduler = InferenceScheduler()
    store = Mock()
    store.snapshot.return_value = {"enabled": True}
    processor.router = Mock(store=store)

    def score(*args: object, **kwargs: object) -> dict:
        if callback := kwargs.get("on_admitted"):
            callback()
        return {"action": "reply"}

    processor.router.score.side_effect = score
    captured = []

    def post(*args: object, **kwargs: object) -> MagicMock:
        captured.append(kwargs["json"])
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            b"data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "zipper",
                                        "type": "function",
                                        "function": {
                                            "name": "vision_look",
                                            "arguments": '{"question":"Does my jacket have a zipper?"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ).encode(),
            b"data: [DONE]",
        ]
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    audio = [{"type": "input_audio", "input_audio": {"data": "original-question-audio", "format": "wav"}}]
    question = "Does my jacket have a zipper?"
    processor.llm_input_queue.put({"role": "user", "content": "[Voice input]" if native_audio else question,
                                   **({"_native_audio": audio} if native_audio else {})})
    thread = threading.Thread(target=processor.run)
    thread.start()
    try:
        call = processor.tool_calls_queue.get(timeout=3)
    finally:
        processor.shutdown_event.set()
        thread.join(2)
    assert call["function"]["name"] == "vision_look" and call["_read_only_tools"]
    assert json.loads(call["function"]["arguments"])["question"] == "Does my jacket have a zipper?"
    assert {t["function"]["name"] for t in captured[0]["tools"]} == {"vision_look"}
    assert captured[0]["messages"][-1]["content"] == (audio if native_audio else question)
    assert "original-question-audio" not in json.dumps(processor._conversation_store.snapshot())


@pytest.mark.parametrize("selected_tool", ["mcp.lights.turn_on", INTERNET_SEARCH_TOOL])
def test_hierarchical_plan_offers_only_selected_mcp_and_carries_scope_permit(
    monkeypatch: pytest.MonkeyPatch,
    selected_tool: str,
) -> None:
    processor = make_processor()
    processor._conversation_store = ConversationStore([])
    store = Mock()
    store.snapshot.return_value = {"speculative": False}
    store.authorize_scope.return_value = True
    processor.router = Mock(store=store)
    processor.router.score.return_value = {
        "action": "plan",
        "strategy": "hierarchical",
        "list_id": "speech",
        "revision": 1,
        "settings_revision": 2,
        "category": "mcp",
        "server": selected_tool.split(".")[1],
        "tool_scope": [selected_tool],
    }
    definitions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {"target": {"type": "string"}}},
            },
        }
        for name in [selected_tool, "mcp.weather.forecast", "get_time"]
    ]
    processor._build_tools = lambda autonomy: definitions
    captured = []

    def post(*args: object, **kwargs: object) -> MagicMock:
        captured.append(kwargs["json"])
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            b"data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "light",
                                        "type": "function",
                                        "function": {"name": selected_tool, "arguments": '{"target":"kitchen"}'},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ).encode(),
            b"data: [DONE]",
        ]
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.llm_input_queue.put({"role": "user", "content": "Turn on the kitchen light"})
    thread = threading.Thread(target=processor.run)
    thread.start()
    try:
        call = processor.tool_calls_queue.get(timeout=3)
    finally:
        processor.shutdown_event.set()
        thread.join(2)
    assert {t["function"]["name"] for t in captured[0]["tools"]} == {selected_tool}
    assert captured[0].get("tool_choice") == ("required" if selected_tool == INTERNET_SEARCH_TOOL else None)
    assert json.loads(call["function"]["arguments"]) == {"target": "kitchen"}
    assert call["_routing_permit"]["tool_scope"] == [selected_tool]
    assert call["_routing_permit"]["settings_revision"] == 2
    assert "_read_only_tools" not in call


def test_tool_result_prompt_preserves_load_values_and_units(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    processor._conversation_store = ConversationStore([])
    processor.llm_input_queue.put(
        {
            "role": "tool",
            "tool_call_id": "cpu",
            "content": '{"load_1m":8.68,"load_5m":6.94,"load_15m":4.87}',
            "_allow_tools": False,
            "_tool_reply_context": {"name": "mcp.system_info.cpu_load", "arguments": "{}"},
        }
    )
    captured = []

    def post(*args: object, **kwargs: object) -> MagicMock:
        captured.append(kwargs["json"])
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Load averages are 8.68, 6.94 and 4.87."}}]}',
            b"data: [DONE]",
        ]
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    worker = threading.Thread(target=processor.run)
    worker.start()
    try:
        while processor.tts_input_queue.get(timeout=3).text != "<EOS>":
            pass
    finally:
        processor.shutdown_event.set()
        worker.join(2)
    messages = captured[0]["messages"]
    assert messages[-1]["content"] == '{"load_1m":8.68,"load_5m":6.94,"load_15m":4.87}'
    assert any("NOT CPU utilization percentages" in str(m["content"]) for m in messages if m["role"] == "system")
    assert "tools" not in captured[0]
    assert any("8.68, 6.94, 4.87" in str(m["content"]) for m in messages if m["role"] == "system")


def test_cpu_measurement_hint_distinguishes_periods_from_values() -> None:
    processor = make_processor()
    assert "4.00, 6.12, 5.00" in processor._measurement_hint(
        '{"task":"cpu_load","ok":true,"stdout":"4.00 6.12 5.00 7/2947 1551546\\n"}'
    )
    assert processor._measurement_hint('{"task":"cpu_load","ok":false,"stdout":"4 6 5"}') == ""
    assert processor._measurement_hint('{"load_1m":"unknown","load_5m":2,"load_15m":3}') == ""
