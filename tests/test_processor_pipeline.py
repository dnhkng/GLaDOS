"""Request parity and media isolation across the extracted processor phases."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from glados.core.context import ContextBuilder
from glados.core.conversation_store import ConversationStore
from glados.core.processor_turn import ProcessorTurn
from tests.test_speech_markup import make_processor

SEARCH_TOOL = {"type": "function", "function": {"name": "mcp.internet_search.web_search_exa"}}
CLOCK = {"datetime": "2026-10-07T12:00:00+02:00", "time": "12:00:00", "date": "2026-10-07",
         "weekday": "Wednesday", "timezone": "CEST", "utc_offset": "+0200"}


def configured_processor(monkeypatch):
    monkeypatch.setattr("glados.core.llm_processor.current_time", lambda: CLOCK)
    processor = make_processor()
    processor._conversation_store = ConversationStore([
        {"role": "system", "content": "Fixed personality"},
        {"role": "user", "content": "An earlier question"},
        {"role": "assistant", "content": "An earlier answer"},
    ])
    builder = ContextBuilder()
    stable = Mock(return_value="Session instructions")
    live = Mock(return_value="Fresh slot observation")
    builder.register("operator", stable)
    builder.register("slots", live, volatile=True)
    processor.context_builder = builder
    processor._reply_tools = lambda available=None: [SEARCH_TOOL]
    return processor, stable, live


@pytest.mark.parametrize("allow_tools", [False, True])
def test_draft_and_reply_build_identical_requests(monkeypatch, allow_tools):
    """Equivalent replies share policy/body/order, independent of history insertion."""
    requests = []
    for draft in (True, False):
        processor, stable, live = configured_processor(monkeypatch)
        message = {"role": "user", "content": "What is new today?"}
        turn = ProcessorTurn(llm_input={**message, "_allow_tools": allow_tools}, llm_message=message,
                             route={"action": "reply"})
        if not draft:
            processor._conversation_store.append(message)
        processor._build_request(turn, draft=draft)
        stable.assert_called_once()
        live.assert_called_once()
        assert turn.base_messages[-1] is message
        serialized = json.dumps(turn.base_messages)
        assert ("call the search tool now" in serialized) == allow_tools
        assert turn.base_messages[0]["content"] == "Fixed personality"
        assert serialized.index("An earlier answer") < serialized.index("Fresh slot observation")
        requests.append((turn.data, turn.base_messages, turn.tools))
    assert requests[0] == requests[1]


def test_native_audio_is_attached_only_to_prepared_request(monkeypatch):
    processor, _, _ = configured_processor(monkeypatch)
    message = {"role": "user", "content": "[Voice input]"}
    media = [{"type": "input_audio", "input_audio": {"data": "private-audio", "format": "wav"}}]
    processor._conversation_store.append(message)
    saved = deepcopy(processor._conversation_store.snapshot())
    turn = ProcessorTurn(llm_input={**message, "_native_audio": media}, llm_message=message,
                         audio_content=media, route={"action": "reply"})
    processor._build_request(turn)
    assert turn.base_messages[-1]["content"] == media
    assert turn.data["chat_template_kwargs"] == {"enable_thinking": False}
    assert processor._conversation_store.snapshot() == saved
    assert "private-audio" not in json.dumps(saved)
    assert message["content"] == "[Voice input]"
