"""The context inspector follows real assembly and submitted request payloads."""

import http.client
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from glados.autonomy.slots import TaskSlotStore
from glados.core.context import ContextBuilder
from glados.core.context_inspection import describe_context
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados
from glados.webapp.serializers import build_context
from glados.webapp.server import WebappServer
from tests.test_speech_markup import make_processor
from tests.test_webapp import _FakeEngine


def flatten(snapshot):
    return [{k: v for k, v in message.items() if k != "index"}
            for section in snapshot["sections"] for message in section["messages"]]


def test_preview_uses_live_sources_in_model_order_without_changing_history(monkeypatch):
    monkeypatch.setattr("glados.core.llm_processor.current_time", lambda: {"time": "12:34"})
    processor = make_processor()
    history = [{"role": "system", "content": "Base personality"},
               {"role": "assistant", "content": "[summary] Previous hour: user wants English."},
               {"role": "user", "content": "What changed?"}]
    processor._conversation_store = ConversationStore(history)
    state = {"emotion": "P=-0.6; speak with irritation", "vision": "A red jacket appeared"}
    builder = ContextBuilder()
    builder.register("operator", lambda: "Be brief", priority=20)
    builder.register("emotion", lambda: state["emotion"], priority=15, volatile=True)
    builder.register("slots", lambda: "Vision Core: red jacket", priority=8, volatile=True)
    builder.register("missing", lambda: None)
    builder.register("broken", lambda: 1 / 0)
    processor.context_builder = builder
    processor.vision_state = SimpleNamespace(as_message=lambda: {"role": "system", "content": state["vision"]})
    processor.mcp_manager = SimpleNamespace(get_context_messages=lambda **kw: [{"role": "system", "content": "System info"}],
                                            get_tool_definitions=lambda: [])
    processor.autonomy_system_prompt = "Autonomy instructions"
    preview = processor.context_preview()
    assert flatten(preview) == processor._build_messages(False)
    sources = [s["source"] for s in preview["sections"]]
    assert sources == ["system", "speech", "console", "operator", "summary", "emotion", "slots", "mcp", "clock", "vision", "history"]
    assert [m["index"] for s in preview["sections"] for m in s["messages"]] == list(range(1, preview["message_count"] + 1))
    assert processor.last_context() is None
    assert processor._conversation_store.snapshot() == history
    state["vision"] = "A blue book appeared"
    assert "blue book" in json.dumps(processor.context_preview())
    assert "blue book" not in json.dumps(preview)
    autonomous = processor.context_preview(True)
    assert flatten(autonomous) == processor._build_messages(True)
    assert "autonomy" in [s["source"] for s in autonomous["sections"]]
    assert "clock" in [s["source"] for s in autonomous["sections"]]
    assert not {"speech", "console"}.intersection(s["source"] for s in autonomous["sections"])


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/api/chat"])
def test_real_submission_captures_extra_instructions_and_omits_binary_media(monkeypatch, endpoint):
    processor = make_processor()
    processor.completion_url = "http://localhost" + endpoint
    processor._ollama_mode = processor._is_ollama_endpoint()
    processor._conversation_store = ConversationStore([{"role": "system", "content": "Base prompt"}])
    audio = [{"type": "input_audio", "input_audio": {"data": "SECRET-AUDIO-BYTES", "format": "wav"}},
             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,SECRET-IMAGE-BYTES"}}]
    processor.llm_input_queue.put({"role": "user", "content": "Audio placeholder", "_native_audio": audio,
                                   "_allow_tools": False, "_tool_reply_context": {"tool": "cpu_load", "result": "0.1"}})
    payloads = []

    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        processor.shutdown_event.set()
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.iter_lines.return_value = iter(())
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()
    snapshot = processor.last_context()
    assert len(payloads) == 1
    assert snapshot["kind"] == "request" and snapshot["mode"] == "user"
    assert snapshot["message_count"] == len(payloads[0]["messages"])
    assert snapshot["sections"][0]["source"] == "system"
    request = next(s for s in snapshot["sections"] if s["source"] == "request")
    assert "cpu_load" in request["messages"][0]["content"]
    assert snapshot["sections"][-1]["source"] == "input"
    encoded = json.dumps(snapshot)
    assert "SECRET" not in encoded and "audio payload omitted" in encoded and "media payload omitted" in encoded
    assert "SECRET-AUDIO-BYTES" in json.dumps(payloads[0]), "Inspection must not change inference input"
    processor._conversation_store.append({"role": "user", "content": "New question"})
    assert "New question" not in json.dumps(processor.last_context())
    assert snapshot["tools"] == []


def test_context_endpoint_modes_latest_worker_and_unavailable_states():
    engine = _FakeEngine()
    processor = make_processor()
    engine.llm_processor = processor
    assert not build_context(engine, view="request")["available"]
    engine.autonomy_llm_processors = [SimpleNamespace(last_context=lambda: {
        "available": True, "mode": "autonomy", "captured_at": 1}), SimpleNamespace(last_context=lambda: {
        "available": True, "mode": "autonomy", "captured_at": 2})]
    assert build_context(engine, "autonomy", "request")["captured_at"] == 2
    server = WebappServer(engine, port=0)
    server.start()

    def get(path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
        try:
            conn.request("GET", path, headers=headers or {})
            response = conn.getresponse()
            return response.status, json.loads(response.read()), response.getheader("Cache-Control")
        finally:
            conn.close()

    try:
        status, preview, cache = get("/api/context")
        assert status == 200 and preview["available"] and cache == "no-store"
        assert get("/api/context?mode=wrong")[0] == 400
        assert get("/api/context?view=wrong")[0] == 400
        assert get("/api/context", {"Origin": "https://other.example"})[0] == 403
        assert get("/api/context", {"Sec-Fetch-Site": "cross-site"})[0] == 403
        assert not get("/api/context?view=request")[1]["available"]
        assert get("/api/context?mode=autonomy&view=request")[1]["captured_at"] == 2
    finally:
        engine.shutdown_event.set()
        server.shutdown()


@pytest.mark.parametrize("pending_tool", [False, True])
def test_live_updates_preserve_completed_history_prefix_and_tool_exchanges(
    monkeypatch: pytest.MonkeyPatch, pending_tool: bool,
) -> None:
    processor = make_processor()
    completed = [{"role": "system", "content": "Base personality"},
                 {"role": "user", "content": "Earlier question"},
                 {"role": "assistant", "content": "Earlier answer"}]
    current = [{"role": "user", "content": "Current question"}]
    if pending_tool:
        current += [{"role": "assistant", "tool_calls": [{"id": "clock", "function": {"name": "get_time"}}]},
                    {"role": "tool", "tool_call_id": "clock", "content": "12:34"}]
    processor._conversation_store = ConversationStore(completed + current)
    state = {"clock": "12:34", "emotion": "Neutral", "vision": "Red cup"}
    monkeypatch.setattr("glados.core.llm_processor.current_time", lambda: {"time": state["clock"]})
    builder = ContextBuilder()
    builder.register("operator", lambda: "Speak briefly")
    builder.register("emotion", lambda: state["emotion"], priority=100, volatile=True)
    processor.context_builder = builder
    processor.vision_state = SimpleNamespace(as_message=lambda: {"role": "system", "content": state["vision"]})
    before = processor._build_messages(False)
    state.update(clock="12:35", emotion="Irritated", vision="Blue cup")
    after = processor._build_messages(False)
    first_live = next(i for i, message in enumerate(after) if message["content"] == "Irritated")
    assert before[:first_live] == after[:first_live]
    assert after[first_live - 1] == completed[-1], "History belongs before rapidly changing sensor data"
    assert after[-len(current):] == current, "Tool calls and their results must remain adjacent"
    processor._add_request_context(after, "New routing result")
    assert after[:first_live] == before[:first_live]
    assert after[-len(current):] == current
    assert after[-len(current) - 1]["content"] == "Blue cup"
    assert processor._conversation_store.snapshot() == completed + current


def test_completed_turn_remains_before_live_context(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    history = [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Answer"}]
    processor._conversation_store = ConversationStore(history)
    monkeypatch.setattr("glados.core.llm_processor.current_time", lambda: {"time": "12:34"})
    messages = processor._build_messages(False)
    clock_index = next(i for i, message in enumerate(messages) if "Live system clock" in str(message["content"]))
    assert messages[clock_index-2:clock_index] == history


def test_context_numbers_use_payload_order_not_internal_history_indices() -> None:
    messages = [{"role": "system", "content": "Personality", "index": 50},
                {"role": "assistant", "content": "Previous reply", "index": 0},
                {"role": "system", "content": "PAD and tone"},
                {"role": "user", "content": "New input", "index": 2}]
    snapshot = describe_context(messages, ["system", "history", "emotion", "input"], [],
                                model="test", mode="user", kind="request")
    assert [m["index"] for section in snapshot["sections"] for m in section["messages"]] == [1, 2, 3, 4]
    assert [section["source"] for section in snapshot["sections"]] == ["system", "history", "emotion", "input"]
    assert "Live PAD" in snapshot["sections"][2]["description"]
    assert messages[1]["index"] == 0, "Inspection must not modify the actual history"


def test_task_context_omits_state_already_supplied_by_dedicated_sources() -> None:
    store = TaskSlotStore()
    store.update_slot("emotion", "Emotion Core", "idle", "Neutral PAD", report="Tone guidance")
    store.update_slot("vision", "Vision Core", "active", "Current room")
    store.update_slot("compaction", "Memory Core", "monitoring", "No compaction needed")
    assert Glados._format_slots(SimpleNamespace(autonomy_slots=store)) is None
    store.update_slot("task_note", "Notes", "open", "Summarise the meeting")
    store.update_slot("weather", "Weather", "done", "Sunny")
    prompt = Glados._format_slots(SimpleNamespace(autonomy_slots=store))
    assert "[task_note]" in prompt and "Summarise the meeting" in prompt and "Sunny" in prompt
    assert all(text not in prompt for text in ("Neutral PAD", "Current room", "No compaction needed"))
    assert store.get_slot("emotion").report == "Tone guidance", "Full reports stay available in Cores"
