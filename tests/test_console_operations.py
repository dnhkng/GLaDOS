"""Console controls and agent tools operate on real shared engine state."""

import http.client
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from glados.autonomy.slots import TaskSlotStore
from glados.core.context import ContextBuilder
from glados.core.operator_state import CONSOLE_PROMPT, OperatorState
from glados.core.search_preferences import SearchPreferences, SearchSources
from glados.tools.manage_slot import ManageSlot, save_task
from glados.tools.safe_command import SafeCommandRunner
from glados.webapp.server import WebappServer
from tests.test_speech_markup import make_processor
from tests.test_webapp import _FakeEngine


def test_task_tool_saves_report_and_reuses_id() -> None:
    store = TaskSlotStore()
    messages = queue.Queue()
    tool = ManageSlot(messages, {"slot_store": store})
    tool.run("create", {"title": "Test plan", "summary": "Write three tests", "status": "open"})
    saved = json.loads(messages.get_nowait()["content"])
    tool.run("update", {"slot_id": saved["slot_id"], "status": "done", "report": "One. Two. Three."})
    assert json.loads(messages.get_nowait()["content"])["status"] == "done"
    assert len(store.list_slots()) == 1
    assert store.get_slot(saved["slot_id"]).report == "One. Two. Three."
    with pytest.raises(ValueError):
        save_task(store, {"slot_id": "weather", "title": "Overwrite agent"})
    assert len(store.list_slots()) == 1


def test_operator_changes_reach_next_reply_and_task_prompt_is_not_in_autonomy() -> None:
    operator = OperatorState()
    builder = ContextBuilder()
    builder.register("operator", operator.as_prompt)
    processor = make_processor()
    processor.context_builder = builder
    operator.set_instructions("Reply with one short sentence.")
    messages = processor._build_messages(False)
    assert any("one short sentence" in message["content"] for message in messages)
    assert any(message["content"] == CONSOLE_PROMPT for message in messages)
    assert all(message["content"] != CONSOLE_PROMPT for message in processor._build_messages(True))
    with pytest.raises(ValueError):
        operator.set_instructions("x" * 4001)
    assert "one short sentence" in operator.as_prompt()


def test_http_operations_change_state_and_reject_invalid_inputs() -> None:
    engine = _FakeEngine()
    engine.operator_state = OperatorState()
    engine.autonomy_slots = TaskSlotStore()
    engine.command_runner = SafeCommandRunner()
    engine.native_audio = SimpleNamespace(config=SimpleNamespace(user_transcripts=False))
    engine.asr_muted_event = threading.Event()
    engine.tts_muted_event = threading.Event()
    engine.set_asr_muted = lambda muted: engine.asr_muted_event.set() if muted else engine.asr_muted_event.clear()
    engine.set_tts_muted = lambda muted: engine.tts_muted_event.set() if muted else engine.tts_muted_event.clear()
    engine.autonomy_config = SimpleNamespace(enabled=False)
    engine.set_autonomy_enabled = lambda enabled: setattr(engine.autonomy_config, "enabled", enabled)
    received = []
    engine.submit_text_input = lambda text, source: received.append((text, source)) or True
    server = WebappServer(engine, port=0)
    server.start()

    def post(path: str, body: object, origin: str | None = None) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        try:
            connection.request("POST", path, json.dumps(body), headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        mind = SimpleNamespace(paused=False, requested=False)
        mind.set_paused = lambda paused: setattr(mind, "paused", paused)
        mind.request_tick = lambda: setattr(mind, "requested", True)
        engine.subagent_manager = SimpleNamespace(
            pause=lambda agent_id, paused: mind.set_paused(paused),
            trigger=lambda agent_id: mind.request_tick(),
            get=lambda name: mind if name == "emotion" else None,
        )
        assert post("/api/minds/control", {"agent_id": "emotion", "action": "pause"})[0] == 200
        assert mind.paused and not engine.shutdown_event.is_set()
        assert post("/api/minds/control", {"agent_id": "emotion", "action": "run"})[0] == 200
        assert mind.requested and mind.paused
        assert post("/api/minds/control", {"agent_id": "emotion", "action": "resume"})[0] == 200
        assert not mind.paused
        assert post("/api/minds/control", {"agent_id": "glados", "action": "pause"})[0] == 404
        assert post("/api/minds/control", {"agent_id": "emotion", "action": "delete"})[0] == 400
        assert post("/api/tools/time", {"timezone": "UTC"})[0] == 404
        result = post("/api/tools/command", {"task": "uptime"})
        assert result[0] == 200 and result[1]["ok"] and result[1]["stdout"]
        assert post("/api/tools/command", {"task": "time; echo unsafe"})[0] == 400
        assert post("/api/tools/command", {"task": "time"})[0] == 400
        assert post("/api/tools/command", {"task": "uptime", "args": []})[0] == 400
        assert post("/api/tools/command", {"task": "uptime"}, "https://example.com")[0] == 403
        engine.subagent_manager = None
        assert post("/api/search/settings", {})[0] == 404
        engine.search_agent = SimpleNamespace(preferences=SearchPreferences(SearchSources()))
        favorites = {"weather": ["https://dwd.de/"], "news": [], "reddit": ["reddit.com/r/Munich"], "general": []}
        status, saved = post("/api/search/settings", favorites)
        assert status == 200 and saved["weather"] == ["dwd.de"]
        assert post("/api/search/settings", {**favorites, "weather": ["not a site"]})[0] == 400
        assert post("/api/search/settings", favorites, "https://example.com")[0] == 403
        assert engine.search_agent.preferences.snapshot() == saved
        status, state = post("/api/control", {"action": "microphone", "enabled": False})
        assert status == 200 and state["controls"]["microphone_muted"]
        assert engine.asr_muted_event.is_set()
        assert post("/api/control", {"action": "transcripts", "enabled": True})[0] == 200
        assert engine.native_audio.config.user_transcripts is True
        status, state = post("/api/control", {"action": "autonomy", "enabled": True})
        assert status == 200 and state["controls"]["autonomy_enabled"]
        assert post("/api/control", {"action": "autonomy", "enabled": False})[0] == 200
        assert not engine.autonomy_config.enabled
        assert post("/api/control", {"action": "voice", "enabled": "false"})[0] == 400
        assert not engine.tts_muted_event.is_set()
        assert post("/api/control", {"action": "voice", "enabled": False}, "https://example.com")[0] == 403
        assert not engine.tts_muted_event.is_set()
        assert post("/api/instructions", {"instructions": "Be concise."})[0] == 200
        assert engine.operator_state.snapshot()["instructions"] == "Be concise."
        assert post("/api/input", {"text": "Hello"})[0] == 202
        assert received == [("Hello", "webapp")]
        assert post("/api/input", {"text": ""})[0] == 400
        assert post("/api/slots", ["invalid"])[0] == 400
        status, task = post("/api/slots", {"title": "Test", "summary": "Prepare a test plan"})
        assert status == 200 and task["status"] == "open"
        status, updated = post("/api/slots", {"slot_id": task["slot_id"], "status": "done"})
        assert status == 200 and updated["title"] == "Test"
        assert engine.autonomy_slots.get_slot(task["slot_id"]).status == "done"
        engine.shutdown_event.set()
        assert post("/api/input", {"text": "Late message"})[0] == 409
    finally:
        engine.shutdown_event.set()
        server.shutdown()
