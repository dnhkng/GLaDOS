"""Automatic defaults, saved-setting migration and emotion-safe private drafts."""

from collections.abc import Iterator
import json
from pathlib import Path
import threading
from unittest.mock import MagicMock, Mock

import pytest
import requests

from glados.core.context import ContextBuilder
from glados.core.decision_lists import DecisionListStore
from glados.core.inference import InferenceConfig, InferenceScheduler
from glados.core.routing import RoutingConfig
from glados.core.speculative import SpeculativeStream
from glados.tools.get_time import tool_definition
from tests.test_routing import wait_for
from tests.test_speech_markup import make_processor


def test_old_switches_migrate_without_losing_choices(tmp_path: Path) -> None:
    path = tmp_path / "choices.json"
    original = DecisionListStore(path, lambda: [tool_definition]).snapshot()
    original.update(enabled=False, speculative=False)
    original["lists"][0]["name"] = "My custom actions"
    path.write_text(json.dumps(original))
    migrated = DecisionListStore(path, lambda: [tool_definition]).snapshot()
    assert migrated["enabled"]
    assert "speculative" not in migrated
    assert migrated["lists"] == original["lists"]
    assert migrated["revision"] == original["revision"] + 1
    assert json.loads(path.read_text()) == migrated


def test_activation_only_selects_list_and_deletion_keeps_routing(tmp_path: Path) -> None:
    store = DecisionListStore(tmp_path / "choices.json", lambda: [tool_definition])
    row = store.get().model_dump()
    row.update(id="custom", name="Custom")
    store.mutate({"action": "save", "revision": 1, "list": row})
    selected = store.mutate({"action": "activate", "revision": 2, "id": "custom"})
    assert selected["enabled"] and selected["active_list"] == "custom"
    deleted = store.mutate({"action": "delete", "revision": 3, "id": "custom"})
    assert deleted["enabled"] and deleted["active_list"] == "speech"
    row = store.get().model_dump()
    row["enabled"] = False
    store.mutate({"action": "save", "revision": 4, "list": row})
    assert store.get(active=True).id != "speech"
    assert not store.get("speech").enabled


@pytest.mark.parametrize("compatible", [True, False])
def test_routing_automatically_detects_backend(monkeypatch: pytest.MonkeyPatch, compatible: bool) -> None:
    response = Mock()
    response.json.return_value = {"default_generation_settings": {}, "total_slots": 2} if compatible else {}
    get = Mock(return_value=response)
    monkeypatch.setattr("glados.core.routing.requests.get", get)
    assert RoutingConfig().enabled_for("http://localhost:18080/v1/chat/completions", {}) is compatible
    assert get.call_args.args[0] == "http://localhost:18080/props"


def test_unavailable_backend_does_not_require_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glados.core.routing.requests.get", Mock(side_effect=requests.ConnectionError()))
    assert not RoutingConfig().enabled_for("http://localhost/v1/chat/completions", {})


@pytest.mark.parametrize("search_tools", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_automatic_draft_survives_background_mood_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: bool, search_tools: bool,
) -> None:
    processor = make_processor()
    processor._inference_scheduler = InferenceScheduler()
    store = DecisionListStore(tmp_path / "choices.json", lambda: [tool_definition])
    if search_tools:
        processor._reply_tools = lambda available=None: [{"type": "function", "function": {"name": "mcp.internet_search.web_search_exa"}}]
    affect = ["Neutral; [emotion:neutral]"]
    builder = ContextBuilder()
    builder.register("emotion", lambda: affect[0], volatile=True)
    recalled = ["Old topic: tea"]
    builder.register("slots", lambda: recalled[0], volatile=True)
    processor.context_builder = builder
    processor._before_context = lambda message: recalled.__setitem__(0, "Current topic: " + message["content"])
    generated = threading.Event()
    payloads = []

    def post(*args: object, **kwargs: object) -> MagicMock:
        payloads.append(kwargs["json"])
        assert "Current topic: Hello" in json.dumps(kwargs["json"])
        assert "Old topic: tea" not in json.dumps(kwargs["json"])
        if search_tools:
            assert "call the search tool now" in json.dumps(kwargs["json"])
            assert "numResults=2" in json.dumps(kwargs["json"])
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        text = "Fresh reply." if len(payloads) > 1 else "Private draft."

        def lines(chunk_size: int = 1) -> Iterator[bytes]:
            generated.set()
            yield b'data: ' + json.dumps({"choices": [{"delta": {"content": text}}]}).encode()
            yield b'data: [DONE]'

        response.iter_lines.side_effect = lines
        return response

    def react(message: dict) -> None:
        assert processor.tts_input_queue.empty()
        assert processor.tool_calls_queue.empty()
        if changed:
            affect[0] = "Angry; [emotion:angry glare]"

    def score(*args: object, **kwargs: object) -> dict:
        with processor._inference_scheduler.lease("Routing", "router", "test"):
            kwargs["on_admitted"]()
            assert generated.wait(2)
            assert processor.tts_input_queue.empty()
            assert processor.tool_calls_queue.empty()
            return {"action": "reply"}

    monkeypatch.setattr("glados.core.speculative.requests.post", post)
    processor._before_reply = react
    processor.router = Mock(store=store, score=score)
    processor.llm_input_queue.put({"role": "user", "content": "Hello", "_allow_tools": search_tools})
    worker = threading.Thread(target=processor.run)
    worker.start()
    try:
        spoken = processor.tts_input_queue.get(timeout=3)
        assert spoken.text.strip() == "Private draft."
        assert len(payloads) == 1
        assert "Neutral; [emotion:neutral]" in json.dumps(payloads[-1])
        assert processor.tool_calls_queue.empty()
    finally:
        processor.shutdown_event.set()
        worker.join(2)
    wait_for(lambda: not processor._inference_scheduler.snapshot()["active"])
    assert not worker.is_alive()


def test_single_slot_uses_normal_reply_without_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    processor._inference_scheduler = InferenceScheduler(InferenceConfig(slots=1, reserved_interactive=0))
    store = Mock()

    def score(*args: object, **kwargs: object) -> dict:
        assert kwargs["on_admitted"] is None
        return {"action": "reply"}

    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.return_value = []
    post = Mock(return_value=response)
    post.side_effect = lambda *args, **kwargs: (processor.shutdown_event.set(), response)[1]
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.router = Mock(store=store, score=score)
    processor.llm_input_queue.put({"role": "user", "content": "Hello", "_allow_tools": False})
    processor.run()
    post.assert_called_once()


def test_unsupported_backend_bypasses_one_token_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    store = Mock()
    store.get.return_value = None
    store.snapshot.return_value = {"enabled": False}
    processor.router = Mock(store=store)
    processor._set_quiet_mode = Mock()
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.return_value = []
    post = Mock(side_effect=lambda *args, **kwargs: (processor.shutdown_event.set(), response)[1])
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.llm_input_queue.put({"role": "user", "content": "Hello", "_allow_tools": False})
    processor.run()
    processor.router.quiet_score.assert_not_called()
    processor.router.score.assert_not_called()
    post.assert_called_once()


def test_draft_generation_cancellation_survives_processing_reset() -> None:
    processor = make_processor()
    generation = [1]
    stream = SpeculativeStream(InferenceScheduler(), "http://test", {}, {"model": "test"},
                               processor.shutdown_event, processor.processing_active_event,
                               cancelled_if=lambda: generation[0] != 1)
    assert not stream.stopped()
    generation[0] = 2
    processor.processing_active_event.clear()
    processor.processing_active_event.set()
    assert stream.stopped()


def test_direct_audio_does_not_launch_speculative_inference(monkeypatch):
    processor = make_processor()
    processor._inference_scheduler = InferenceScheduler()
    draft = Mock()
    monkeypatch.setattr("glados.core.llm_processor.SpeculativeStream", draft)
    def score(*args, **kwargs):
        assert kwargs["on_admitted"] is None
        processor.shutdown_event.set()
        return {"action": "ignore"}
    processor.router = Mock(store=Mock(), score=score)
    processor.llm_input_queue.put({"role": "user", "content": "[Voice input]",
        "_native_audio": [{"type": "input_audio", "input_audio": {"data": "test", "format": "wav"}}]})
    processor.run()
    draft.assert_not_called()
