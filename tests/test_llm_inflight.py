"""Inference counters include active work and clear on every exit path."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from glados.core.llm_processor import LanguageModelProcessor
from glados.core.llm_tracking import InFlightCounter


@pytest.mark.parametrize("lane", ["priority", "autonomy"])
@pytest.mark.parametrize("failure", [None, "prepare", "request"])
def test_inflight_tracks_work_after_dequeue_and_clears_on_exit(
    monkeypatch: pytest.MonkeyPatch, lane: str, failure: str | None
) -> None:
    pending = queue.Queue()
    pending.put({"role": "user", "content": "test", "_allow_tools": False})
    active = threading.Event()
    active.set()
    shutdown = threading.Event()
    counter = InFlightCounter()
    processor = LanguageModelProcessor(
        llm_input_queue=pending,
        tool_calls_queue=queue.Queue(),
        tts_input_queue=queue.Queue(),
        conversation_store=SimpleNamespace(append=lambda message: None),
        completion_url="http://localhost/v1/chat/completions",
        model_name="test",
        api_key=None,
        processing_active_event=active,
        shutdown_event=shutdown,
        lane=lane,
        inflight_counter=counter,
    )
    seen = []

    def build_messages(autonomy_mode: bool) -> list[dict[str, object]]:
        seen.append((counter.value(), pending.qsize()))
        if failure == "prepare":
            shutdown.set()
            raise RuntimeError("preparation failed")
        return []

    def post(*args: object, **kwargs: object) -> MagicMock:
        seen.append((counter.value(), pending.qsize()))
        shutdown.set()
        if failure == "request":
            raise requests.exceptions.ConnectionError("offline")
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.iter_lines.return_value = iter(())
        return response

    monkeypatch.setattr(processor, "_build_messages", build_messages)
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()
    assert seen
    assert all(item == (1, 0) for item in seen)
    assert counter.value() == 0
