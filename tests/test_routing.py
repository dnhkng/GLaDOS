"""Admission, versioned settings, complete scoring, and side-effect gating."""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from unittest.mock import Mock

import pytest
import requests

from glados.autonomy.llm_client import LLMConfig, llm_call
from glados.core.decision_lists import DecisionListStore
from glados.core.inference import InferenceCancelledError, InferenceScheduler
from glados.core.routing import DecisionRouter, RoutingConfig
from glados.tools.get_time import tool_definition


def wait_for(predicate: Callable[[], bool]) -> None:
    end = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < end
        time.sleep(0.005)


def test_background_does_not_block_reserved_interactive_slot() -> None:
    scheduler = InferenceScheduler()
    first = scheduler.acquire("emotion", "autonomy", "test")
    cancelled = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        queued = pool.submit(scheduler.acquire, "observer", "autonomy", "test", cancelled.is_set)
        wait_for(lambda: len(scheduler.snapshot()["waiting"]) == 1)
        # Router jumps the background queue and executes concurrently.
        with scheduler.lease("router", "router", "test"):
            assert len(scheduler.snapshot()["active"]) == 2
            assert not queued.done()
        cancelled.set()
        with pytest.raises(InferenceCancelledError):
            queued.result(2)
    scheduler.release(first)
    assert scheduler.snapshot()["active"] == scheduler.snapshot()["waiting"] == []


def test_lease_released_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = InferenceScheduler()

    def fail(*args: object, **kwargs: object) -> None:
        assert scheduler.snapshot()["active"][0]["owner"] == "emotion"
        raise requests.Timeout()

    monkeypatch.setattr("glados.autonomy.llm_client.requests.post", fail)
    assert llm_call(LLMConfig("http://test", scheduler=scheduler, owner="emotion"), "system", "user") is None
    assert scheduler.snapshot()["active"] == []


@pytest.fixture
def store(tmp_path: Path) -> DecisionListStore:
    store = DecisionListStore(tmp_path / "decisions.json", lambda: [tool_definition], enabled=True)
    # Fixed-binding tests exercise an explicit foreign zone, not the context clock.
    store._data["lists"][0]["options"].insert(2, {
        "id": "time", "description": "Read the current time in UTC", "action": "tool", "tool": "get_time",
        "arguments": {"timezone": "UTC"}, "enabled": True, "category": None, "context_source": None,
    })
    return store


def test_settings_persist_conflicts_and_revoke_old_actions(store: DecisionListStore) -> None:
    decision = store.get(active=True)
    permit = {"list_id": decision.id, "revision": decision.revision, "option_id": "time"}
    assert store.authorize(permit)
    edit = decision.model_dump()
    edit["options"][2]["arguments"] = {"timezone": "Europe/Berlin"}
    saved = store.mutate({"action": "save", "revision": 1, "list": edit})
    assert saved["revision"] == 2
    assert not store.authorize(permit)
    assert decision.options[2].arguments == {"timezone": "UTC"}  # In-flight snapshot retains its meaning.
    with pytest.raises(ValueError, match="changed"):
        store.mutate({"action": "save", "revision": 1, "list": edit})
    reopened = DecisionListStore(store.path, store.tools)
    assert reopened.snapshot() == store.snapshot()
    store.mutate({"action": "delete", "revision": 2, "id": decision.id})
    assert store.get(active=True).id != decision.id
    assert store.snapshot()["enabled"]
    assert not store.authorize(permit)


@pytest.mark.parametrize("change", ["unknown_tool", "bad_argument", "duplicate", "one_enabled"])
def test_invalid_settings_are_atomic(store: DecisionListStore, change: str) -> None:
    before = store.snapshot()
    edit = before["lists"][0]
    if change == "unknown_tool":
        edit["options"][2]["tool"] = "missing_light"
    elif change == "bad_argument":
        edit["options"][2]["arguments"] = {"timezone": 42}
    elif change == "duplicate":
        edit["options"][2]["id"] = edit["options"][1]["id"]
    else:
        for option in edit["options"][1:]:
            option["enabled"] = False
    with pytest.raises(ValueError):
        store.mutate({"action": "save", "revision": 1, "list": edit})
    assert store.snapshot()["revision"] == 1
    assert not store.path.exists()


def make_router(
    store: DecisionListStore, monkeypatch: pytest.MonkeyPatch, values: list[float | None]
) -> tuple[DecisionRouter, Mock]:
    router = DecisionRouter(store, InferenceScheduler(), "http://test/v1/chat/completions", "test", {}, RoutingConfig())
    monkeypatch.setattr(router, "token_ids", lambda labels: {label: i for i, label in enumerate(labels)})
    post = Mock(
        return_value=Mock(
            json=lambda: {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"top_probs": [{"id": i, "prob": v} for i, v in enumerate(values) if v is not None]}
                            ]
                        }
                    }
                ]
            }
        )
    )
    monkeypatch.setattr("glados.core.routing.requests.post", post)
    return router, post


def test_scoring_uses_one_token_and_never_executes_preview(
    store: DecisionListStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    router, post = make_router(store, monkeypatch, [0.01, 0.01, 0.95, 0.02, 0.01])
    result = router.score(store.get().model_copy(update={"strategy": "flat"}), "What time is it?", dry_run=True)
    assert result["action"] == "tool" and result["tool"] == "get_time" and result["dry_run"]
    assert result["scores"][0]["probability"] == pytest.approx(0.95)
    sent = post.call_args.kwargs["json"]
    assert sent["max_tokens"] == 1
    assert len(set(sent["logit_bias"].values())) == 1  # Equal bias cannot change relative option odds.
    assert sent["samplers"] == ["temperature"]
    assert router.scheduler.snapshot()["active"] == []
    assert not store.path.exists()


@pytest.mark.parametrize("values", [[0.01, 0.01, 0.95, None, 0.01], [0.01, 0.01, 0.95, float("nan"), 0.01]])
def test_missing_or_invalid_scores_never_authorize(
    store: DecisionListStore, monkeypatch: pytest.MonkeyPatch, values: list[float | None]
) -> None:
    router, _ = make_router(store, monkeypatch, values)
    with pytest.raises(ValueError, match="Incomplete"):
        router.score(store.get().model_copy(update={"strategy": "flat"}), "Do it")
    assert router.scheduler.snapshot()["active"] == []


def test_uncertainty_and_typed_ignore_fall_back(store: DecisionListStore, monkeypatch: pytest.MonkeyPatch) -> None:
    router, _ = make_router(store, monkeypatch, [0.01, 0.45, 0.44, 0.09, 0.01])
    assert router.score(store.get().model_copy(update={"strategy": "flat"}), "Do it")["action"] == "assist"
    router, _ = make_router(store, monkeypatch, [0.99, 0.0025, 0.0025, 0.0025, 0.0025])
    assert router.score(store.get().model_copy(update={"strategy": "flat"}), "Someone said hello")["action"] == "assist"
    assert router.score(store.get().model_copy(update={"strategy": "flat"}), "Someone said hello", spoken=True)["action"] == "ignore"


@pytest.mark.parametrize("action", ["ignore", "tool"])
def test_routing_gates_generation_and_tool_dispatch(
    store: DecisionListStore, action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_speech_markup import make_processor

    processor = make_processor()
    processor.processing_active_event.set()
    processor._inference_scheduler = InferenceScheduler()
    result = {
        "action": action,
        "tool": "get_time",
        "arguments": {"timezone": "UTC"},
        "list_id": "speech",
        "revision": 1,
        "option_id": "time",
    }
    scored = threading.Event()

    def score(*args: object, **kwargs: object) -> dict:
        scored.set()
        return result

    processor.router = Mock(store=store, score=score)
    post = Mock(side_effect=AssertionError("Routing must not generate a speculative audible reply"))
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.llm_input_queue.put({"role": "user", "content": "What time is it?"})
    thread = threading.Thread(target=processor.run)
    thread.start()
    try:
        assert scored.wait(2)
        if action == "tool":
            call = processor.tool_calls_queue.get(timeout=2)
            assert call["function"]["name"] == "get_time"
            assert store.authorize(call["_decision_permit"])
            assert processor.tts_input_queue.empty()
        elif action == "clarify":
            assert "clarify" in str(processor.tts_input_queue.get(timeout=2)).lower()
            assert processor.tool_calls_queue.empty()
        else:
            assert processor.tool_calls_queue.empty()
            assert processor.tts_input_queue.empty()
    finally:
        processor.shutdown_event.set()
        thread.join(2)
    post.assert_not_called()
    assert not thread.is_alive()
    assert processor._inference_scheduler.snapshot()["active"] == []


@pytest.mark.parametrize("action", ["reply", "ignore", "tool"])
def test_parallel_draft_is_private_until_routing_approves(
    store: DecisionListStore, action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_speech_markup import make_processor

    processor = make_processor()
    processor.processing_active_event.set()
    scheduler = InferenceScheduler()
    processor._inference_scheduler = scheduler
    generated, release_decision = threading.Event(), threading.Event()
    result = {
        "action": action,
        "tool": "get_time",
        "arguments": {"timezone": "UTC"},
        "list_id": "speech",
        "revision": 1,
        "option_id": "time",
    }

    def score(*args: object, **kwargs: object) -> dict:
        with scheduler.lease("Routing", "router", "test"):
            kwargs["on_admitted"]()
            assert generated.wait(2)
            assert release_decision.wait(2)
        return result

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)

    def chunks(chunk_size: int = 1) -> Iterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"A private draft."}}]}'
        generated.set()
        yield b"data: [DONE]"

    response.iter_lines = chunks
    post = Mock(return_value=response)
    monkeypatch.setattr("glados.core.speculative.requests.post", post)
    processor.router = Mock(store=store, score=score)
    processor.llm_input_queue.put({"role": "user", "content": "Hello"})
    thread = threading.Thread(target=processor.run)
    thread.start()
    try:
        assert generated.wait(2)
        assert processor.tts_input_queue.empty()
        assert processor.tool_calls_queue.empty()
        processor._conversation_store.append.assert_not_called()
        release_decision.set()
        if action == "reply":
            assert "private draft" in str(processor.tts_input_queue.get(timeout=2))
            assert processor.tool_calls_queue.empty()
        elif action == "tool":
            assert processor.tool_calls_queue.get(timeout=2)["function"]["name"] == "get_time"
            assert processor.tts_input_queue.empty()
        else:
            wait_for(lambda: not scheduler.snapshot()["active"])
            assert processor.tts_input_queue.empty()
        assert processor.processing_active_event.is_set()
    finally:
        release_decision.set()
        processor.shutdown_event.set()
        thread.join(2)
    assert post.call_count == 1
    assert "tools" not in post.call_args.kwargs["json"]
    wait_for(lambda: not scheduler.snapshot()["active"])


def test_decision_http_crud_and_preview_cannot_dispatch(store: DecisionListStore) -> None:
    import http.client
    import json

    from glados.webapp.server import WebappServer
    from tests.test_webapp import _FakeEngine

    engine = _FakeEngine()
    engine.decision_lists = store
    engine.router = Mock()
    engine.router.score.return_value = {"action": "tool", "dry_run": True}
    engine.tool_calls_queue = Mock()
    server = WebappServer(engine, port=0)
    server.start()

    def request(path: str, body: dict | None = None, origin: str | None = None) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        try:
            connection.request("GET" if body is None else "POST", path, json.dumps(body) if body else None, headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        code, data = request("/api/decisions")
        assert code == 200
        row = data["lists"][0]
        row.update(id="preview-test", name="A test list")
        code, saved = request("/api/decisions", {"action": "save", "revision": 1, "list": row})
        assert code == 200 and len(saved["lists"]) == 2
        code, result = request("/api/decisions/test", {"list_id": row["id"], "text": "What time is it?"})
        assert code == 200 and result["dry_run"]
        engine.tool_calls_queue.put.assert_not_called()
        assert engine.router.score.call_args.kwargs["dry_run"] is True
        assert request("/api/decisions/test", {"list_id": row["id"], "text": ""})[0] == 400
        assert (
            request("/api/decisions", {"action": "delete", "revision": 2, "id": row["id"]}, "https://evil.test")[0]
            == 403
        )
        assert len(store.snapshot()["lists"]) == 2
        assert request("/api/decisions", {"action": "delete", "revision": 2, "id": row["id"]})[0] == 200
        assert len(store.snapshot()["lists"]) == 1
    finally:
        engine.shutdown_event.set()
        server.shutdown()


def test_backend_change_resets_activation_without_losing_lists(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    first = DecisionListStore(path, lambda: [tool_definition], enabled=True, backend_key="llama")
    first.mutate({"action": "activate", "revision": 1, "id": "speech"})
    changed = DecisionListStore(path, first.tools, enabled=False, backend_key="another-provider")
    assert not changed.snapshot()["enabled"]
    assert "speculative" not in changed.snapshot()
    assert changed.snapshot()["lists"] == first.snapshot()["lists"]
