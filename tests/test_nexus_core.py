"""Nexus Core consults external advisors in the background and moves Central to an API when too slow."""

import json
from pathlib import Path
import queue
import threading
from unittest.mock import Mock

import pytest
import requests

from glados.autonomy.agents.nexus_agent import NexusAgent, NexusBaseConfig, NexusConfig
from glados.autonomy.loop import NEXUS_INSTRUCTION
from glados.autonomy.slots import TaskSlotStore
from glados.autonomy.task_manager import TaskManager
from glados.core.tool_executor import ToolExecutor
from tests.test_autonomy_core import make_loop, wait_until

LOCAL = {"url": "http://localhost:11434/api/chat", "model": "qwen3:4b", "headers": {}, "request_options": {}}


def reply(content: object, status: int = 200, usage: dict | None = None) -> Mock:
    text = content if isinstance(content, str) else json.dumps(content)
    return Mock(status_code=status, json=Mock(return_value={
        "choices": [{"message": {"content": text}}],
        "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 500},
    }), raise_for_status=Mock())


ADVICE = {"answer": "About 4.24 years.", "confidence": "high", "caveats": []}


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> NexusAgent:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    core = NexusAgent(NexusConfig(), LOCAL, slot_store=TaskSlotStore(), settings_path=tmp_path / "nexus.yaml")
    core.control("on")
    return core


def install(monkeypatch: pytest.MonkeyPatch, *responses: Mock) -> Mock:
    post = Mock(side_effect=list(responses))
    monkeypatch.setattr("glados.autonomy.agents.nexus_agent.requests.post", post)
    return post


def test_dispatcher_choice_is_used_and_the_answer_is_reported_with_its_cost(
        agent: NexusAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    post = install(monkeypatch, reply({"model": "gpt-5-mini", "reasoning_effort": "low", "max_output_tokens": 400,
                                       "reason": "simple astronomy"}), reply(ADVICE))
    report = json.loads(agent.consult({"need": "Distance check", "question": "Light time to Proxima?"}))
    assert report["status"] == "done" and report["model"] == "gpt-5-mini"
    assert report["answer"] == ADVICE["answer"] and report["reason"] == "simple astronomy"
    dispatcher, advisor = (call.kwargs["json"] for call in post.call_args_list)
    assert dispatcher["model"] == "gpt-5-nano" and dispatcher["response_format"] == {"type": "json_object"}
    assert advisor["model"] == "gpt-5-mini" and advisor["reasoning_effort"] == "low"
    assert advisor["max_completion_tokens"] == 400 + 2000
    assert "persona" in advisor["messages"][0]["content"]
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert agent.spent_today() == pytest.approx((1000 * 0.05 + 500 * 0.4 + 1000 * 0.25 + 500 * 2.0) / 1e6)
    assert agent._stats_summary()["gpt-5-mini"]["calls"] == 1


def test_unknown_model_falls_back_to_the_cheapest_and_failures_choose_another(
        agent: NexusAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    post = install(monkeypatch, reply({"model": "gpt-99"}), reply("", status=500),
                   reply({"model": "gpt-5", "reason": "the cheap model failed"}), reply(ADVICE))
    report = json.loads(agent.consult({"question": "Hard proof"}))
    assert report["status"] == "done" and report["model"] == "gpt-5"
    assert report["failed_attempts"] == [{"model": "gpt-5-nano", "error": "gpt-5-nano returned HTTP 500"}]
    retry = json.loads(post.call_args_list[2].kwargs["json"]["messages"][1]["content"])
    assert retry["excluded"] == ["gpt-5-nano"] and all(m["id"] != "gpt-5-nano" for m in retry["catalog"])


def test_missing_key_budget_and_inactive_consultation_make_no_paid_request(
        agent: NexusAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    post = install(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY")
    assert "OPENAI_API_KEY" in json.loads(agent.consult({"question": "Q"}))["error"]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agent.settings.daily_budget_usd = 0
    assert "budget" in json.loads(agent.consult({"question": "Q"}))["error"]
    agent.settings.daily_budget_usd = 1
    agent.control("off")
    assert json.loads(agent.consult({"question": "Q"}))["error"] == "Nexus consultation is off"
    assert agent.as_prompt() is None
    post.assert_not_called()


def test_controls_persist_and_prompt_central_only_when_active(agent: NexusAgent, tmp_path: Path) -> None:
    agent.control("prefer", "Use the strongest model for code")
    restored = NexusAgent(NexusConfig(), LOCAL, slot_store=TaskSlotStore(), settings_path=tmp_path / "nexus.yaml")
    assert restored.consult_active and restored.status()["preferences"] == ["Use the strongest model for code"]
    assert "nexus_consult" in restored.as_prompt()
    with pytest.raises(ValueError):
        restored.control("explode")


def test_slow_local_replies_move_central_to_the_api_and_failures_bring_it_back(agent: NexusAgent) -> None:
    switched = []
    agent._set_backend = switched.append
    for seconds in (20, 22):
        agent.observe_first_token(seconds)
    assert not switched
    agent.observe_first_token(21)
    backend = switched[-1]
    assert backend["model"] == "gpt-5-nano" and backend["remote"]
    assert backend["request_options"]["reasoning_effort"] == "minimal"
    assert agent.status()["base"] == "api"
    agent.observe_failure(True, "ConnectionError")
    assert switched[-1] is None and agent.status()["base"] == "local"
    for seconds in (30, 30, 30):  # A recently failed API is not retried at once.
        agent.observe_first_token(seconds)
    assert switched[-1] is None


def test_probe_switches_when_local_server_is_unreachable_or_slow(
        agent: NexusAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    switched = []
    agent._set_backend = switched.append
    monkeypatch.setattr("glados.autonomy.agents.nexus_agent.requests.post",
                        Mock(side_effect=requests.ConnectionError()))
    agent._update_base()
    assert switched[-1]["remote"] and "unreachable" in agent.status()["base_reason"]


def test_without_api_key_the_base_stays_local_and_is_not_probed(
        agent: NexusAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    post = install(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY")
    agent._update_base()
    agent.observe_first_token(60)
    agent.observe_first_token(60)
    agent.observe_first_token(60)
    assert agent.status()["base"] == "local"
    post.assert_not_called()


def test_forced_local_mode_never_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    core = NexusAgent(NexusConfig(base=NexusBaseConfig(mode="local")), LOCAL, slot_store=TaskSlotStore())
    for _ in range(5):
        core.observe_first_token(99)
    assert core.status()["base"] == "local"


def test_consultation_runs_in_background_and_goes_straight_to_central(
        agent: NexusAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def post(*args: object, **kwargs: object) -> Mock:
        if kwargs["json"]["model"] == "gpt-5-nano" and "catalog" in kwargs["json"]["messages"][1]["content"]:
            return reply({"model": "gpt-5-mini", "reason": "astronomy"})
        assert release.wait(2)
        return reply(ADVICE)

    monkeypatch.setattr("glados.autonomy.agents.nexus_agent.requests.post", post)
    loop = make_loop()
    loop._central_queue = queue.Queue()
    slots, bus = loop._slot_store, loop._event_bus
    agent._slot_store = slots
    tasks = TaskManager(slots, bus)
    calls, replies, active, shutdown = queue.Queue(), queue.Queue(), threading.Event(), threading.Event()
    active.set()
    executor = ToolExecutor(replies, queue.Queue(), calls, active, shutdown,
                            tool_config={"task_manager": tasks, "nexus_agent": agent})
    calls.put({"id": "n1", "function": {"name": "nexus_consult",
                                        "arguments": {"need": "Distance", "question": "Light time to Proxima?"}}})
    thread = threading.Thread(target=executor.run)
    thread.start()
    try:
        data = json.loads(replies.get(timeout=2)["content"])
        assert data["task_id"].startswith("task_nexus_") and "acknowledgement" in data["instruction"]
        wait_until(lambda: slots.get_slot(data["task_id"]).status == "running")
        release.set()
        wait_until(lambda: slots.get_slot(data["task_id"]).status == "done")
        loop._scan_slots()
        assert loop._deliver_direct()
        request = loop._central_queue.get_nowait()
        assert request["_autonomy_response"] and NEXUS_INSTRUCTION in request["content"]
        assert ADVICE["answer"] in request["content"]
        assert loop._llm_queue.empty(), "The reviewer is not consulted for an awaited answer"
        loop.finish_cycle(request["_autonomy_cycle"], "response")
        assert slots.get_slot(data["task_id"]).handled
    finally:
        release.set()
        shutdown.set()
        thread.join(2)
        tasks.shutdown(wait=True)


def test_inactive_nexus_rejects_consultation_without_a_task(agent: NexusAgent) -> None:
    agent.control("off")
    tasks = TaskManager(TaskSlotStore(), Mock())
    replies, calls, active, shutdown = queue.Queue(), queue.Queue(), threading.Event(), threading.Event()
    active.set()
    executor = ToolExecutor(replies, queue.Queue(), calls, active, shutdown,
                            tool_config={"task_manager": tasks, "nexus_agent": agent})
    calls.put({"id": "n1", "function": {"name": "nexus_consult", "arguments": {"question": "Q"}}})
    thread = threading.Thread(target=executor.run)
    thread.start()
    try:
        assert "off" in json.loads(replies.get(timeout=2)["content"])["error"]
        assert not [s for s in tasks._slot_store.list_slots() if s.slot_id.startswith("task_nexus_")]
    finally:
        shutdown.set()
        thread.join(2)
        tasks.shutdown(wait=True)


def test_central_switches_backend_between_turns_and_hides_nexus_tools_until_needed() -> None:
    from tests.test_speech_markup import make_processor

    processor = make_processor()
    names = lambda: {t["function"]["name"] for t in processor._build_tools(False)}  # noqa: E731
    assert not names() & {"nexus_control", "nexus_consult"}
    nexus = Mock(consult_active=False)
    processor._nexus = lambda: nexus
    assert "nexus_control" in names() and "nexus_consult" not in names()
    nexus.consult_active = True
    assert {"nexus_control", "nexus_consult"} <= names()

    processor._request_options = {"think": False}
    processor._local_backend["request_options"] = {"think": False}
    processor.set_backend({"url": "https://api.example/v1/chat/completions", "model": "gpt-5-nano",
                           "headers": {"Authorization": "Bearer sk"}, "request_options": {"x": 1}, "remote": True})
    assert processor.model_name == "test", "A switch never changes a request already in progress"
    processor._apply_backend()
    assert processor.model_name == "gpt-5-nano" and processor._remote_backend
    assert processor._request_options == {"x": 1} and processor.prompt_headers == {"Authorization": "Bearer sk"}
    assert processor._process_chunk({"choices": [], "usage": {"prompt_tokens": 3}}) is None
    nexus.observe_usage.assert_called_once_with("gpt-5-nano", {"prompt_tokens": 3})
    processor.set_backend(None)
    processor._apply_backend()
    assert processor.model_name == "test" and not processor._remote_backend
    assert processor._request_options == {"think": False}


def test_console_settings_switch_consultation_base_and_budget(agent: NexusAgent, tmp_path: Path) -> None:
    switched = []
    agent._set_backend = switched.append
    agent.update_settings({"consultation": False, "base_mode": "api", "daily_budget_usd": 2})
    assert not agent.consult_active and agent.status()["base"] == "api" and switched[-1]["remote"]
    state = agent.update_settings({"base_mode": "local"})
    assert state["base"] == "local" and switched[-1] is None and state["daily_budget_usd"] == 2
    for invalid in ({"base_mode": "cloud"}, {"daily_budget_usd": -1}, {"consultation": "yes"}, {"colour": 1}, {}):
        with pytest.raises(ValueError):
            agent.update_settings(invalid)
    assert agent.status()["base_mode"] == "local" and agent.settings.daily_budget_usd == 2
    restored = NexusAgent(NexusConfig(), LOCAL, slot_store=TaskSlotStore(), settings_path=tmp_path / "nexus.yaml")
    assert restored.settings.base.mode == "local" and restored.settings.daily_budget_usd == 2
    assert not restored.consult_active


def test_webapp_endpoint_updates_nexus_and_reports_it_in_state(agent: NexusAgent) -> None:
    import http.client

    from glados.webapp.serializers import build_state
    from glados.webapp.server import WebappServer
    from tests.test_webapp import _FakeEngine

    engine = _FakeEngine()
    server = WebappServer(engine, port=0)
    server.start()

    def post(body: object) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
        try:
            connection.request("POST", "/api/nexus/settings", json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        assert post({"consultation": True})[0] == 404
        engine.nexus_agent = agent
        status, saved = post({"consultation": False, "daily_budget_usd": 1.5})
        assert status == 200 and saved["consultation"] == "off" and saved["daily_budget_usd"] == 1.5
        assert post({"base_mode": "cloud"})[0] == 400
        assert build_state(engine)["nexus"]["consultation"] == "off"
    finally:
        server.shutdown()


def test_probe_skipped_during_api_failure_cooldown(agent: NexusAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    post = install(monkeypatch)
    agent._switch("api", "test")
    agent.observe_failure(remote=True, reason="500 Server Error")
    assert agent._base == "local"
    # While in cooldown, _update_base must NOT probe the local model
    agent._update_base()
    post.assert_not_called()


def test_tool_hierarchy_respects_lsp() -> None:
    from glados.tools.nexus import NexusConsult, NexusControl, _BaseNexusTool

    assert issubclass(NexusControl, _BaseNexusTool)
    assert issubclass(NexusConsult, _BaseNexusTool)
    assert not issubclass(NexusConsult, NexusControl)

