"""Constrained execution and audio-to-tool result handoff."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import threading
from unittest.mock import MagicMock, Mock

import pytest

from glados.core.conversation_store import ConversationStore
from glados.core.decision_lists import DecisionListStore
from glados.core.inference import InferenceScheduler
from glados.core.llm_processor import LanguageModelProcessor
from glados.core.native_audio import NativeAudioConfig, NativeAudioInput
from glados.core.tool_executor import ToolExecutor, _ToolResultQueue
from glados.tools import tool_definitions
from glados.tools.safe_command import SafeCommandRunner


def test_new_settings_include_fixed_commands_and_preserve_saved_edits(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    store = DecisionListStore(path, lambda: tool_definitions)
    data = store.snapshot()
    decision = data["lists"][0]
    commands = [o for o in decision["options"] if o["tool"] == "run_safe_command"]
    assert {o["arguments"]["task"] for o in commands} == {
        "uptime", "disk_usage", "memory_usage", "system_info", "cpu_load",
    }
    decision["options"] = [o for o in decision["options"] if o["id"] != "command_uptime"]
    store.mutate({"action": "save", "revision": data["revision"], "list": decision})
    reopened = DecisionListStore(path, lambda: tool_definitions)
    assert all(o.id != "command_uptime" for o in reopened.get().options)


@pytest.mark.parametrize("args", [
    {}, {"task": "time; touch /tmp/bad"}, {"task": "bash"}, {"task": []},
    {"task": "time", "arguments": ["--set", "yesterday"]},
])
def test_rejects_custom_commands_before_execution(monkeypatch: pytest.MonkeyPatch, args: dict) -> None:
    run = Mock()
    monkeypatch.setattr("glados.tools.safe_command.subprocess.run", run)
    with pytest.raises(ValueError):
        SafeCommandRunner().run(args)
    run.assert_not_called()


def test_fixed_command_has_clean_environment_and_bounded_output(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "x" * 9000, ""))
    monkeypatch.setattr("glados.tools.safe_command.subprocess.run", run)
    runner = SafeCommandRunner()
    result = runner.run({"task": "uptime"})
    args, kwargs = run.call_args
    assert args == (("/usr/bin/uptime", "-p"),)
    assert kwargs["shell"] is False and kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["env"] == {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    assert kwargs["timeout"] == 3 and kwargs["cwd"] == "/"
    assert len(result["stdout"]) == 4096 and result["ok"]
    snapshot = runner.snapshot()
    snapshot["recent"][0]["stdout"] = "changed"
    assert runner.snapshot()["recent"][0]["stdout"] == result["stdout"]


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("date", 3), FileNotFoundError(2, "Not found")])
def test_failures_release_slot_and_are_recorded(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr("glados.tools.safe_command.subprocess.run", Mock(side_effect=error))
    runner = SafeCommandRunner()
    result = runner.run({"task": "uptime"})
    assert not result["ok"] and result["error"]
    assert runner.snapshot()["active"] is None
    assert runner.snapshot()["recent"][0] == result


def test_command_slot_excludes_simultaneous_console_and_assistant_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        started.set()
        assert release.wait(2)
        return subprocess.CompletedProcess([], 0, "up one day", "")

    monkeypatch.setattr("glados.tools.safe_command.subprocess.run", run)
    runner = SafeCommandRunner()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(runner.run, {"task": "uptime"}, "assistant")
        try:
            assert started.wait(2)
            assert runner.snapshot()["active"]["task"] == "uptime"
            assert "busy" in runner.run({"task": "uptime"}, "console")["error"]
        finally:
            release.set()
        assert future.result()["ok"]


def test_nonblocking_tool_results_keep_context_and_single_action_guard() -> None:
    target = queue.Queue()
    call = {"function": {"name": "mcp.clock", "arguments": "{}"}}
    wrapper = _ToolResultQueue(target, call, bound=True)
    ToolExecutor._enqueue(wrapper, {"role": "tool", "content": "12:34"})
    result = target.get_nowait()
    assert result["_tool_reply_context"] == call["function"]
    assert result["_allow_tools"] is False


@pytest.mark.parametrize("transcripts", [False, True])
def test_native_audio_routed_time_roundtrip_preserves_interpreted_request(
    monkeypatch: pytest.MonkeyPatch, transcripts: bool,
) -> None:
    active = threading.Event()
    active.set()
    native = NativeAudioInput(NativeAudioConfig(enabled=True, user_transcripts=transcripts))
    history = ConversationStore([])
    processor = LanguageModelProcessor(
        queue.Queue(), queue.Queue(), queue.Queue(), history, "http://localhost/v1/chat/completions", "test",
        None, active, threading.Event(), native_audio=native, inference_scheduler=InferenceScheduler(),
    )
    native.transcribe = Mock(return_value="What time is it in UTC?")
    route = {"action": "tool", "tool": "get_time", "arguments": {"timezone": "UTC"},
             "list_id": "speech", "revision": 1, "option_id": "time"}
    store = Mock()
    store.snapshot.return_value = {"speculative": False}
    store.authorize.return_value = True
    processor.router = Mock(store=store)
    processor.router.score.return_value = route
    processor.llm_input_queue.put({"role": "user", "content": "[User spoke via audio; transcript disabled.]",
                                   "_native_audio": [{"type": "input_audio", "data": "private-audio"}]})
    captured = []

    def post(*args: object, **kwargs: object) -> MagicMock:
        captured.append(kwargs["json"])
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            b"data: " + json.dumps({"choices": [{"delta": {"content": "It is twelve thirty-four."}}]}).encode(),
            b"data: [DONE]",
        ]
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    executor = ToolExecutor(processor.llm_input_queue, queue.Queue(), processor.tool_calls_queue,
                            processor.processing_active_event, processor.shutdown_event, decision_store=store)
    threads = [threading.Thread(target=processor.run), threading.Thread(target=executor.run)]
    for thread in threads:
        thread.start()
    try:
        while processor.tts_input_queue.get(timeout=3).text != "<EOS>":
            pass
    finally:
        processor.shutdown_event.set()
        for thread in threads:
            thread.join(2)
    assert len(captured) == 1 and "tools" not in captured[0]
    messages = captured[0]["messages"]
    assert any("performed action" in str(m["content"]) for m in messages if m["role"] == "system")
    users = [m["content"] for m in messages if m["role"] == "user"]
    assert users == ["What time is it in UTC?"] if transcripts else "get_time" in users[0]
    assert native.transcribe.call_count == int(transcripts)
    if transcripts:
        assert history.snapshot()[0]["content"] == "What time is it in UTC?"
    assert any("system clock" in m["content"] for m in messages if m["role"] == "tool")
    assert "private-audio" not in json.dumps(messages)
    assert "_tool_reply_context" not in json.dumps(processor._conversation_store.snapshot())
