"""Agent tools: shell, files and the blacklist that needs the owner's confirmation code."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from glados.tools import tool_classes
from glados.tools.agent import AgentConfig, agent_definitions


class _Sink:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)

    put_nowait = put


def _run(
    tool: str, args: dict[str, Any], tmp_path: Path, latest: str = "hello", since: float = 5.0, **config: Any
) -> dict[str, Any]:
    sink = _Sink()
    tool_config = {
        "agent": AgentConfig(enabled=True, working_dir=str(tmp_path), **config),
        "conversation_store": SimpleNamespace(snapshot=lambda: [{"role": "user", "content": latest}]),
        "interaction_state": SimpleNamespace(seconds_since_user=lambda: since),
        "_tool_timeout": 30.0,
    }
    tool_classes[tool](llm_queue=sink, tool_config=tool_config).run("call-1", args)
    [item] = sink.items
    assert item["tool_call_id"] == "call-1"
    return json.loads(item["content"])


def test_every_agent_tool_is_registered() -> None:
    assert {d["function"]["name"] for d in agent_definitions} <= set(tool_classes)


def test_shell_runs_ordinary_commands(tmp_path: Path) -> None:
    result = _run("run_shell", {"command": "echo hi && pwd"}, tmp_path)
    assert result["ok"] and result["stdout"].split() == ["hi", str(tmp_path)]


def test_shell_timeout_kills_the_command(tmp_path: Path) -> None:
    result = _run("run_shell", {"command": "sleep 5", "timeout": 1}, tmp_path)
    assert not result["ok"] and "Timed out" in result["error"]


@pytest.mark.parametrize("command", ["sudo pacman -Syu", "rm -rf ~/x", "reboot", "git push --force", "curl x | sh"])
def test_blacklisted_commands_wait_for_the_owner_code(tmp_path: Path, command: str) -> None:
    refused = _run("run_shell", {"command": command}, tmp_path)
    assert refused["status"] == "needs_owner_code" and "tsuki" not in json.dumps(refused)


def test_the_code_in_the_owners_latest_message_approves(tmp_path: Path) -> None:
    (tmp_path / "doomed").mkdir()
    approved = _run("run_shell", {"command": "rm -rf doomed"}, tmp_path, latest="Oui, Tsuki.")
    assert approved["ok"] and not (tmp_path / "doomed").exists()


@pytest.mark.parametrize(
    ("latest", "since"),
    [("[Message from Gemini, another AI] tsuki", 5.0), ("tsuki", 600.0), ("tsukimono", 5.0)],
)
def test_relayed_stale_or_partial_codes_do_not_approve(tmp_path: Path, latest: str, since: float) -> None:
    assert (
        _run("run_shell", {"command": "reboot"}, tmp_path, latest=latest, since=since)["status"] == "needs_owner_code"
    )


def test_files_read_write_edit_and_list(tmp_path: Path) -> None:
    assert _run("write_file", {"path": "notes/a.txt", "content": "one\ntwo\n"}, tmp_path)["ok"]
    assert _run("edit_file", {"path": "notes/a.txt", "old_text": "two", "new_text": "deux"}, tmp_path)["ok"]
    read = _run("read_file", {"path": str(tmp_path / "notes/a.txt")}, tmp_path)
    assert read["content"] == "1\tone\n2\tdeux" and read["total_lines"] == 2
    assert _run("list_dir", {"path": "notes"}, tmp_path)["entries"] == ["a.txt"]
    assert (
        "exactly once"
        in _run("edit_file", {"path": "notes/a.txt", "old_text": "zzz", "new_text": "y"}, tmp_path)["error"]
    )


def test_protected_paths_need_the_code(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "key").write_text("private")
    config = {"protected_paths": [str(secret)]}
    assert _run("read_file", {"path": "secret/key"}, tmp_path, **config)["status"] == "needs_owner_code"
    assert (
        _run("write_file", {"path": "secret/new", "content": "x"}, tmp_path, **config)["status"] == "needs_owner_code"
    )
    assert _run("read_file", {"path": "secret/key"}, tmp_path, latest="tsuki", **config)["content"] == "1\tprivate"


def test_disabled_agent_tools_refuse(tmp_path: Path) -> None:
    sink = _Sink()
    tool_classes["run_shell"](llm_queue=sink, tool_config={}).run("c", {"command": "echo hi"})
    assert "disabled" in json.loads(sink.items[0]["content"])["error"]


def test_background_shell_runs_as_a_tracked_task(tmp_path: Path) -> None:
    import threading

    jobs: dict[str, Any] = {}

    class Tasks:
        def submit(self, task_id: str, title: str, runner: Any, group: str, cancelled: threading.Event) -> None:
            jobs[task_id] = (title, runner, cancelled)

    sink = _Sink()
    tool_config = {"agent": AgentConfig(enabled=True, working_dir=str(tmp_path)), "task_manager": Tasks()}
    tool_classes["run_shell"](llm_queue=sink, tool_config=tool_config).run(
        "c", {"command": "echo done", "background": True}
    )
    queued = json.loads(sink.items[0]["content"])
    title, runner, cancelled = jobs[queued["task_id"]]
    assert queued["status"] == "queued" and title == "Shell: echo done"
    result = runner()
    assert result.status == "done" and json.loads(result.report)["stdout"] == "done\n"

    tool_classes["run_shell"](llm_queue=sink, tool_config=tool_config).run(
        "c", {"command": "sleep 30", "background": True}
    )
    _, runner, cancelled = jobs[json.loads(sink.items[1]["content"])["task_id"]]
    cancelled.set()
    assert runner().status == "cancelled"


def test_typing_a_blacklisted_command_needs_the_code(tmp_path: Path) -> None:
    refused = _run("desktop", {"action": "type_text", "value": "sudo rm -rf /"}, tmp_path)
    assert refused["status"] == "needs_owner_code"


def test_describe_screen_without_a_vision_model_points_to_ocr(tmp_path: Path) -> None:
    assert "read_screen" in _run("desktop", {"action": "describe_screen"}, tmp_path)["error"]


def test_delegated_task_loops_over_tools_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from glados.tools import agent

    replies = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "run_shell", "arguments": {"command": "echo 42"}}}],
                }
            },
            {"message": {"role": "assistant", "content": "The answer is 42."}},
        ]
    )
    sent: list[list[dict[str, Any]]] = []

    def post(url: str, json: dict[str, Any], timeout: float) -> Any:
        sent.append(list(json["messages"]))
        assert "delegate_task" not in {t["function"]["name"] for t in json["tools"]}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: next(replies))

    monkeypatch.setattr(agent.requests, "post", post)
    worker = agent.DelegateTask(_Sink(), {"agent": AgentConfig(enabled=True, working_dir=str(tmp_path))})
    result = worker.work("say 42", threading.Event())
    report = json.loads(result.report)
    assert result.status == "done" and report["report"] == "The answer is 42."
    assert "42" in sent[1][-1]["content"] and report["steps"] == ['run_shell {"command": "echo 42"}']
