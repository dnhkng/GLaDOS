"""run_shell: bash on the owner's PC, in the foreground or as a tracked background task."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any
import uuid

from ...autonomy.task_manager import TaskResult
from .base import _AgentTool, _notify_desktop


class RunShell(_AgentTool):
    name = "run_shell"
    untrusted = True

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Give a shell command")
        rule = self.guard.blacklisted(command)
        if refusal := self.guard.check(rule and f"command matches {rule}", f"run_shell:{command}", command):
            return refusal
        cwd = self.path(args["cwd"]) if args.get("cwd") else None
        background = bool(args.get("background"))
        timeout = args.get("timeout", 600 if background else 30)
        if isinstance(timeout, bool) or not isinstance(timeout, int | float):
            raise TypeError("timeout must be a number of seconds")
        if background:
            return self.start_background(command, float(timeout), cwd)
        return {"command": command, **self.execute(command, float(timeout), cwd)}

    def start_background(self, command: str, timeout: float, cwd: Path | None) -> dict[str, Any]:
        """Queue a long command as a tracked task; its report reaches the user when it finishes."""
        tasks = self.tool_config.get("task_manager")
        if tasks is None:
            raise ValueError("Background tasks are unavailable in this profile")
        task_id = "task_shell_" + uuid.uuid4().hex[:10]
        cancelled = threading.Event()

        def job() -> TaskResult:
            result = self.execute(command, timeout, cwd, cancelled=cancelled, background=True)
            outcome = "cancelled" if result.get("error") == "Cancelled" else "done" if result["ok"] else "error"
            detail = f"exit {result['exit_code']}" if "exit_code" in result else result.get("error", "")
            if outcome != "cancelled":
                _notify_desktop(f"GLaDOS: command {outcome}", command[:120])
            return TaskResult(
                outcome,
                f"Shell command {outcome} ({detail}): {command[:120]}",
                notify_user=outcome != "cancelled",
                report=json.dumps({"command": command, **result}),
                importance=0.7,
                update_priority="important" if outcome != "cancelled" else "regular",
            )

        tasks.submit(task_id, "Shell: " + command[:100], job, group="shell", cancelled=cancelled)
        return {
            "ok": True,
            "status": "queued",
            "task_id": task_id,
            "instruction": "Tell the user in one short sentence that the command runs in the background and that you "
            "will report when it finishes. Use cancel_task with this task_id to stop it; get_report shows its output.",
        }
