"""The background worker (delegate_task) and its durable notes (agent_notes)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

import requests

from ...autonomy.task_manager import TaskResult
from .base import _AgentTool, _Collect, _notify_desktop, redact
from .definitions import agent_definitions

WORKER_TOOL_TIMEOUT_S = 300.0  # The worker is already in the background, so its commands may run longer.
WORKER_PROMPT = (
    "You are the background worker of GLaDOS, an assistant running on its owner's Arch Linux PC with Hyprland. "
    "Complete the task with the tools: work step by step, check each result, fix errors and verify the outcome "
    "before stopping. Prefer one run_shell with standard tools (find, grep, wc, du, sort, git) over many file "
    "reads: a folder includes its subfolders unless told otherwise. Never ask the owner questions; if something is "
    "blocked (for example a blacklisted action "
    "that needs the owner's code), skip it and say so. When finished, reply without tool calls: a short report of "
    "what you did, the result, and anything left to do. Save durable facts you discover (project locations, "
    "working commands, the owner's habits) with agent_notes so later tasks start faster."
)


class AgentNotes(_AgentTool):
    """Durable notes the agent keeps for itself: project paths, habits, commands that worked."""

    name = "agent_notes"

    def notes_file(self) -> Path:
        return Path(self.config.notes_path).expanduser()

    def read(self) -> str:
        try:
            return self.notes_file().read_text()
        except OSError:
            return ""

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("action", "read") == "read":
            notes = self.read()
            return {"ok": True, "notes": notes[-self.config.output_chars :] or "No notes yet."}
        text = " ".join(str(args.get("text") or "").split())
        if not text:
            raise ValueError("Give the note text")
        path = self.notes_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(f"- {time.strftime('%Y-%m-%d')} {redact(text)[:500]}\n")
        return {"ok": True, "saved": text[:500]}


class DelegateTask(_AgentTool):
    name = "delegate_task"

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Describe the task")
        tasks = self.tool_config.get("task_manager")
        if tasks is None:
            raise ValueError("Background tasks are unavailable in this profile")
        task_id = "task_agent_" + uuid.uuid4().hex[:10]
        cancelled = threading.Event()
        tasks.submit(
            task_id, "Agent: " + task[:100], lambda: self.work(task, cancelled), group="agent", cancelled=cancelled
        )
        return {
            "ok": True,
            "status": "queued",
            "task_id": task_id,
            "instruction": "Tell the user in one short sentence that you are handing this to your background worker "
            "and will report when it is done. cancel_task with this task_id stops it.",
        }

    def work(self, task: str, cancelled: threading.Event) -> TaskResult:
        """A tool loop on the worker model; tools run in-process with the same guard."""
        status, text, actions = self.loop(task, cancelled)
        if status == "cancelled":
            return TaskResult("cancelled", "Agent task cancelled", notify_user=False)
        return self.report(status, task, text, actions)

    def loop(
        self, task: str, cancelled: threading.Event, prompt: str = WORKER_PROMPT, steps: int | None = None
    ) -> tuple[str, str, list[str]]:
        """Run the worker until it answers without tool calls: (status, final text, actions taken)."""
        tools = [d for d in agent_definitions if d["function"]["name"] != "delegate_task"]
        notes = AgentNotes(_Collect(), self.tool_config).read()[-2000:]
        system = prompt + (f"\n\nYour notes from earlier tasks:\n{notes}" if notes else "")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        actions: list[str] = []
        limit = steps or self.config.worker_steps
        for _ in range(limit):
            if cancelled.is_set():
                return "cancelled", "", actions
            message = self.ask_worker(messages, tools)
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                return "done", str(message.get("content", "")), actions
            for call in calls:
                name = call.get("function", {}).get("name", "")
                arguments = call.get("function", {}).get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments or "{}")
                actions.append(f"{name} {json.dumps(arguments, ensure_ascii=False)[:160]}")
                result = self.run_tool(name, arguments)
                link = (
                    {"tool_name": name} if self.config.worker_api == "ollama" else {"tool_call_id": call.get("id", "")}
                )
                messages.append({"role": "tool", **link, "content": result})
        return "error", f"Stopped after {limit} steps without finishing.", actions

    def ask_worker(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """One worker turn, on Ollama's native API or any OpenAI-compatible chat completions endpoint."""
        config = self.config
        if config.worker_api == "ollama":
            body = {"model": config.worker_model, "messages": messages, "tools": tools, "stream": False}
            response = requests.post(config.worker_url, json={**body, "think": config.worker_think}, timeout=600)
            response.raise_for_status()
            return dict(response.json().get("message") or {})
        key = os.environ.get(config.worker_api_key_env, "")
        if not key:
            raise OSError(f"Set the {config.worker_api_key_env} environment variable for the worker API")
        # Paid API calls share the Nexus Core's daily budget.
        nexus = self.tool_config.get("nexus_agent")
        if nexus is not None and nexus.budget_left() <= 0:
            raise OSError("The daily Nexus budget is exhausted; raise it in Facility Settings to continue")
        response = requests.post(
            config.worker_url,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": config.worker_model, "messages": messages, "tools": tools},
            timeout=600,
        )
        response.raise_for_status()
        if nexus is not None:
            nexus.charge(config.worker_model, response.json().get("usage"))
        message = dict(response.json()["choices"][0]["message"])
        message.setdefault("content", "")
        return {k: v for k, v in message.items() if v is not None or k == "content"}

    def run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        from . import agent_classes  # The registry imports this module, so look it up at call time.

        tool = agent_classes.get(name)
        if tool is None or tool is DelegateTask:
            return json.dumps({"ok": False, "error": f"No tool named {name}"})
        sink = _Collect()
        config = {**self.tool_config, "_tool_timeout": WORKER_TOOL_TIMEOUT_S, "_agent_source": "worker"}
        tool(sink, config).run("worker", arguments)
        return str(sink.items[0]["content"]) if sink.items else json.dumps({"ok": False, "error": "No result"})

    def report(self, status: str, task: str, text: str, actions: list[str]) -> TaskResult:
        _notify_desktop(f"GLaDOS: agent task {status}", task[:120])
        return TaskResult(
            status,
            f"Agent task {status}: {task[:120]}",
            notify_user=True,
            report=json.dumps({"task": task, "report": text, "steps": actions}, ensure_ascii=False),
            importance=0.8,
            update_priority="important",
        )
