"""Agent tools: shell, files and the Hyprland desktop, behind a blacklist that needs the owner's code.

Everything runs directly except actions matching ``AgentConfig.blacklist`` (commands) or
``protected_paths`` (files). Those run only when the owner's latest message, spoken or typed in
the last few minutes, contains ``confirm_code``. The model is never told the code, so it cannot
supply it itself, and messages relayed from other AIs through the API never count.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any
import unicodedata
import uuid

from pydantic import BaseModel, Field
import requests

from ..autonomy.task_manager import TaskResult
from .protocol import MessageSink

DEFAULT_BLACKLIST = [
    r"\b(sudo|doas|pkexec)\b",
    r"(^|[;&|]\s*)su\b",
    r"\brm\b[^;&|]*\s-[a-zA-Z]*[rRf]",
    r"\b(shutdown|reboot|poweroff|halt)\b",
    r"\bsystemctl\b.*\b(poweroff|reboot|halt|suspend|hibernate|stop|disable|mask|kill)\b",
    r"\b(mkfs\S*|wipefs|fdisk|sfdisk|parted|sgdisk|shred)\b",
    r"\bdd\b",
    r">\s*/dev/(sd|nvme|mmcblk|disk)",
    r"\b(pacman|yay|paru)\b.*\s-[a-zA-Z]*[SRU]",
    r"\b(chmod|chown)\b.*\s-[a-zA-Z]*R",
    r"\bgit\b.*\bpush\b.*(--force|\s-f\b)",
    r"\bgit\b.*\b(reset\s+--hard|clean\s+-[a-zA-Z]*f)",
    r"\b(curl|wget)\b.*\|\s*(sudo\s+)?(ba|z|fi)?sh\b",
    r"\b(killall|pkill)\b",
    r"\bkill\b.*\s-(9|KILL)\b",
    r"\bcrontab\s+-r\b",
    r":\(\)\s*\{",
]
DEFAULT_PROTECTED = ["~/.ssh", "~/.gnupg", "~/.password-store", "~/.local/share/keyrings", "/etc", "/boot", "/usr"]
APPROVAL_WINDOW_S = 180.0
BACKGROUND_MAX_S = 3600.0
WORKER_TOOL_TIMEOUT_S = 300.0  # The worker is already in the background, so its commands may run longer.
RELAYED_PREFIX = "[Message from "


class AgentConfig(BaseModel):
    enabled: bool = False
    confirm_code: str = Field(default="tsuki", min_length=3)
    blacklist: list[str] = Field(default_factory=lambda: list(DEFAULT_BLACKLIST))
    protected_paths: list[str] = Field(default_factory=lambda: list(DEFAULT_PROTECTED))
    working_dir: str = "~"
    max_steps: int = Field(default=12, ge=1, le=50, description="Tool calls allowed per user request.")
    output_chars: int = Field(default=6000, ge=500, le=50000)
    ocr_languages: str = "fra+eng"
    vision_model: str | None = Field(default=None, description="Ollama vision model for describe_screen")
    vision_url: str = "http://localhost:11434/api/chat"
    worker_model: str = Field(default="qwen3:8b", description="Ollama model for delegate_task's background worker")
    worker_url: str = "http://localhost:11434/api/chat"
    worker_steps: int = Field(default=25, ge=1, le=100)
    worker_think: bool = Field(default=False, description="Let a reasoning worker model think first (slower)")


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


class Guard:
    """Decides whether a blacklisted action may run, from the owner's latest message."""

    def __init__(self, config: AgentConfig, tool_config: dict[str, Any]) -> None:
        self.config = config
        self._store = tool_config.get("conversation_store")
        self._interaction = tool_config.get("interaction_state")
        self._patterns = [re.compile(p, re.IGNORECASE) for p in config.blacklist]
        self._protected = [Path(p).expanduser().resolve() for p in config.protected_paths]

    def blacklisted(self, command: str) -> str | None:
        return next((p.pattern for p in self._patterns if p.search(command)), None)

    def protected(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self._protected)

    def approved(self) -> bool:
        if self._store is None:
            return False
        if self._interaction is not None:
            since = self._interaction.seconds_since_user()
            if since is None or since > APPROVAL_WINDOW_S:
                return False
        latest = next(
            (
                m.get("content")
                for m in reversed(self._store.snapshot())
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ),
            None,
        )
        if not latest or latest.startswith(RELAYED_PREFIX):
            return False
        return re.search(rf"\b{re.escape(_fold(self.config.confirm_code))}\b", _fold(latest)) is not None

    @staticmethod
    def refusal(reason: str) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "needs_owner_code",
            "reason": reason,
            "instruction": "This action is on the owner's blacklist. Do not run it another way. Tell the user what "
            "you want to do and ask them to say their confirmation code; then call the same tool again. You do not "
            "know the code: never guess, invent or repeat one.",
        }


class _AgentTool:
    def __init__(self, llm_queue: MessageSink, tool_config: dict[str, Any] | None = None) -> None:
        self.llm_queue = llm_queue
        self.tool_config = tool_config or {}
        config = self.tool_config.get("agent")
        self.config = config if isinstance(config, AgentConfig) else AgentConfig()
        self.guard = Guard(self.config, self.tool_config)
        self.timeout = float(self.tool_config.get("_tool_timeout", 30.0))

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if not self.config.enabled:
            result: dict[str, Any] = {"ok": False, "error": "Agent tools are disabled (agent.enabled)"}
        else:
            try:
                result = self.call(call_args if isinstance(call_args, dict) else {})
            except (ValueError, TypeError, OSError) as exc:
                result = {"ok": False, "error": str(exc)}
        self.llm_queue.put({"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)})

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def trim(self, text: str) -> str:
        limit = self.config.output_chars
        return text if len(text) <= limit else text[: limit // 2] + "\n…[truncated]…\n" + text[-limit // 2 :]

    def path(self, raw: object) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Give a file path")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(self.config.working_dir).expanduser() / path
        return path.resolve()

    def execute(
        self,
        argv: list[str] | str,
        timeout: float,
        cwd: Path | None = None,
        cancelled: threading.Event | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        """Run without a terminal; on timeout or cancellation the whole process group is killed."""
        # A foreground call must finish before the tool executor gives up on it.
        limit = max(1.0, min(timeout, BACKGROUND_MAX_S if background else self.timeout - 2))
        started = time.monotonic()
        process = subprocess.Popen(
            argv,
            shell=isinstance(argv, str),
            executable="/bin/bash" if isinstance(argv, str) else None,
            cwd=cwd or Path(self.config.working_dir).expanduser(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        deadline = started + limit
        while True:
            try:
                stdout, stderr = process.communicate(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                stopped = cancelled is not None and cancelled.is_set()
                if not stopped and time.monotonic() < deadline:
                    continue
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                error = "Cancelled" if stopped else f"Timed out after {limit:g} s"
                return {"ok": False, "error": error, "stdout": self.trim(stdout), "stderr": self.trim(stderr)}
        return {
            "ok": process.returncode == 0,
            "exit_code": process.returncode,
            "stdout": self.trim(stdout),
            "stderr": self.trim(stderr),
            "elapsed_s": round(time.monotonic() - started, 2),
        }


class RunShell(_AgentTool):
    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Give a shell command")
        if (rule := self.guard.blacklisted(command)) and not self.guard.approved():
            return self.guard.refusal(f"command matches {rule}")
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


class ReadFile(_AgentTool):
    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.path(args.get("path"))
        if self.guard.protected(path) and not self.guard.approved():
            return self.guard.refusal(f"{path} is a protected path")
        if path.is_dir():
            raise ValueError(f"{path} is a directory; use list_dir")
        lines = path.read_text(errors="replace").splitlines()
        offset = max(0, int(args.get("offset", 1)) - 1)
        count = max(1, min(int(args.get("limit", 200)), 2000))
        chunk = lines[offset : offset + count]
        numbered = "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(chunk))
        return {"path": str(path), "total_lines": len(lines), "content": self.trim(numbered)}


class WriteFile(_AgentTool):
    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        path, content = self.path(args.get("path")), args.get("content")
        if not isinstance(content, str):
            raise TypeError("content must be text")
        if self.guard.protected(path) and not self.guard.approved():
            return self.guard.refusal(f"{path} is a protected path")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if args.get("append") else "w") as handle:
            handle.write(content)
        return {"ok": True, "path": str(path), "bytes": len(content.encode()), "appended": bool(args.get("append"))}


class EditFile(_AgentTool):
    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        path, old, new = self.path(args.get("path")), args.get("old_text"), args.get("new_text")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("Give old_text (non-empty) and new_text")
        if self.guard.protected(path) and not self.guard.approved():
            return self.guard.refusal(f"{path} is a protected path")
        text = path.read_text()
        count = text.count(old)
        if count != 1:
            raise ValueError(f"old_text must appear exactly once; found {count} times")
        path.write_text(text.replace(old, new, 1))
        return {"ok": True, "path": str(path)}


class ListDir(_AgentTool):
    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.path(args.get("path") or ".")
        if self.guard.protected(path) and not self.guard.approved():
            return self.guard.refusal(f"{path} is a protected path")
        pattern = args.get("pattern") or "*"
        entries = sorted(path.rglob(pattern) if args.get("recursive") else path.glob(pattern))[:300]
        listing = [f"{p.relative_to(path)}{'/' if p.is_dir() else ''}" for p in entries if not p.name.startswith(".")]
        folders = sum(1 for entry in listing if entry.endswith("/"))
        # Small models miscount long lists: give the totals.
        return {
            "path": str(path),
            "folders": folders,
            "files": len(listing) - folders,
            "entries": listing,
            "truncated": len(entries) == 300,
        }


DESKTOP_ACTIONS = {
    "open_app": "value=command",
    "open_url": "value=URL or file",
    "workspace": "value=number",
    "move_to_workspace": "active window, value=number",
    "focus_window": "value=window class",
    "close_window": "active window",
    "fullscreen": "toggle, active window",
    "windows": "list open windows",
    "active_window": "focused window",
    "volume": "value=40%, +5%, -5% or mute",
    "brightness": "value=50%, +10% or 10%-",
    "media": "value=play-pause, next, previous or stop",
    "notify": "value=text",
    "screenshot": "save to ~/Pictures, returns path",
    "clipboard_get": "read clipboard",
    "clipboard_set": "value=text",
    "lock": "lock the session",
    "read_screen": "OCR the text on screen; value=window for the focused window only",
    "describe_screen": "describe the screen with the vision model; value=question",
    "type_text": "type into the focused window, value=text",
    "key": "press keys, value=e.g. Return, ctrl+c, super+1",
    "move_cursor": "value=x y in screen pixels",
    "click": "value=left, right or middle (needs ydotool)",
}


class Desktop(_AgentTool):
    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        action, value = args.get("action"), str(args.get("value") or "").strip()
        if action not in DESKTOP_ACTIONS:
            raise ValueError("Unknown desktop action")
        needs_value = {"open_app", "open_url", "workspace", "move_to_workspace", "focus_window", "volume"}
        needs_value |= {"brightness", "media", "notify", "clipboard_set", "type_text", "key", "move_cursor"}
        if action in needs_value and not value:
            raise ValueError(f"{action} needs a value")
        # Typing into a terminal runs a command just as surely as run_shell does.
        guarded = action in {"open_app", "type_text"}
        if guarded and (rule := self.guard.blacklisted(value)) and not self.guard.approved():
            return self.guard.refusal(f"command matches {rule}")
        hypr = ["hyprctl", "dispatch"]
        match action:
            case "open_app":
                return self.execute([*hypr, "exec", value], 5)
            case "open_url":
                return self.execute([*hypr, "exec", "xdg-open " + _quote(value)], 5)
            case "workspace" | "move_to_workspace":
                return self.execute([*hypr, "workspace" if action == "workspace" else "movetoworkspace", value], 5)
            case "focus_window":
                return self.execute([*hypr, "focuswindow", value if ":" in value else f"class:(?i).*{value}.*"], 5)
            case "close_window":
                return self.execute([*hypr, "killactive"], 5)
            case "fullscreen":
                return self.execute([*hypr, "fullscreen", "0"], 5)
            case "windows" | "active_window":
                result = self.execute(["hyprctl", "clients" if action == "windows" else "activewindow", "-j"], 5)
                if result["ok"]:
                    data = json.loads(result["stdout"] or "null")
                    windows = data if isinstance(data, list) else [data] if data else []
                    keys = ("class", "title", "workspace")
                    result["stdout"] = json.dumps([{k: w.get(k) for k in keys} for w in windows], ensure_ascii=False)
                return result
            case "volume":
                sink = "@DEFAULT_AUDIO_SINK@"
                if value == "mute":
                    return self.execute(["wpctl", "set-mute", sink, "toggle"], 5)
                amount = value.lstrip("+-")
                step = amount + value[0] if value[0] in "+-" else amount
                return self.execute(["wpctl", "set-volume", "-l", "1.0", sink, step], 5)
            case "brightness":
                return self.execute(["brightnessctl", "set", value], 5)
            case "media":
                return self.execute(["playerctl", value], 5)
            case "notify":
                return self.execute(["notify-send", "GLaDOS", value], 5)
            case "screenshot":
                target = Path("~/Pictures").expanduser() / time.strftime("glados-%Y%m%d-%H%M%S.png")
                target.parent.mkdir(parents=True, exist_ok=True)
                return {**self.execute(["grim", str(target)], 10), "path": str(target)}
            case "clipboard_get":
                return self.execute(["wl-paste", "--no-newline"], 5)
            case "clipboard_set":
                process = subprocess.run(["wl-copy"], input=value, text=True, timeout=5, check=False)
                return {"ok": process.returncode == 0}
            case "lock":
                return self.execute(["loginctl", "lock-session"], 5)
            case "read_screen":
                return self.read_screen(window=value == "window")
            case "describe_screen":
                return self.describe_screen(value or "Describe what is on the screen.")
            case "type_text":
                return self.execute(["wtype", "--", value], 15)
            case "key":
                *mods, key = [part.strip() for part in value.split("+")]
                mods = [{"super": "logo", "win": "logo", "control": "ctrl"}.get(m.lower(), m.lower()) for m in mods]
                argv = [
                    "wtype",
                    *[a for m in mods for a in ("-M", m)],
                    "-k",
                    key,
                    *[a for m in mods for a in ("-m", m)],
                ]
                return self.execute(argv, 5)
            case "move_cursor":
                x, _, y = value.replace(",", " ").partition(" ")
                return self.execute([*hypr, "movecursor", x.strip(), y.strip()], 5)
            case _:
                button = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}.get(value or "left")
                if button is None:
                    raise ValueError("click value must be left, right or middle")
                if shutil.which("ydotool") is None:
                    return {"ok": False, "error": "Clicking needs ydotool (sudo pacman -S ydotool, then run ydotoold)"}
                return self.execute(["ydotool", "click", button], 5)

    def screenshot(self, window: bool) -> bytes:
        argv = ["grim"]
        if window:
            active = json.loads(self.execute(["hyprctl", "activewindow", "-j"], 5)["stdout"] or "{}")
            if active.get("at") and active.get("size"):
                (x, y), (w, h) = active["at"], active["size"]
                argv += ["-g", f"{x},{y} {w}x{h}"]
        else:
            monitors = json.loads(self.execute(["hyprctl", "monitors", "-j"], 5)["stdout"] or "[]")
            focused = next((m["name"] for m in monitors if m.get("focused")), None)
            if focused:
                argv += ["-o", focused]
        captured = subprocess.run([*argv, "-"], capture_output=True, timeout=10, check=False)
        if captured.returncode != 0 or not captured.stdout:
            raise OSError(captured.stderr.decode(errors="replace").strip() or "Screen capture failed")
        return captured.stdout

    def read_screen(self, window: bool) -> dict[str, Any]:
        image = self.screenshot(window)
        ocr = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", self.config.ocr_languages],
            input=image,
            capture_output=True,
            timeout=30,
            check=False,
        )
        text = "\n".join(line for line in ocr.stdout.decode(errors="replace").splitlines() if line.strip())
        return {"ok": ocr.returncode == 0, "text": self.trim(text), "note": "OCR text; layout and images are lost"}

    def describe_screen(self, question: str) -> dict[str, Any]:
        if not self.config.vision_model:
            return {"ok": False, "error": "No vision model configured (agent.vision_model); use read_screen instead"}
        image = base64.b64encode(self.screenshot(window=False)).decode()
        response = requests.post(
            self.config.vision_url,
            json={
                "model": self.config.vision_model,
                "stream": False,
                "messages": [{"role": "user", "content": question, "images": [image]}],
            },
            timeout=max(5.0, self.timeout - 5),
        )
        response.raise_for_status()
        return {"ok": True, "description": self.trim(response.json().get("message", {}).get("content", ""))}


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def _definition(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_PATH = {"type": "string", "description": "Absolute, ~/ or relative to home"}
agent_definitions: list[dict[str, Any]] = [
    _definition(
        "run_shell",
        "Run bash on the owner's Arch Linux PC (stdout, stderr, exit code). For anything the other tools lack. "
        "GUI apps: use desktop open_app. Chain calls for multi-step tasks and check each result.",
        {
            "command": {"type": "string"},
            "cwd": _PATH,
            "timeout": {"type": "number", "description": "Seconds (default 30; background: 600, max 3600)"},
            "background": {
                "type": "boolean",
                "description": "For long jobs (builds, updates, downloads): run as a tracked task and report later",
            },
        },
        ["command"],
    ),
    _definition(
        "read_file",
        "Read a text file with line numbers.",
        {
            "path": _PATH,
            "offset": {"type": "integer", "description": "First line, from 1"},
            "limit": {"type": "integer"},
        },
        ["path"],
    ),
    _definition(
        "write_file",
        "Create or overwrite a text file (or append to it).",
        {"path": _PATH, "content": {"type": "string"}, "append": {"type": "boolean"}},
        ["path", "content"],
    ),
    _definition(
        "edit_file",
        "Replace one exact, unique passage of a text file. Read the file first.",
        {"path": _PATH, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
        ["path", "old_text", "new_text"],
    ),
    _definition(
        "list_dir",
        "List a folder's entries, optionally matching a glob pattern, optionally recursive.",
        {"path": _PATH, "pattern": {"type": "string"}, "recursive": {"type": "boolean"}},
        [],
    ),
    _definition(
        "desktop",
        "Control the Hyprland desktop. " + "; ".join(f"{name} ({what})" for name, what in DESKTOP_ACTIONS.items()),
        {"action": {"type": "string", "enum": list(DESKTOP_ACTIONS)}, "value": {"type": "string"}},
        ["action"],
    ),
    _definition(
        "delegate_task",
        "Hand a complex or long multi-step job (coding, investigating, organising files, setting something up) to "
        "a stronger background worker that has the same tools. It reports when done; keep talking meanwhile.",
        {"task": {"type": "string", "description": "Complete, self-contained description of the goal"}},
        ["task"],
    ),
]
WORKER_PROMPT = (
    "You are the background worker of GLaDOS, an assistant running on its owner's Arch Linux PC with Hyprland. "
    "Complete the task with the tools: work step by step, check each result, fix errors and verify the outcome "
    "before stopping. Prefer one run_shell with standard tools (find, grep, wc, du, sort, git) over many file "
    "reads: a folder includes its subfolders unless told otherwise. Never ask the owner questions; if something is "
    "blocked (for example a blacklisted action "
    "that needs the owner's code), skip it and say so. When finished, reply without tool calls: a short report of "
    "what you did, the result, and anything left to do."
)


class DelegateTask(_AgentTool):
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
        tools = [d for d in agent_definitions if d["function"]["name"] != "delegate_task"]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": task},
        ]
        actions: list[str] = []
        for _ in range(self.config.worker_steps):
            if cancelled.is_set():
                return TaskResult("cancelled", "Agent task cancelled", notify_user=False)
            response = requests.post(
                self.config.worker_url,
                json={
                    "model": self.config.worker_model,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    "think": self.config.worker_think,
                },
                timeout=600,
            )
            response.raise_for_status()
            message = response.json().get("message", {})
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                return self.report("done", task, message.get("content", ""), actions)
            for call in calls:
                name = call.get("function", {}).get("name", "")
                arguments = call.get("function", {}).get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments or "{}")
                actions.append(f"{name} {json.dumps(arguments, ensure_ascii=False)[:160]}")
                messages.append({"role": "tool", "tool_name": name, "content": self.run_tool(name, arguments)})
        return self.report("error", task, f"Stopped after {self.config.worker_steps} steps without finishing.", actions)

    def run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tool = agent_classes.get(name)
        if tool is None or tool is DelegateTask:
            return json.dumps({"ok": False, "error": f"No tool named {name}"})
        sink = _Collect()
        tool(sink, {**self.tool_config, "_tool_timeout": WORKER_TOOL_TIMEOUT_S}).run("worker", arguments)
        return str(sink.items[0]["content"]) if sink.items else json.dumps({"ok": False, "error": "No result"})

    def report(self, status: str, task: str, text: str, actions: list[str]) -> TaskResult:
        return TaskResult(
            status,
            f"Agent task {status}: {task[:120]}",
            notify_user=True,
            report=json.dumps({"task": task, "report": text, "steps": actions}, ensure_ascii=False),
            importance=0.8,
            update_priority="important",
        )


class _Collect:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)

    put_nowait = put


agent_classes: dict[str, type[_AgentTool]] = {
    "delegate_task": DelegateTask,
    "run_shell": RunShell,
    "read_file": ReadFile,
    "write_file": WriteFile,
    "edit_file": EditFile,
    "list_dir": ListDir,
    "desktop": Desktop,
}
