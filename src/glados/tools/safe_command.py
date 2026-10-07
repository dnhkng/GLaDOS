"""One CPU execution slot for a fixed catalogue of read-only host commands."""

from collections import deque
from copy import deepcopy
import json
import queue
import subprocess
import threading
import time
from typing import Any

from ..observability import ObservabilityBus

COMMANDS = {
    "uptime": ("System uptime", ("/usr/bin/uptime", "-p")),
    "disk_usage": ("Root filesystem space", ("/usr/bin/df", "-h", "/")),
    "memory_usage": ("System memory usage", ("/usr/bin/free", "-h")),
    "system_info": ("Operating system information", ("/usr/bin/uname", "-srmo")),
    "cpu_load": ("CPU load averages", ("/usr/bin/cat", "/proc/loadavg")),
}


class SafeCommandRunner:
    """Shared by the tool executor and console; never accepts shell text or arguments."""

    def __init__(self, observability_bus: ObservabilityBus | None = None, timeout: float = 3.0) -> None:
        self._bus = observability_bus
        self._timeout = timeout
        self._lock = threading.Lock()
        self._active: dict[str, Any] | None = None
        self._recent: deque = deque(maxlen=20)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy({
                "capacity": 1, "active": self._active, "recent": list(self._recent),
                "catalog": [{"task": task, "label": label, "command": list(argv)}
                            for task, (label, argv) in COMMANDS.items()],
            })

    def run(self, args: dict[str, Any], source: str = "assistant") -> dict[str, Any]:
        if not isinstance(args, dict) or set(args) != {"task"}:
            raise ValueError("Supply only a task from the fixed command catalogue")
        task = args["task"]
        if not isinstance(task, str) or task not in COMMANDS:
            raise ValueError("Unknown command task; arbitrary commands are not supported")
        label, argv = COMMANDS[task]
        started = time.monotonic()
        record = {"task": task, "label": label, "command": list(argv),
                  "source": source, "started_at": time.time()}
        with self._lock:
            if self._active is not None:
                return {"ok": False, "task": task, "error": "Command slot is busy. Try again shortly."}
            self._active = record
        result: dict[str, Any] = {**record, "ok": False}
        try:
            if self._bus:
                self._bus.emit("command", "start", label, meta={"task": task, "source": source})
            completed = subprocess.run(
                argv, shell=False, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, errors="replace", timeout=self._timeout, cwd="/",
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}, check=False,
            )
            result.update(ok=completed.returncode == 0, exit_code=completed.returncode,
                          stdout=completed.stdout[:4096], stderr=completed.stderr[:4096])
            if task == "cpu_load" and result["ok"]:
                result["measurement"] = "1, 5 and 15 minute load averages; not CPU utilization percentages"
        except subprocess.TimeoutExpired:
            result["error"] = f"Command timed out after {self._timeout:g} seconds"
        except OSError as exc:
            result["error"] = f"Command unavailable: {exc.strerror or type(exc).__name__}"
        finally:
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
            with self._lock:
                self._recent.appendleft(result)
                self._active = None
        if self._bus:
            self._bus.emit("command", "finish" if result["ok"] else "error", label,
                           level="info" if result["ok"] else "warning",
                           meta={"task": task, "ok": result["ok"], "elapsed_ms": result["elapsed_ms"]})
        return deepcopy(result)


tool_definition = {
    "type": "function",
    "function": {
        "name": "run_safe_command",
        "description": "Read host uptime, disk space, RAM usage, CPU load or operating system information. "
                       "cpu_load returns 1, 5 and 15 minute load averages, not utilization percentages. "
                       "Runs a fixed read-only command in the command slot. No custom commands or arguments.",
        "parameters": {
            "type": "object", "properties": {
                "task": {"type": "string", "enum": list(COMMANDS)},
            }, "required": ["task"], "additionalProperties": False,
        },
    },
}


class RunSafeCommand:
    def __init__(self, llm_queue: queue.Queue, tool_config: dict[str, Any] | None = None) -> None:
        self.llm_queue = llm_queue
        self.runner = (tool_config or {}).get("command_runner")

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        try:
            result = self.runner.run(call_args) if self.runner else {"ok": False, "error": "Command slot unavailable"}
        except ValueError as exc:
            result = {"ok": False, "error": str(exc)}
        self.llm_queue.put({"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)})
