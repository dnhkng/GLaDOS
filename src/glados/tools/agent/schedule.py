"""schedule: reminders and recurring jobs as systemd user timers."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any
import uuid

from .base import SCHEDULED_PREFIX, _AgentTool

SCHEDULE_DIR = Path("~/.config/glados/scheduled")
UNIT_DIR = Path("~/.config/systemd/user")


class Schedule(_AgentTool):
    """Reminders and recurring jobs as systemd user timers that send their text to GLaDOS when due."""

    name = "schedule"

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        action = args.get("action", "list")
        if action == "list":
            return {"ok": True, "scheduled": self.entries()}
        if action == "remove":
            return self.remove(str(args.get("id") or ""))
        if action != "add":
            raise ValueError("action must be add, list or remove")
        task = " ".join(str(args.get("task") or "").split())
        when, delay = str(args.get("when") or "").strip(), str(args.get("delay") or "").strip()
        if not task or bool(when) == bool(delay):
            raise ValueError("Give task and exactly one of when (systemd OnCalendar) or delay (e.g. 10min)")
        # A scheduled message must never carry the owner's code into a later, unattended turn.
        task = re.sub(re.escape(self.config.confirm_code), "[code]", task, flags=re.IGNORECASE)[:1000]
        if when:
            check = self.execute(["systemd-analyze", "calendar", when], 5)
            if not check["ok"]:
                raise ValueError(f"Invalid calendar expression: {check['stderr'][:200]}")
        elif not re.fullmatch(r"\d+\s*(s|sec|m|min|h|hour|d|day)s?", delay):
            raise ValueError("delay looks like 30s, 10min, 2h or 1d")
        task_id = uuid.uuid4().hex[:8]
        unit = f"glados-sched-{task_id}"
        folder = SCHEDULE_DIR.expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        body = folder / f"{task_id}.json"
        body.write_text(json.dumps({"text": f"{SCHEDULED_PREFIX}{task_id}] {task}"}, ensure_ascii=False))
        ca = Path(self.config.console_ca)
        curl = [
            "/usr/bin/curl",
            "-s",
            "--cacert",
            str(ca if ca.is_absolute() else Path.cwd() / ca),
            "-H",
            "Content-Type: application/json",
            "-d",
            f"@{body}",
            self.config.console_url.rstrip("/") + "/api/input",
        ]
        if delay:
            started = self.execute(["systemd-run", "--user", f"--unit={unit}", f"--on-active={delay}", *curl], 10)
        else:
            units = UNIT_DIR.expanduser()
            units.mkdir(parents=True, exist_ok=True)
            (units / f"{unit}.service").write_text(
                f"[Unit]\nDescription=GLaDOS scheduled task: {task[:80]}\n\n[Service]\nType=oneshot\n"
                f"ExecStart={' '.join(_systemd_quote(part) for part in curl)}\n"
            )
            (units / f"{unit}.timer").write_text(
                f"[Unit]\nDescription=GLaDOS scheduled task: {task[:80]}\n\n[Timer]\nOnCalendar={when}\n"
                "Persistent=true\n\n[Install]\nWantedBy=timers.target\n"
            )
            self.execute(["systemctl", "--user", "daemon-reload"], 10)
            started = self.execute(["systemctl", "--user", "enable", "--now", f"{unit}.timer"], 10)
        if not started["ok"]:
            body.unlink(missing_ok=True)
            return started
        created = time.time()
        meta = {"id": task_id, "task": task, "when": when or None, "delay": delay or None, "created": created}
        (folder / f"{task_id}.meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        next_run: str | None = (
            time.strftime("%a %Y-%m-%d %H:%M:%S", time.localtime(created + _seconds(delay)))
            if delay
            else self.next_run(unit)
        )
        return {"ok": True, "id": task_id, "next": next_run}

    def next_run(self, unit: str) -> str | None:
        shown = self.execute(["systemctl", "--user", "show", f"{unit}.timer", "-p", "NextElapseUSecRealtime"], 5)
        value = shown.get("stdout", "").strip().partition("=")[2]
        return value or None

    def entries(self) -> list[dict[str, Any]]:
        rows = []
        for meta in sorted(SCHEDULE_DIR.expanduser().glob("*.meta.json")):
            entry = json.loads(meta.read_text())
            if entry.get("delay"):
                due = entry["created"] + _seconds(entry["delay"])
                if due < time.time():  # A one-off reminder that already fired.
                    meta.unlink(missing_ok=True)
                    (SCHEDULE_DIR.expanduser() / f"{entry['id']}.json").unlink(missing_ok=True)
                    continue
                entry["next"] = time.strftime("%a %Y-%m-%d %H:%M:%S", time.localtime(due))
            else:
                entry["next"] = self.next_run(f"glados-sched-{entry['id']}")
            rows.append(entry)
        return rows

    def remove(self, task_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{8}", task_id):
            raise ValueError("Give the id of a scheduled task (see action list)")
        unit = f"glados-sched-{task_id}"
        self.execute(["systemctl", "--user", "disable", "--now", f"{unit}.timer"], 10)
        self.execute(["systemctl", "--user", "stop", f"{unit}.timer"], 10)
        for path in (
            UNIT_DIR.expanduser() / f"{unit}.service",
            UNIT_DIR.expanduser() / f"{unit}.timer",
            SCHEDULE_DIR.expanduser() / f"{task_id}.json",
            SCHEDULE_DIR.expanduser() / f"{task_id}.meta.json",
        ):
            path.unlink(missing_ok=True)
        self.execute(["systemctl", "--user", "daemon-reload"], 10)
        return {"ok": True, "removed": task_id}


def _seconds(delay: str) -> float:
    amount, unit = re.fullmatch(r"(\d+)\s*([a-z]+)", delay).groups()  # type: ignore[union-attr]
    return (
        int(amount)
        * {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hour": 3600, "d": 86400, "day": 86400}[
            unit.rstrip("s") or "s"
        ]
    )


def _systemd_quote(part: str) -> str:
    return '"' + part.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
