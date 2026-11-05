"""Tool schemas the model sees for the agent tools."""

from __future__ import annotations

from typing import Any

from .base import _definition
from .browser import BROWSER_ACTIONS
from .desktop import DESKTOP_ACTIONS

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
        "Read a text file (plain text; secrets are masked and long files are shortened, so change files with "
        "edit_file rather than rewriting what you read).",
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
        "browser",
        "Drive your own Chromium window (separate profile, visible to the owner) to visit sites, read pages, follow "
        "links, fill forms and click. Actions: " + "; ".join(f"{n} ({w})" for n, w in BROWSER_ACTIONS.items()),
        {
            "action": {"type": "string", "enum": list(BROWSER_ACTIONS)},
            "value": {"type": "string"},
            "selector": {"type": "string"},
        },
        ["action"],
    ),
    _definition(
        "agent_notes",
        "Your durable notes for agent work: read them before an unfamiliar task; add short facts worth reusing "
        "(project paths, working commands, the owner's habits).",
        {"action": {"type": "string", "enum": ["read", "add"]}, "text": {"type": "string"}},
        ["action"],
    ),
    _definition(
        "schedule",
        "Plan a reminder or a recurring job: when due, the task text is sent to you as a new request. "
        "add needs task plus when (systemd OnCalendar, e.g. '*-*-* 08:00', 'Mon *-*-* 18:30', '2026-11-06 14:00') "
        "or delay (e.g. '10min', '2h'). list shows them; remove needs id.",
        {
            "action": {"type": "string", "enum": ["add", "list", "remove"]},
            "task": {"type": "string", "description": "What to do or say when due, written as an instruction"},
            "when": {"type": "string"},
            "delay": {"type": "string"},
            "id": {"type": "string"},
        },
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
