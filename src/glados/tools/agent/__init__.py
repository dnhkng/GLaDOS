"""Agent tools: shell, files and the Hyprland desktop, behind a blacklist that needs the owner's code.

Everything runs directly except actions matching ``AgentConfig.blacklist`` (commands),
``protected_paths`` (files) or scripts the agent wrote itself. Those run only when the owner's
latest message, spoken or typed in the last few minutes, contains ``confirm_code``, or when the
owner approves that exact action in the console. The model is never told the code, so it cannot
supply it itself, and messages relayed from other AIs through the API never count. Every call is
written to an audit log, and secrets are masked before output reaches the model.
"""

from .base import AgentConfig, AgentState, Guard, _AgentTool, _Collect, _definition, _notify_desktop, redact
from .browser import Browser
from .definitions import agent_definitions
from .desktop import Desktop
from .files import EditFile, ListDir, ReadFile, WriteFile
from .schedule import Schedule
from .shell import RunShell
from .worker import AgentNotes, DelegateTask

agent_classes: dict[str, type[_AgentTool]] = {
    "browser": Browser,
    "agent_notes": AgentNotes,
    "schedule": Schedule,
    "delegate_task": DelegateTask,
    "run_shell": RunShell,
    "read_file": ReadFile,
    "write_file": WriteFile,
    "edit_file": EditFile,
    "list_dir": ListDir,
    "desktop": Desktop,
}

__all__ = [
    "AgentConfig",
    "AgentNotes",
    "AgentState",
    "Browser",
    "DelegateTask",
    "Desktop",
    "EditFile",
    "Guard",
    "ListDir",
    "ReadFile",
    "RunShell",
    "Schedule",
    "WriteFile",
    "_AgentTool",
    "_Collect",
    "_definition",
    "_notify_desktop",
    "agent_classes",
    "agent_definitions",
    "redact",
]
