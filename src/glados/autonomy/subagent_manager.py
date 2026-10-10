"""Compatibility imports; MindScheduler owns mind timing and lifecycle."""

from .mind_scheduler import MindScheduler, SubagentStatus

SubagentManager = MindScheduler

__all__ = ["SubagentManager", "SubagentStatus"]
