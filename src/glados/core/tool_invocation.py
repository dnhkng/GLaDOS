"""Execution context for one authorized tool invocation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolInvocation:
    tool_call: dict[str, Any] = field(default_factory=dict)
    generation: int | None = None
    autonomy_mode: bool = False
    autonomy_epoch: int = 0
    cancelled: Callable[[], bool] = lambda: False
    tool: str = ""
    tool_call_id: str = ""
    started_at: float = 0.0
    autonomy_flag: dict[str, bool] = field(default_factory=dict)
    base_queue: Any = None
    llm_queue: Any = None
    lane: str = "priority"
    args: dict[str, Any] = field(default_factory=dict)
