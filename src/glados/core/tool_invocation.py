"""Execution context for one authorized tool invocation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tool_executor import ToolResultQueue


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
    llm_queue: ToolResultQueue | None = None
    lane: str = "priority"
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def results(self) -> ToolResultQueue:
        """Where this call's tool reports; set once the executor has authorized the call."""
        if self.llm_queue is None:
            raise RuntimeError(f"Tool call {self.tool_call_id or '?'} was used before it was prepared")
        return self.llm_queue
