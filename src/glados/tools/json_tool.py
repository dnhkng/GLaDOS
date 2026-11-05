"""Base for tools that run one synchronous action and report its JSON result."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any

from .protocol import MessageSink


class JsonTool(ABC):
    """Reports ``execute``'s result as JSON, or an ``{"error": ...}`` object when the request is invalid."""

    def __init__(self, llm_queue: MessageSink, tool_config: dict[str, Any] | None = None) -> None:
        self.queue = llm_queue
        self.config = tool_config or {}

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> object: ...

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        try:
            result = self.execute(call_args)
        except (ValueError, TypeError, OSError) as exc:
            result = {"error": str(exc)}
        self.queue.put({"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)})
