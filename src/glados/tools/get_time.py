"""Timezone-specific clock queries; local time is already supplied in context."""

import json
import queue
from typing import Any

from ..core.clock import current_time

tool_definition = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Read the current time/date in another explicitly named IANA timezone. "
                       "Local time/date/weekday are in live context; answer those without a tool call.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "minLength": 1, "maxLength": 100,
                             "description": "Required IANA zone, e.g. Asia/Tokyo or America/New_York."}
            },
            "required": ["timezone"],
            "additionalProperties": False,
        },
    },
}


class GetTime:
    def __init__(self, llm_queue: queue.Queue, tool_config: dict[str, Any] | None = None) -> None:
        self.llm_queue = llm_queue

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        try:
            timezone = call_args.get("timezone")
            if not timezone:
                raise ValueError("Use the live context for local time; this tool requires a named timezone")
            result = current_time(timezone)
        except ValueError as exc:
            result = {"error": str(exc)}
        self.llm_queue.put({"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)})
