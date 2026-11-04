"""Nexus Core controls and background consultation of external advisor models."""

from __future__ import annotations

import json
import queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..autonomy.agents.nexus_agent import NexusAgent

control_definition = {
    "type": "function",
    "function": {
        "name": "nexus_control",
        "description": (
            "Turn Nexus Core consultation on or off when the user asks (e.g. 'activate the Nexus Core'), "
            "report its status, base model and spending, or save/clear a standing preference about which "
            "external models to use."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["on", "off", "status", "prefer", "clear_preferences"]},
                "preference": {"type": "string", "maxLength": 300},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}
consult_definition = {
    "type": "function",
    "function": {
        "name": "nexus_consult",
        "description": (
            "Ask an external advisor AI in the background when answering alone would be too slow, risk a "
            "mistake, or needs something you cannot do. Returns at once; keep talking. The advisor's answer "
            "is handed back to you later to interpret before you answer that question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "need": {"type": "string", "maxLength": 1000,
                         "description": "Why you need help and what you will do with the answer."},
                "question": {"type": "string", "minLength": 1, "maxLength": 4000,
                             "description": "The self-contained question for the advisor."},
                "context": {"type": "string", "maxLength": 4000,
                            "description": "Only the conversation details the advisor needs."},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


class _BaseNexusTool:
    def __init__(self, llm_queue: queue.Queue[dict[str, Any]], tool_config: dict[str, Any] | None = None) -> None:
        self.queue = llm_queue
        self.config = tool_config or {}

    def _get_agent(self) -> NexusAgent:
        core = self.config.get("nexus_agent")
        if core is None:
            raise ValueError("Nexus Core unavailable")
        return core

    def execute(self, args: dict[str, Any]) -> dict:
        raise NotImplementedError

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        try:
            result = self.execute(call_args)
        except (ValueError, TypeError, OSError) as exc:
            result = {"error": str(exc)}
        self.queue.put({"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)})


class NexusControl(_BaseNexusTool):
    def execute(self, args: dict[str, Any]) -> dict:
        return self._get_agent().control(str(args.get("action")), args.get("preference"))


class NexusConsult(_BaseNexusTool):
    """Synchronous fallback; the tool executor normally runs consultations as background tasks."""

    def execute(self, args: dict[str, Any]) -> dict:
        return json.loads(self._get_agent().consult(args, cancelled=self.config.get("_cancelled", lambda: False)))
