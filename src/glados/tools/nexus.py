"""Nexus Core controls and background consultation of external advisor models."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .json_tool import JsonTool

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
                "need": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": "Why you need help and what you will do with the answer.",
                },
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4000,
                    "description": "The self-contained question for the advisor.",
                },
                "context": {
                    "type": "string",
                    "maxLength": 4000,
                    "description": "Only the conversation details the advisor needs.",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


class _BaseNexusTool(JsonTool):
    def _get_agent(self) -> NexusAgent:
        core: NexusAgent | None = self.config.get("nexus_agent")
        if core is None:
            raise ValueError("Nexus Core unavailable")
        return core


class NexusControl(_BaseNexusTool):
    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._get_agent().control(str(args.get("action")), args.get("preference"))


class NexusConsult(_BaseNexusTool):
    """Synchronous fallback; the tool executor normally runs consultations as background tasks."""

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        report: dict[str, Any] = json.loads(
            self._get_agent().consult(args, cancelled=self.config.get("_cancelled", lambda: False))
        )
        return report
