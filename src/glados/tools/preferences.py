"""
Preferences tools for managing user preferences.

These tools allow the main agent to get and set user preferences,
which subagents can read to customize their behavior.
"""

import json
import queue
from typing import Any


get_preferences_definition = {
    "type": "function",
    "function": {
        "name": "get_preferences",
        "description": "Get current user preferences. Returns all stored preferences as JSON.",
        "parameters": {"type": "object", "properties": {}},
    },
}


set_preference_definition = {
    "type": "function",
    "function": {
        "name": "set_preference",
        "description": (
            "Set a user preference. Use this to remember user likes/dislikes. "
            "Examples: news_topics=['AI', 'science'], news_exclude=['crypto'], weather_units='celsius'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Preference key (e.g., 'news_topics', 'weather_units')",
                },
                "value": {
                    "description": "Preference value (string, number, boolean, or array)",
                },
            },
            "required": ["key", "value"],
        },
    },
}


class GetPreferences:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.preferences_store = (tool_config or {}).get("preferences_store")

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if not self.preferences_store:
            result = "error: preferences store not configured"
        else:
            prefs = self.preferences_store.all()
            result = json.dumps(prefs, indent=2) if prefs else "{}"
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
                "type": "function_call_output",
            }
        )


class SetPreference:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.preferences_store = (tool_config or {}).get("preferences_store")

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if not self.preferences_store:
            result = "error: preferences store not configured"
        else:
            key = call_args.get("key", "")
            value = call_args.get("value")
            if not key:
                result = "error: key is required"
            else:
                self.preferences_store.set(key, value)
                result = f"Set {key} = {json.dumps(value)}"
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
                "type": "function_call_output",
            }
        )
