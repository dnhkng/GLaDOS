"""Read a camera overview or ask E4B a question about a fresh frame."""

import queue
from typing import Any

tool_definition = {
    "type": "function",
    "function": {
        "name": "vision_look",
        "description": (
            "Inspect the webcam using E4B. Supply question to check a specific visible detail on a fresh frame, "
            "such as whether the user's jacket has a zipper. Without question, read the latest scene and changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000,
                    "description": "The user's specific visual question, preserving the detail they want checked.",
                },
            },
            "additionalProperties": False,
        },
    },
}


class VisionLook:
    def __init__(self, llm_queue: queue.Queue[dict[str, Any]], tool_config: dict[str, Any] | None = None) -> None:
        self.llm_queue = llm_queue
        self._mind = (tool_config or {}).get("vision_agent")

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if self._mind is None:
            result = "error: Vision Core is unavailable"
        elif "question" in call_args:
            result = self._mind.ask(call_args["question"])
        else:
            result = self._mind.look()
        self.llm_queue.put(
            {"role": "tool", "tool_call_id": tool_call_id, "content": result, "type": "function_call_output"}
        )
