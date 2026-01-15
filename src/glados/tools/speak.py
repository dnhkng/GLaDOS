import queue
from typing import Any

from loguru import logger

tool_definition = {
    "type": "function",
    "function": {
        "name": "speak",
        "description": "Speak the provided text aloud.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak.",
                },
            },
            "required": ["text"],
        },
    },
}


class Speak:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        tool_config = tool_config or {}
        self._tts_queue: queue.Queue[str] | None = tool_config.get("tts_queue")

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if self._tts_queue is None:
            error_msg = "error: TTS queue is unavailable"
            logger.error(f"Speak: {error_msg}")
            self._send_result(tool_call_id, error_msg)
            return

        text = str(call_args.get("text", "")).strip()
        if not text:
            error_msg = "error: no text provided to speak"
            logger.error(f"Speak: {error_msg}")
            self._send_result(tool_call_id, error_msg)
            return

        self._tts_queue.put(text)
        self._send_result(tool_call_id, "success")

    def _send_result(self, tool_call_id: str, content: str) -> None:
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "type": "function_call_output",
            }
        )
