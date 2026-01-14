import queue
from typing import Any

from loguru import logger

from ..vision.constants import VISION_DETAIL_PROMPT
from ..vision.vision_request import VisionRequest

tool_definition = {
    "type": "function",
    "function": {
        "name": "vision_look",
        "description": "Capture the current camera view and describe it in detail.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Optional instruction for what to describe.",
                },
                "max_tokens": {
                    "type": "number",
                    "description": "Maximum tokens to generate for the description.",
                },
            },
        },
    },
}


class VisionLook:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the tool with a queue for communication with the LLM.

        Args:
            llm_queue: A queue for sending tool results to the language model.
            tool_config: Configuration dictionary containing tool settings.
        """
        self.llm_queue = llm_queue
        tool_config = tool_config or {}
        self._request_queue: queue.Queue[VisionRequest] | None = tool_config.get("vision_request_queue")
        self._timeout = float(tool_config.get("vision_tool_timeout", 30.0))
        self._default_prompt = tool_config.get("vision_detail_prompt", VISION_DETAIL_PROMPT)

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        """
        Executes a detailed vision request via the VisionProcessor thread.

        Args:
            tool_call_id: Unique identifier for the tool call.
            call_args: Arguments passed by the LLM related to this tool call.
        """
        if self._request_queue is None:
            error_msg = "error: vision tool is unavailable"
            logger.error(f"VisionLook: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )
            return

        prompt = str(call_args.get("prompt") or self._default_prompt).strip()
        max_tokens = call_args.get("max_tokens", 256)
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = 256
        max_tokens = max(1, min(max_tokens, 512))

        response_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        request = VisionRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            response_queue=response_queue,
        )

        try:
            self._request_queue.put(request, timeout=self._timeout)
        except queue.Full:
            error_msg = "error: vision request queue is full"
            logger.error(f"VisionLook: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )
            return

        try:
            result = response_queue.get(timeout=self._timeout)
        except queue.Empty:
            error_msg = "error: vision request timed out"
            logger.error(f"VisionLook: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )
            return

        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
                "type": "function_call_output",
            }
        )
