import queue
from typing import Any

tool_definition = {
    "type": "function",
    "function": {
        "name": "do_nothing",
        "description": "Explicitly do nothing.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class DoNothing:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "success",
                "type": "function_call_output",
            }
        )
