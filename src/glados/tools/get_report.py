import queue
from typing import Any

tool_definition = {
    "type": "function",
    "function": {
        "name": "get_report",
        "description": "Get detailed report from a subagent. Use when the summary in context isn't enough and you need more details.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the subagent (e.g., 'weather', 'hn_top', 'emotion')",
                }
            },
            "required": ["agent_id"],
        },
    },
}


class GetReport:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.tool_config = tool_config or {}

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        agent_id = call_args.get("agent_id", "")
        slot_store = self.tool_config.get("slot_store")

        if not slot_store:
            content = "Error: slot_store not available"
        elif not agent_id:
            content = "Error: agent_id is required"
        else:
            slot = slot_store.get_slot(agent_id)
            if slot is None:
                content = f"No slot found for agent '{agent_id}'"
            elif slot.report:
                content = slot.report
            else:
                content = f"No detailed report available for '{agent_id}'. Summary: {slot.summary}"

        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "type": "function_call_output",
            }
        )
