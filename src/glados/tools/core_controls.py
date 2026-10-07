"""Shared chat controls for memory records and background jobs."""

import json

memory_definition = {
    "type": "function",
    "function": {
        "name": "manage_memory",
        "description": "List/read saved facts and historical summaries, or edit/delete an exact record. Read first to obtain its ID and revision. Only change memories when requested by the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "get", "edit", "delete"]},
                "id": {"type": "string"},
                "revision": {"type": "string"},
                "content": {"type": "string"},
                "query": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["action"],
        },
    },
}
cancel_definition = {
    "type": "function",
    "function": {
        "name": "cancel_task",
        "description": "Cancel a running or queued background task by its exact slot ID.",
        "parameters": {"type": "object", "properties": {"slot_id": {"type": "string"}}, "required": ["slot_id"]},
    },
}


class ManageMemory:
    def __init__(self, llm_queue, tool_config=None):
        self.queue = llm_queue
        self.config = tool_config or {}

    def execute(self, args):
        core = self.config.get("memory_agent")
        if core is None:
            raise ValueError("Memory Core unavailable")
        action = args.get("action")
        if action == "list":
            return core.memory_snapshot(args.get("query", ""), offset=args.get("offset", 0))
        if not isinstance(args.get("id"), str):
            raise ValueError("An exact memory ID is required")
        if action == "get":
            return core.memory_entry(args["id"])
        return core.mutate_memory(args["id"], action, args.get("revision"), args.get("content"))

    def run(self, tool_call_id, call_args):
        try:
            result = self.execute(call_args)
        except (ValueError, TypeError, OSError) as exc:
            result = {"error": str(exc)}
        self.queue.put({"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)})


class CancelTask(ManageMemory):
    def execute(self, args):
        manager = self.config.get("task_manager")
        if manager is None:
            raise ValueError("Task manager unavailable")
        slot_id = args.get("slot_id")
        if not isinstance(slot_id, str):
            raise ValueError("An exact task ID is required")
        return {"slot_id": slot_id, "cancellation_requested": manager.cancel(slot_id)}
