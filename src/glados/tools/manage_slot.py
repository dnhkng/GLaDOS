"""The reply agent and console use the same task records and validation."""

import json
import queue
from typing import Any
import uuid

from ..autonomy.slots import TaskSlot, TaskSlotStore


def save_task(store: TaskSlotStore, values: dict[str, Any]) -> TaskSlot:
    slot_id = values.get("slot_id") or "task_" + uuid.uuid4().hex[:12]
    if not isinstance(slot_id, str) or not slot_id.startswith("task_") or len(slot_id) > 80:
        raise ValueError("Only user task slots (task_ IDs) can be edited")
    old = store.get_slot(slot_id)
    if values.get("slot_id") and old is None:
        raise ValueError("Task not found")
    if old and old.owner_id:
        raise ValueError("Managed jobs cannot be edited; use cancel_task to cancel them")
    fields = {}
    for name, limit, default in (("title", 160, ""), ("summary", 2000, ""), ("report", 20000, "")):
        value = values.get(name, getattr(old, name, default) or default)
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"{name} must be text of at most {limit} characters")
        fields[name] = value.strip()
    if not fields["title"]:
        raise ValueError("A task title is required")
    status = values.get("status", old.status if old else "open")
    if status not in {"open", "done", "blocked"}:
        raise ValueError("Task status must be open, done, or blocked")
    return store.update_slot(slot_id, status=status, notify_user=False, **fields)


tool_definition = {
    "type": "function",
    "function": {
        "name": "manage_slot",
        "description": "Create or update a user task on the shared board. "
        "Saves a record, not a scheduled/background job. "
        "Omit slot_id to create; supply an existing task_ ID to update. Done requires a completed result.",
        "parameters": {
            "type": "object",
            "properties": {
                "slot_id": {"type": "string", "description": "Existing task_ ID; omit for a new task"},
                "title": {"type": "string"},
                "summary": {"type": "string", "description": "Short task description or result"},
                "status": {"type": "string", "enum": ["open", "done", "blocked"]},
                "report": {"type": "string", "description": "Full result, analysis, or draft"},
            },
            "required": ["title", "summary", "status"],
        },
    },
}


class ManageSlot:
    def __init__(self, llm_queue: queue.Queue, tool_config: dict[str, Any] | None = None) -> None:
        self.llm_queue = llm_queue
        self.tool_config = tool_config or {}

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        store = self.tool_config.get("slot_store")
        try:
            if store is None:
                raise ValueError("Task board unavailable")
            slot = save_task(store, call_args)
            content = json.dumps({"saved": True, "slot_id": slot.slot_id, "status": slot.status})
        except ValueError as exc:
            content = json.dumps({"error": str(exc)})
        self.llm_queue.put({"role": "tool", "tool_call_id": tool_call_id, "content": content})
