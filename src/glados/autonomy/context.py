"""Compact reviewer evidence, separate from the Central Core's response context."""

import json
import time
from typing import Any

from ..core.conversation_store import ConversationStore
from .slots import TaskSlot


def slot_version(slot: TaskSlot) -> tuple:
    return (slot.attention_key, slot.status, slot.summary, slot.context, slot.report)


def slot_evidence(slots: list[TaskSlot], budget_chars: int = 32000) -> list[dict[str, Any]]:
    """Bound details before summaries; handled results remain available via report lookup."""
    now = time.time()
    selected = [s for s in sorted(slots, key=lambda slot: slot.slot_id) if not s.handled]
    rows = [
        {
            "slot_id": s.slot_id,
            "title": s.title[:80],
            "status": s.status,
            "summary": s.summary[:200],
            "update_priority": s.update_priority,
            "attention_key": s.attention_key,
            "turn_id": s.turn_id,
            "queue_position": s.queue_position,
            "age_s": round(max(0, now - s.updated_at), 1),
        }
        for s in selected
    ]
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # Drop prose before states. Active jobs take precedence if a pathological number
    # of user records exceeds even the identity budget.
    if len(encode(rows)) > budget_chars:
        for row in rows:
            row.pop("summary", None)
            row.pop("title", None)
    if len(encode(rows)) > budget_chars:
        rows.sort(key=lambda row: row["status"] not in {"running", "queued"})
        omitted = 0
        while rows and len(encode(rows)) > max(2, budget_chars - 100):
            rows.pop()
            omitted += 1
        rows.append({"omitted_slots": omitted, "note": "Additional records available on the task board"})
    by_id = {row.get("slot_id"): row for row in rows}
    for slot in sorted(selected, key=lambda slot: slot.update_priority != "important"):
        row = by_id.get(slot.slot_id)
        if row is None:
            continue
        detail = slot.report or slot.context or ""
        if slot.context and slot.context not in detail:
            detail += "\n" + slot.context
        if not detail:
            continue
        low, high = 0, min(3500, len(detail))
        while low < high:
            mid = (low + high + 1) // 2
            row.update(evidence=detail[:mid], excerpt=mid < len(detail))
            if len(encode(rows)) <= budget_chars:
                low = mid
            else:
                high = mid - 1
        if low:
            row.update(evidence=detail[:low], excerpt=low < len(detail))
        else:
            row.pop("evidence", None)
            row.pop("excerpt", None)
    return rows


def chat_evidence(store: ConversationStore) -> str:
    records = store.records()
    summaries = [
        {"from": r.start_at, "until": r.end_at, "text": str(r.message.get("content", ""))[:1600]}
        for r in records
        if r.summary_level is not None or str(r.message.get("content", "")).startswith("[summary]")
    ][-8:]
    turns = [
        {"at": r.end_at, "role": r.message["role"], "text": r.message["content"][:1000]}
        for r in records
        if r.summary_level is None
        and r.message.get("role") in {"user", "assistant"}
        and isinstance(r.message.get("content"), str)
        and not r.message["content"].startswith("[summary]")
    ][-8:]
    return "[Quoted conversation evidence]\n" + json.dumps(
        {
            "summaries": summaries,
            "recent_turns": turns,
            "note": "No summary yet" if not summaries else "Existing Memory Core summaries; no new summarization call",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
