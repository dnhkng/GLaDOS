from dataclasses import dataclass
import threading
import time

from loguru import logger

from ..observability import ObservabilityBus


@dataclass
class TaskSlot:
    slot_id: str
    title: str
    status: str
    summary: str
    updated_at: float
    notify_user: bool = True
    importance: float | None = None
    confidence: float | None = None
    next_run: float | None = None


class TaskSlotStore:
    def __init__(self, observability_bus: ObservabilityBus | None = None) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, TaskSlot] = {}
        self._observability_bus = observability_bus

    def update_slot(
        self,
        slot_id: str,
        title: str,
        status: str,
        summary: str,
        notify_user: bool = True,
        updated_at: float | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        next_run: float | None = None,
    ) -> TaskSlot:
        if updated_at is None:
            updated_at = time.time()
        with self._lock:
            existing = self._slots.get(slot_id)
            if existing:
                if importance is None:
                    importance = existing.importance
                if confidence is None:
                    confidence = existing.confidence
                if next_run is None:
                    next_run = existing.next_run
            slot = TaskSlot(
                slot_id=slot_id,
                title=title,
                status=status,
                summary=summary,
                updated_at=updated_at,
                notify_user=notify_user,
                importance=importance,
                confidence=confidence,
                next_run=next_run,
            )
            self._slots[slot_id] = slot
        if existing is None or existing.status != status or existing.summary != summary:
            logger.success("Slot update: {} -> {} ({})", title, status, summary)
        if self._observability_bus:
            self._observability_bus.emit(
                source="autonomy",
                kind="slot.update",
                message=f"{title} -> {status}",
                meta={
                    "slot_id": slot_id,
                    "notify_user": notify_user,
                    "importance": importance,
                    "confidence": confidence,
                    "next_run": next_run,
                },
            )
        return slot

    def list_slots(self) -> list[TaskSlot]:
        with self._lock:
            return list(self._slots.values())

    def as_message(self) -> dict[str, str] | None:
        slots = self.list_slots()
        if not slots:
            return None
        lines = ["[tasks]"]
        for slot in slots:
            summary = slot.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            meta_parts = []
            if slot.importance is not None:
                meta_parts.append(f"importance={slot.importance:.2f}")
            if slot.confidence is not None:
                meta_parts.append(f"confidence={slot.confidence:.2f}")
            if slot.next_run is not None:
                meta_parts.append(f"next_run={slot.next_run:.0f}")
            meta_text = f" ({', '.join(meta_parts)})" if meta_parts else ""
            lines.append(f"- {slot.title}: {slot.status}{summary_text}{meta_text}")
        return {"role": "system", "content": "\n".join(lines)}
