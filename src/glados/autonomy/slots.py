from dataclasses import dataclass, replace
import threading
import time

from loguru import logger

from ..observability import ObservabilityBus
from .event_bus import EventBus
from .events import TaskUpdateEvent


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
    report: str | None = None  # Full detailed report, available on-demand
    context: str | None = None  # Bounded relevant facts supplied by a core.
    attention_key: str | None = None  # Stable identity for an unresolved condition/arrival.
    update_priority: str = "regular"
    revision: int = 0
    turn_id: str | None = None
    owner_id: str | None = None
    handled: bool = False
    queue_position: int | None = None


class TaskSlotStore:
    def __init__(self, observability_bus: ObservabilityBus | None = None, event_bus: EventBus | None = None) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, TaskSlot] = {}
        self._observability_bus = observability_bus
        self._event_bus = event_bus

    def bind_events(self, event_bus: EventBus) -> None:
        """Connect publication once; producers never send a second notification."""
        with self._lock:
            if self._event_bus is not None and self._event_bus is not event_bus:
                raise ValueError("Slot store is already connected to another event bus")
            self._event_bus = event_bus

    @staticmethod
    def update_event(slot: TaskSlot) -> TaskUpdateEvent:
        return TaskUpdateEvent(slot.slot_id, slot.title, slot.status, slot.summary,
                               slot.notify_user, slot.updated_at, slot.importance, slot.confidence,
                               slot.next_run, slot.attention_key, slot.update_priority, slot.revision)

    def update_slot(
        self,
        slot_id: str,
        title: str,
        status: str,
        summary: str,
        report: str | None = None,
        notify_user: bool | None = None,
        updated_at: float | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        next_run: float | None = None,
        context: str | None = None,
        attention_key: str | None = None,
        update_priority: str | None = None,
        turn_id: str | None = None,
        owner_id: str | None = None,
        queue_position: int | None = None,
    ) -> TaskSlot:
        # Legacy callers are adapted here. New producers use the explicit two-value label.
        priority = update_priority or ("important" if notify_user or attention_key else "regular")
        if priority not in {"regular", "important"}:
            raise ValueError("update_priority must be regular or important")
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
                notify_user=bool(notify_user) if notify_user is not None else priority == "important",
                importance=importance,
                confidence=confidence,
                next_run=next_run,
                context=context,
                report=report,
                attention_key=attention_key,
                update_priority=priority,
                turn_id=turn_id,
                owner_id=owner_id or (existing.owner_id if existing else None),
                queue_position=queue_position,
            )
            # Timestamps and elapsed progress do not create a new attention episode.
            changed = existing is None or any(
                getattr(existing, name) != getattr(slot, name)
                for name in ("title", "status", "summary", "report", "context", "update_priority",
                             "importance", "confidence", "next_run", "attention_key", "turn_id", "queue_position")
            )
            slot.handled = existing.handled if existing and not changed else False
            slot.revision = (existing.revision if existing else 0) + int(changed)
            self._slots[slot_id] = slot
            if changed and self._event_bus is not None:
                self._event_bus.publish(self.update_event(slot))
        level = "warning" if status == "error" else "info" if priority == "important" else "debug"
        if changed:
            logger.log(level.upper(), "Slot update: {} -> {} ({})", title, status, summary)
        if changed and self._observability_bus:
            self._observability_bus.emit(
                source="autonomy",
                kind="slot.update",
                message=f"{title} -> {status}",
                level=level,
                meta={
                    "slot_id": slot_id,
                    "notify_user": notify_user,
                    "update_priority": priority,
                    "importance": importance,
                    "confidence": confidence,
                    "next_run": next_run,
                },
            )
        return slot

    def mark_handled(self, slot_id: str, revision: int) -> None:
        """Retire a consumed terminal job result from routine context, retaining its report."""
        with self._lock:
            slot = self._slots.get(slot_id)
            if (slot and slot.revision == revision and slot.owner_id
                    and slot.status in {"done", "partial", "error", "cancelled"}):
                self._slots[slot_id] = replace(slot, handled=True)

    def list_slots(self) -> list[TaskSlot]:
        with self._lock:
            return list(self._slots.values())

    def get_slot(self, slot_id: str) -> TaskSlot | None:
        """Get a specific slot by ID."""
        with self._lock:
            return self._slots.get(slot_id)

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
            report_hint = " [report available]" if slot.report else ""
            lines.append(f"- {slot.title}: {slot.status}{summary_text}{meta_text}{report_hint}")
            if slot.context:
                lines.append(slot.context)
        return {"role": "system", "content": "\n".join(lines)}
