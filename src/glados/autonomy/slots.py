from dataclasses import dataclass
import threading
import time


@dataclass
class TaskSlot:
    slot_id: str
    title: str
    status: str
    summary: str
    updated_at: float
    notify_user: bool = True


class TaskSlotStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, TaskSlot] = {}

    def update_slot(
        self,
        slot_id: str,
        title: str,
        status: str,
        summary: str,
        notify_user: bool = True,
        updated_at: float | None = None,
    ) -> TaskSlot:
        if updated_at is None:
            updated_at = time.time()
        slot = TaskSlot(
            slot_id=slot_id,
            title=title,
            status=status,
            summary=summary,
            updated_at=updated_at,
            notify_user=notify_user,
        )
        with self._lock:
            self._slots[slot_id] = slot
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
            lines.append(f"- {slot.title}: {slot.status}{summary_text}")
        return {"role": "system", "content": "\n".join(lines)}
