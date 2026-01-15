from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Callable

from loguru import logger

from .event_bus import EventBus
from .events import TaskUpdateEvent
from .slots import TaskSlotStore


class TaskManager:
    def __init__(self, slot_store: TaskSlotStore, event_bus: EventBus, max_workers: int = 2) -> None:
        self._slot_store = slot_store
        self._event_bus = event_bus
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def submit(
        self,
        slot_id: str,
        title: str,
        runner: Callable[[], str],
        notify_user: bool = True,
    ) -> None:
        self._slot_store.update_slot(slot_id, title, "queued", "Waiting to start", notify_user=notify_user)
        with self._lock:
            self._executor.submit(self._run_task, slot_id, title, runner, notify_user)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_task(
        self,
        slot_id: str,
        title: str,
        runner: Callable[[], str],
        notify_user: bool,
    ) -> None:
        self._slot_store.update_slot(slot_id, title, "running", "Working...", notify_user=notify_user)
        started_at = time.time()
        try:
            result = runner()
            summary = result.strip() if result else "Completed."
            status = "done"
        except Exception as exc:
            logger.error("TaskManager: task %s failed: %s", slot_id, exc)
            summary = f"Failed: {exc}"
            status = "error"

        updated_at = time.time()
        self._slot_store.update_slot(slot_id, title, status, summary, notify_user=notify_user, updated_at=updated_at)
        self._event_bus.publish(
            TaskUpdateEvent(
                slot_id=slot_id,
                title=title,
                status=status,
                summary=summary,
                notify_user=notify_user,
                updated_at=updated_at,
            )
        )
