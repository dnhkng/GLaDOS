"""Existing task workers with a bounded serial search group and inspectable handles."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time

from .event_bus import EventBus
from .slots import TaskSlotStore


@dataclass(frozen=True)
class TaskResult:
    status: str
    summary: str
    notify_user: bool = True
    importance: float | None = None
    confidence: float | None = None
    next_run: float | None = None
    report: str | None = None
    update_priority: str | None = None


@dataclass
class TaskHandle:
    id: str
    title: str
    group: str
    future: Future = field(default_factory=Future)
    cancelled: threading.Event = field(default_factory=threading.Event)
    execution: Future | None = None
    status: str = "queued"
    started_at: float | None = None


class TaskManager:
    def __init__(
        self, slot_store: TaskSlotStore, event_bus: EventBus, max_workers: int = 2, progress_interval_s: float = 5.0
    ) -> None:
        self._slot_store = slot_store
        slot_store.bind_events(event_bus)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._search_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SearchQueue")
        self._lock = threading.RLock()
        self._jobs: dict[str, TaskHandle] = {}
        self._stopped = False
        self._progress_interval_s = max(0.01, progress_interval_s)

    def submit(
        self,
        slot_id: str,
        title: str,
        runner: Callable[[], str | TaskResult],
        notify_user: bool = True,
        progress: Callable[[], str] | None = None,
        group: str = "background",
        cancelled: threading.Event | None = None,
    ) -> TaskHandle:
        with self._lock:
            if self._stopped:
                raise ValueError("Task manager is shutting down")
            if slot_id in self._jobs:
                raise ValueError("Task ID already exists")
            active = [h for h in self._jobs.values() if h.group == group and not h.future.done()]
            if group == "search" and len(active) >= 9:
                raise ValueError("Search queue is full (one running and eight waiting)")
            handle = TaskHandle(slot_id, title, group, cancelled=cancelled or threading.Event())
            self._jobs[slot_id] = handle
            self._slot_store.update_slot(
                slot_id, title, "queued", "Waiting to start", update_priority="regular", owner_id=group
            )
            self._publish_group(group)
            executor = self._search_executor if group == "search" else self._executor
            handle.execution = executor.submit(self._run_task, handle, runner, notify_user, progress)
            return handle

    def _publish_group(self, group: str) -> None:
        if group != "search":
            return
        active = [h for h in self._jobs.values() if h.group == group and not h.future.done()]
        waiting = [h for h in active if h.status == "queued"]
        for position, handle in enumerate(waiting, 1):
            self._slot_store.update_slot(
                handle.id,
                handle.title,
                "queued",
                f"Waiting in queue, position {position}",
                update_priority="regular",
                owner_id=group,
                queue_position=position,
            )
        rows = []
        for handle in active:
            slot = self._slot_store.get_slot(handle.id)
            rows.append(f"[{handle.id}] {handle.title[:100]}: {slot.status}; {slot.summary[:180]}")
        self._slot_store.update_slot(
            "search",
            "Search Core",
            "active" if active else "idle",
            "\n".join(rows) if rows else "Ready for requested research; no queued searches",
            update_priority="regular",
        )

    def cancel(self, slot_id: str) -> bool:
        with self._lock:
            handle = self._jobs.get(slot_id)
            if handle is None:
                raise ValueError("Task not found")
            if handle.future.done():
                return False
            handle.cancelled.set()
            if handle.execution and handle.execution.cancel():
                self._finish(handle, TaskResult("cancelled", "Request cancelled"))
            return True

    def _finish(self, handle: TaskHandle, result: TaskResult) -> None:
        with self._lock:
            if handle.future.done():
                return
            if handle.cancelled.is_set():
                result = TaskResult("cancelled", "Request cancelled", report=result.report)
            handle.status = result.status
            slot = self._slot_store.update_slot(
                handle.id,
                handle.title,
                result.status,
                result.summary,
                report=result.report,
                importance=result.importance,
                confidence=result.confidence,
                update_priority=result.update_priority or ("important" if result.notify_user else "regular"),
                owner_id=handle.group,
            )
            if not result.notify_user:
                self._slot_store.mark_handled(slot.slot_id, slot.revision)
            handle.future.set_result(result)
            self._publish_group(handle.group)

    def _run_task(self, handle, runner, notify_user, progress):
        with self._lock:
            if handle.cancelled.is_set():
                self._finish(handle, TaskResult("cancelled", "Request cancelled"))
                return
            handle.status, handle.started_at = "running", time.monotonic()
            self._slot_store.update_slot(
                handle.id, handle.title, "running", "Working...", update_priority="regular", owner_id=handle.group
            )
            self._publish_group(handle.group)
        finished = threading.Event()

        def heartbeat():
            while not finished.wait(self._progress_interval_s):
                try:
                    detail = progress()[:240] if progress else "Working"
                    with self._lock:
                        self._slot_store.update_slot(
                            handle.id,
                            handle.title,
                            "running",
                            f"{time.monotonic() - handle.started_at:.0f}s elapsed: {detail}",
                            update_priority="regular",
                            owner_id=handle.group,
                        )
                        self._publish_group(handle.group)
                except Exception:
                    pass

        thread = threading.Thread(target=heartbeat, daemon=True, name="TaskProgress")
        thread.start()
        try:
            value = runner()
            result = (
                value
                if isinstance(value, TaskResult)
                else TaskResult("done", "Completed", notify_user, report=str(value))
            )
        except Exception as exc:
            result = TaskResult("error", "Task failed", notify_user, report=str(exc))
        finally:
            finished.set()
            thread.join()
        self._finish(handle, result)

    def shutdown(self, wait: bool = False, timeout: float | None = None) -> None:
        with self._lock:
            self._stopped = True
            ids = [h.id for h in self._jobs.values() if not h.future.done()]
        for slot_id in ids:
            self.cancel(slot_id)

        def close():
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._search_executor.shutdown(wait=wait, cancel_futures=True)

        if wait and timeout is not None:
            thread = threading.Thread(target=close, daemon=True)
            thread.start()
            thread.join(timeout)
        else:
            close()
