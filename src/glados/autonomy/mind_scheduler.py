"""Shared timing and lifecycle for minds; inference admission remains separate."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time

from loguru import logger

from ..observability import MindRegistry, ObservabilityBus
from .mind_schedule import Schedule
from .slots import TaskSlotStore
from .subagent import Subagent, SubagentOutput


@dataclass
class SubagentStatus:
    agent_id: str
    title: str
    running: bool
    tick_count: int
    last_tick: float
    paused: bool
    status: str
    interval_s: float | None
    next_due_in_s: float | None


@dataclass
class _Entry:
    mind: Subagent
    policy: Schedule
    run_on_start: bool
    anchor: float = 0
    pending: bool = False
    manual: bool = False
    busy: bool = False
    initialized: bool = False
    initializing: bool = False
    count: int = 0
    last_run: float = 0
    status: str = "stopped"
    controlling: bool = False


class MindScheduler:
    """One timer thread dispatches bounded work, never runs a mind itself.

    Deadlines use monotonic time and reset after completion. A mind cannot overlap
    itself. Triggers during work coalesce into at most one subsequent run.
    """

    def __init__(
        self,
        slot_store: TaskSlotStore,
        mind_registry: MindRegistry | None = None,
        observability_bus: ObservabilityBus | None = None,
        shutdown_event: threading.Event | None = None,
        max_workers: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_workers < 1:
            raise ValueError("Mind worker count must be positive")
        self._registry = mind_registry
        self._bus = observability_bus
        self._shutdown = shutdown_event or threading.Event()
        self._closed = False
        self._clock = clock
        self._capacity = max_workers
        self._active = 0
        self._condition = threading.Condition()
        self._entries: dict[str, _Entry] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="MindWorker")
        self._thread: threading.Thread | None = None

    def register(self, mind: Subagent, policy: Schedule, *, run_on_start: bool = True) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Mind scheduler is shut down")
            if mind.agent_id in self._entries or mind.runtime.scheduler is not None:
                raise ValueError(f"Mind {mind.agent_id} is already registered")
            policy.delay()  # Validate configuration before accepting the mind.
            self._entries[mind.agent_id] = _Entry(mind, policy, run_on_start)
            mind.runtime.scheduler = self
            if self._registry:
                self._registry.register(mind.agent_id, mind.title, "stopped", "Registered", role=mind.config.role)

    def get(self, agent_id: str) -> Subagent | None:
        with self._condition:
            entry = self._entries.get(agent_id)
            return entry.mind if entry else None

    def start(self, agent_id: str) -> None:
        with self._condition:
            if self._closed or self._shutdown.is_set():
                raise RuntimeError("Mind scheduler is shut down")
            entry = self._entries[agent_id]
            if entry.initialized:
                return
            entry.initialized = entry.initializing = True
            generation = entry.mind.runtime.generation
        try:
            entry.mind.on_start()
        except Exception:
            with self._condition:
                entry.initializing = False
                self._state(entry, "error", "Startup failed")
                self._condition.notify_all()
            self._cleanup(entry)
            raise
        with self._condition:
            entry.initializing = False
            cancelled = self._closed or self._shutdown.is_set() or generation != entry.mind.runtime.generation
            if not cancelled:
                try:
                    entry.policy.reset()
                except Exception as exc:
                    self._policy_error(entry, exc)
                    cancelled = True
            if not cancelled:
                entry.anchor = self._clock()
                entry.pending = (entry.pending or entry.run_on_start) and not entry.mind.paused
                entry.mind.runtime.enabled = True
                self._state(entry, "paused" if entry.mind.paused else "waiting", "Ready")
                if self._thread is None:
                    self._thread = threading.Thread(target=self._dispatch, name="MindScheduler", daemon=True)
                    self._thread.start()
            self._condition.notify_all()
        if cancelled:
            self._cleanup(entry)
            return
        if self._bus:
            self._bus.emit("subagent", "start", f"{entry.mind.title} started", meta={"agent_id": agent_id})

    def start_all(self) -> None:
        for agent_id in list(self._entries):
            try:
                self.start(agent_id)
            except Exception:
                logger.exception("Mind {} startup failed", agent_id)

    def trigger(self, agent_id: str, *, manual: bool = True) -> None:
        with self._condition:
            entry = self._entries[agent_id]
            if self._closed or self._shutdown.is_set() or (manual and not entry.mind.is_running):
                raise ValueError("Mind is not running")
            if entry.mind.paused and not manual:
                return
            entry.pending = True
            entry.manual |= manual
            self._condition.notify_all()

    def pause(self, agent_id: str, paused: bool = True) -> None:
        with self._condition:
            entry = self._entries[agent_id]
            runtime = entry.mind.runtime
            if runtime.paused == paused:
                return
            entry.controlling = True
            runtime.paused = paused
            runtime.generation += 1
            entry.pending = entry.manual = False
            if not paused:
                entry.anchor = self._clock()
                entry.policy.reset()
                entry.pending = entry.policy.delay() is not None
            if not entry.busy:
                self._state(entry, "paused" if paused else "waiting", "Paused" if paused else "Ready")
            self._condition.notify_all()
        try:
            entry.mind.on_pause(paused)
        finally:
            with self._condition:
                entry.controlling = False
                self._condition.notify_all()

    def resume(self, agent_id: str) -> None:
        self.pause(agent_id, False)

    def reschedule(self, agent_id: str) -> None:
        """Reevaluate changed policy settings without executing domain work."""
        with self._condition:
            entry = self._entries[agent_id]
            entry.policy.reset()
            self._condition.notify_all()

    def delay(self, agent_id: str) -> float | None:
        with self._condition:
            return self._delay(self._entries[agent_id])

    def _policy_error(self, entry: _Entry, exc: Exception) -> None:
        entry.mind.runtime.enabled = False
        entry.mind.runtime.generation += 1
        entry.pending = entry.manual = False
        self._state(entry, "error", f"Scheduling failed: {exc}")
        logger.error("Mind {} scheduling failed: {}", entry.mind.agent_id, exc)
        if not entry.busy and not self._closed:
            self._pool.submit(self._cleanup, entry)

    def _delay(self, entry: _Entry) -> float | None:
        if entry.status == "error":
            return None
        try:
            return entry.policy.delay()
        except Exception as exc:
            self._policy_error(entry, exc)
            return None

    def _due(self, entry: _Entry, now: float) -> bool:
        if not entry.mind.is_running or entry.busy or entry.controlling:
            return False
        if entry.pending:
            return True
        delay = self._delay(entry)
        return not entry.mind.paused and delay is not None and now >= entry.anchor + delay

    def _dispatch(self) -> None:
        with self._condition:
            while not self._closed and not self._shutdown.is_set():
                now = self._clock()
                for entry in self._entries.values():
                    if not self._due(entry, now):
                        continue
                    if self._active >= self._capacity:
                        self._state(entry, "queued", "Waiting for a mind worker")
                        continue
                    runtime = entry.mind.runtime
                    runtime.manual = entry.manual
                    runtime.executing = True
                    runtime.run_generation = runtime.generation
                    entry.pending = entry.manual = False
                    entry.busy = True
                    entry.count += 1
                    entry.last_run = time.time()
                    self._active += 1
                    self._state(entry, "running", f"Run #{entry.count}")
                    self._pool.submit(self._execute, entry)
                # Adaptive policies must notice changing motion without redrawing.
                deadlines = [
                    e.anchor + d
                    for e in self._entries.values()
                    if e.mind.is_running
                    and not e.busy
                    and not e.mind.paused
                    and not e.pending
                    and (d := self._delay(e)) is not None
                ]
                wait = (
                    min(1.0, max(0.001, min(deadlines) - self._clock()))
                    if deadlines and self._active < self._capacity
                    else 1.0
                )
                self._condition.wait(wait)

    def _execute(self, entry: _Entry) -> None:
        runtime = entry.mind.runtime
        try:
            if not runtime.cancelled():
                runtime.publish(entry.mind.run(runtime))
        except Exception as exc:
            logger.exception("Mind {} failed", entry.mind.agent_id)
            runtime.publish(SubagentOutput(status="error", summary=f"Run failed: {exc}", update_priority="regular"))
        finally:
            with self._condition:
                runtime.executing = runtime.manual = False
                entry.busy = False
                self._active -= 1
                entry.anchor = self._clock()
                try:
                    entry.policy.reset()
                except Exception as exc:
                    self._policy_error(entry, exc)
                self._state(
                    entry,
                    "error"
                    if entry.status == "error"
                    else "stopped"
                    if not runtime.enabled
                    else "paused"
                    if runtime.paused
                    else "waiting",
                    "Stopped" if not runtime.enabled else "Paused" if runtime.paused else "Waiting for next run",
                )
                self._condition.notify_all()
            if not runtime.enabled:
                self._cleanup(entry)

    def _state(self, entry: _Entry, status: str, summary: str) -> None:
        entry.status = status
        if self._registry:
            self._registry.update(entry.mind.agent_id, status, summary)

    def _cleanup(self, entry: _Entry) -> None:
        with self._condition:
            if entry.busy or entry.initializing or not entry.initialized:
                return
            entry.initialized = False
        try:
            entry.mind.on_stop()
        except Exception:
            logger.exception("Mind {} cleanup failed", entry.mind.agent_id)

    def stop(self, agent_id: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            entry = self._entries[agent_id]
            entry.mind.runtime.enabled = False
            entry.mind.runtime.generation += 1
            entry.pending = entry.manual = False
            self._condition.notify_all()
            while (entry.busy or entry.initializing) and (remaining := deadline - time.monotonic()) > 0:
                self._condition.wait(remaining)
            if not entry.busy:
                self._state(entry, "stopped", "Stopped")
        self._cleanup(entry)

    def list_agents(self) -> list[SubagentStatus]:
        with self._condition:
            now = self._clock()
            result = []
            for entry in self._entries.values():
                delay = self._delay(entry)
                due = (
                    None
                    if entry.mind.paused or not entry.mind.is_running or delay is None
                    else 0.0
                    if entry.pending
                    else max(0.0, entry.anchor + delay - now)
                )
                result.append(
                    SubagentStatus(
                        entry.mind.agent_id,
                        entry.mind.title,
                        entry.mind.is_running,
                        entry.count,
                        entry.last_run,
                        entry.mind.paused,
                        entry.status,
                        delay,
                        due,
                    )
                )
            return result

    def shutdown(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closed = True
            # Invalidate every mind before waiting for any individual worker.
            for entry in self._entries.values():
                entry.mind.runtime.enabled = False
                entry.mind.runtime.generation += 1
                entry.pending = entry.manual = False
            self._condition.notify_all()
        for agent_id in list(self._entries):
            self.stop(agent_id, max(0, deadline - time.monotonic()))
        if self._thread:
            self._thread.join(max(0, deadline - time.monotonic()))
        self._pool.shutdown(wait=False, cancel_futures=True)
