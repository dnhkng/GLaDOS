"""
Subagent base class and configuration for the GLaDOS autonomy system.

Subagents are independent agents that run their own loops, process events,
and write outputs to slots for the main agent to consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from .subagent_memory import SubagentMemory

if TYPE_CHECKING:
    from .slots import TaskSlotStore
    from ..observability import MindRegistry, ObservabilityBus


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""

    agent_id: str
    title: str
    role: str = ""
    system_prompt: str = ""
    loop_interval_s: float = 10.0
    memory_max_entries: int = 100
    run_on_start: bool = True


@dataclass
class SubagentOutput:
    """Output from a subagent tick."""

    status: str
    summary: str
    notify_user: bool = True
    importance: float | None = None
    confidence: float | None = None
    next_run: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Subagent(ABC):
    """
    Base class for all subagents.

    Subagents run independently in their own threads, process on configurable
    intervals, and write outputs to slots that the main agent can read.
    """

    def __init__(
        self,
        config: SubagentConfig,
        slot_store: TaskSlotStore,
        mind_registry: MindRegistry | None = None,
        observability_bus: ObservabilityBus | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        self._config = config
        self._slot_store = slot_store
        self._mind_registry = mind_registry
        self._observability_bus = observability_bus
        self._shutdown_event = shutdown_event or threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_tick: float = 0.0
        self._tick_count: int = 0
        self._memory = SubagentMemory(
            agent_id=config.agent_id,
            max_entries=config.memory_max_entries,
        )

    @property
    def agent_id(self) -> str:
        return self._config.agent_id

    @property
    def title(self) -> str:
        return self._config.title

    @property
    def config(self) -> SubagentConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def memory(self) -> SubagentMemory:
        """Access this subagent's persistent memory."""
        return self._memory

    @abstractmethod
    def tick(self) -> SubagentOutput | None:
        """
        Execute one iteration of the subagent's work.

        Returns:
            SubagentOutput if there's something to report, None otherwise.
        """
        ...

    def on_start(self) -> None:
        """Called when the subagent starts. Override for initialization."""
        pass

    def on_stop(self) -> None:
        """Called when the subagent stops. Override for cleanup."""
        pass

    def run(self) -> None:
        """Main loop for the subagent. Runs tick() at configured intervals."""
        self._running = True
        logger.info("Subagent %s started.", self._config.agent_id)

        if self._mind_registry:
            self._mind_registry.register(
                mind_id=self._config.agent_id,
                title=self._config.title,
                status="running",
                summary="Starting",
                role=self._config.role,
            )

        if self._observability_bus:
            self._observability_bus.emit(
                source="subagent",
                kind="start",
                message=f"{self._config.title} started",
                meta={"agent_id": self._config.agent_id},
            )

        try:
            self.on_start()

            # Run immediately on start if configured
            if self._config.run_on_start:
                self._do_tick()

            while not self._shutdown_event.is_set():
                next_tick_in = self._config.loop_interval_s - (time.time() - self._last_tick)
                if next_tick_in > 0:
                    if self._shutdown_event.wait(timeout=min(next_tick_in, 1.0)):
                        break
                    continue

                self._do_tick()

        except Exception as exc:
            logger.exception("Subagent %s crashed: %s", self._config.agent_id, exc)
            if self._mind_registry:
                self._mind_registry.update(
                    self._config.agent_id,
                    status="error",
                    summary=f"Crashed: {exc}",
                )
        finally:
            self._running = False
            self.on_stop()

            if self._mind_registry:
                self._mind_registry.update(
                    self._config.agent_id,
                    status="stopped",
                    summary="Shutdown",
                )

            if self._observability_bus:
                self._observability_bus.emit(
                    source="subagent",
                    kind="stop",
                    message=f"{self._config.title} stopped",
                    meta={"agent_id": self._config.agent_id, "tick_count": self._tick_count},
                )

            logger.info("Subagent %s stopped.", self._config.agent_id)

    def _do_tick(self) -> None:
        """Execute a single tick and update slot."""
        self._last_tick = time.time()
        self._tick_count += 1

        if self._mind_registry:
            self._mind_registry.update(
                self._config.agent_id,
                status="running",
                summary=f"Tick #{self._tick_count}",
            )

        try:
            output = self.tick()
        except Exception as exc:
            logger.warning("Subagent %s tick failed: %s", self._config.agent_id, exc)
            output = SubagentOutput(
                status="error",
                summary=f"Tick failed: {exc}",
                notify_user=False,
            )

        if output is not None:
            self.write_slot(
                status=output.status,
                summary=output.summary,
                notify_user=output.notify_user,
                importance=output.importance,
                confidence=output.confidence,
                next_run=output.next_run,
            )

    def write_slot(
        self,
        status: str,
        summary: str,
        notify_user: bool = True,
        importance: float | None = None,
        confidence: float | None = None,
        next_run: float | None = None,
    ) -> None:
        """Write output to this subagent's slot."""
        self._slot_store.update_slot(
            slot_id=self._config.agent_id,
            title=self._config.title,
            status=status,
            summary=summary,
            notify_user=notify_user,
            importance=importance,
            confidence=confidence,
            next_run=next_run,
        )

    def start(self) -> threading.Thread:
        """Start the subagent in a new daemon thread."""
        if self._running:
            raise RuntimeError(f"Subagent {self._config.agent_id} is already running")

        self._thread = threading.Thread(
            target=self.run,
            name=f"Subagent-{self._config.agent_id}",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the subagent to stop and wait for it."""
        if not self._running:
            return

        self._shutdown_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Subagent %s did not stop within timeout.", self._config.agent_id)
