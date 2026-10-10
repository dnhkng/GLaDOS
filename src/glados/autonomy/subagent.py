"""
Subagent base class and configuration for the GLaDOS autonomy system.

Subagents perform domain work and publish outputs to context slots.
The shared mind scheduler owns their lifecycle and timing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
from typing import TYPE_CHECKING, Any

from .mind_runtime import MindRuntime
from .subagent_memory import SubagentMemory

if TYPE_CHECKING:
    from ..observability import MindRegistry, ObservabilityBus
    from .slots import TaskSlotStore


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""

    agent_id: str
    title: str
    role: str = ""
    memory_max_entries: int = 100


@dataclass
class SubagentOutput:
    """Output from a subagent tick.

    Attributes:
        status: Current status (e.g., "done", "update", "error", "idle")
        summary: Short text for context injection (~20 tokens)
        report: Full detailed report, available on-demand via get_report tool
        notify_user: Whether this update should notify the user
        importance: Priority signal 0.0-1.0
        confidence: Confidence score 0.0-1.0
        next_run: Optional progress metadata (does not control scheduling)
        raw: Arbitrary data for internal use
    """

    status: str
    summary: str
    report: str | None = None
    context: str | None = None
    turn_id: str | None = None
    notify_user: bool = False  # Compatibility for older producers; use update_priority in new code.
    importance: float | None = None
    confidence: float | None = None
    next_run: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    attention_key: str | None = None
    update_priority: str | None = None


class Subagent(ABC):
    """Domain worker. Timing, threads and execution status belong to MindScheduler.

    Constructors receive only domain dependencies; inference is optional.
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
        self._observability_bus = observability_bus
        self._shutdown_event = shutdown_event or threading.Event()
        self.runtime = MindRuntime(self, self._shutdown_event)
        self._memory = SubagentMemory(agent_id=config.agent_id, max_entries=config.memory_max_entries)

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
        return self.runtime.enabled

    @property
    def paused(self) -> bool:
        return self.runtime.paused

    @property
    def memory(self) -> SubagentMemory:
        return self._memory

    @abstractmethod
    def run(self, runtime: MindRuntime) -> SubagentOutput | None:
        """Perform one unit of work, returning an optional final update."""
        ...

    def on_start(self) -> None:
        return None

    def on_stop(self) -> None:
        return None

    def on_pause(self, paused: bool) -> None:
        """Optional domain cleanup, such as disabling camera capture."""
        return None

    def set_paused(self, paused: bool) -> None:
        """Compatibility control; the scheduler owns pause state."""
        self.runtime.set_paused(paused)

    def request_tick(self) -> None:
        """Compatibility control for callers requesting a manual run."""
        if not self.runtime.scheduler:
            raise ValueError("Mind is not registered with a scheduler")
        self.runtime.scheduler.trigger(self.agent_id)

    def write_slot(
        self,
        status: str,
        summary: str,
        report: str | None = None,
        notify_user: bool | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        next_run: float | None = None,
        context: str | None = None,
        attention_key: str | None = None,
        update_priority: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Publish domain context; progress and returned updates share this path."""
        self._slot_store.update_slot(
            slot_id=self.agent_id,
            title=self.title,
            status=status,
            summary=summary,
            report=report,
            notify_user=notify_user,
            importance=importance,
            confidence=confidence,
            next_run=next_run,
            context=context,
            attention_key=attention_key,
            update_priority=update_priority,
            turn_id=turn_id,
        )
