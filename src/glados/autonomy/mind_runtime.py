"""Execution context supplied to a mind, independent of model inference."""

from __future__ import annotations

from dataclasses import asdict
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mind_scheduler import MindScheduler
    from .subagent import Subagent, SubagentOutput


class MindRuntime:
    def __init__(self, mind: Subagent, shutdown: threading.Event) -> None:
        self.mind = mind
        self.shutdown = shutdown
        self.scheduler: MindScheduler | None = None
        self.paused = False
        self.enabled = False
        self.manual = False
        self.executing = False
        self.generation = 0
        self.run_generation = 0

    def cancelled(self) -> bool:
        return self.shutdown.is_set() or (self.executing and self.run_generation != self.generation)

    def publish(self, update: SubagentOutput | None) -> None:
        """Progress and final updates use the mind's canonical slot publisher."""
        if update is not None and not self.cancelled():
            fields = asdict(update)
            fields.pop("raw")
            self.mind.write_slot(**fields)

    def request_run(self) -> None:
        """Coalesce new domain input into one pending run."""
        if self.scheduler:
            self.scheduler.trigger(self.mind.agent_id, manual=False)

    def set_paused(self, paused: bool) -> None:
        if self.scheduler:
            self.scheduler.pause(self.mind.agent_id, paused)
        elif self.paused != paused:
            self.paused = paused
            self.generation += 1
            self.mind.on_pause(paused)
