"""
Subagent manager for GLaDOS autonomy system.

Manages the lifecycle of subagents: registration, start, stop, and monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

from loguru import logger

from .subagent import Subagent, SubagentConfig

if TYPE_CHECKING:
    from .slots import TaskSlotStore
    from ..observability import MindRegistry, ObservabilityBus


@dataclass
class SubagentStatus:
    """Status of a registered subagent."""

    agent_id: str
    title: str
    running: bool
    tick_count: int
    last_tick: float


class SubagentManager:
    """
    Manages the lifecycle of subagents.

    Provides methods to register, start, stop, and monitor subagents.
    All subagents share the same shutdown event for coordinated shutdown.
    """

    def __init__(
        self,
        slot_store: TaskSlotStore,
        mind_registry: MindRegistry | None = None,
        observability_bus: ObservabilityBus | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        self._slot_store = slot_store
        self._mind_registry = mind_registry
        self._observability_bus = observability_bus
        self._shutdown_event = shutdown_event or threading.Event()
        self._lock = threading.Lock()
        self._agents: dict[str, Subagent] = {}
        self._threads: dict[str, threading.Thread] = {}

    def register(self, subagent: Subagent) -> None:
        """
        Register a subagent with the manager.

        The subagent must be initialized but not started.
        """
        with self._lock:
            agent_id = subagent.agent_id
            if agent_id in self._agents:
                raise ValueError(f"Subagent {agent_id} is already registered")
            self._agents[agent_id] = subagent
            logger.info("Registered subagent: %s", agent_id)

    def create_and_register(
        self,
        subagent_class: type[Subagent],
        config: SubagentConfig,
    ) -> Subagent:
        """
        Create a subagent instance and register it.

        Args:
            subagent_class: The subagent class to instantiate.
            config: Configuration for the subagent.

        Returns:
            The created subagent instance.
        """
        subagent = subagent_class(
            config=config,
            slot_store=self._slot_store,
            mind_registry=self._mind_registry,
            observability_bus=self._observability_bus,
            shutdown_event=self._shutdown_event,
        )
        self.register(subagent)
        return subagent

    def start(self, agent_id: str) -> None:
        """Start a registered subagent."""
        with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Subagent {agent_id} is not registered")
            subagent = self._agents[agent_id]
            if subagent.is_running:
                logger.warning("Subagent %s is already running", agent_id)
                return
            thread = subagent.start()
            self._threads[agent_id] = thread

    def stop(self, agent_id: str, timeout: float = 5.0) -> None:
        """Stop a running subagent."""
        with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Subagent {agent_id} is not registered")
            subagent = self._agents[agent_id]

        subagent.stop(timeout=timeout)

    def start_all(self) -> None:
        """Start all registered subagents."""
        with self._lock:
            agent_ids = list(self._agents.keys())

        for agent_id in agent_ids:
            try:
                self.start(agent_id)
            except Exception as exc:
                logger.error("Failed to start subagent %s: %s", agent_id, exc)

    def stop_all(self, timeout: float = 5.0, global_timeout: bool = True) -> None:
        """
        Stop all running subagents.

        Args:
            timeout: Timeout per subagent (if global_timeout=False) or
                    total timeout for all subagents (if global_timeout=True).
            global_timeout: If True, timeout is the total time for all subagents.
                           If False, each subagent gets the full timeout.
        """
        import time

        with self._lock:
            agent_ids = list(self._agents.keys())

        if not agent_ids:
            return

        if global_timeout:
            per_agent_timeout = timeout / len(agent_ids)
            deadline = time.time() + timeout
        else:
            per_agent_timeout = timeout
            deadline = None

        for agent_id in agent_ids:
            # Adjust timeout based on remaining time
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    logger.warning("Global timeout reached, abandoning remaining subagents")
                    break
                current_timeout = min(per_agent_timeout, remaining)
            else:
                current_timeout = per_agent_timeout

            try:
                self.stop(agent_id, timeout=current_timeout)
            except Exception as exc:
                logger.error("Failed to stop subagent %s: %s", agent_id, exc)

    def list_agents(self) -> list[SubagentStatus]:
        """Get status of all registered subagents."""
        with self._lock:
            result = []
            for agent_id, subagent in self._agents.items():
                result.append(
                    SubagentStatus(
                        agent_id=agent_id,
                        title=subagent.title,
                        running=subagent.is_running,
                        tick_count=subagent._tick_count,
                        last_tick=subagent._last_tick,
                    )
                )
            return result

    def get(self, agent_id: str) -> Subagent | None:
        """Get a subagent by ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def unregister(self, agent_id: str, stop_if_running: bool = True) -> None:
        """
        Unregister a subagent.

        Args:
            agent_id: The ID of the subagent to unregister.
            stop_if_running: If True, stop the subagent if it's running.
        """
        with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Subagent {agent_id} is not registered")
            subagent = self._agents[agent_id]

        if stop_if_running and subagent.is_running:
            subagent.stop()

        with self._lock:
            del self._agents[agent_id]
            self._threads.pop(agent_id, None)

        logger.info("Unregistered subagent: %s", agent_id)

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Shutdown the manager and all subagents.

        Sets the shutdown event and waits for all subagents to stop.
        """
        logger.info("SubagentManager shutting down...")
        self._shutdown_event.set()
        self.stop_all(timeout=timeout)
        logger.info("SubagentManager shutdown complete.")
