"""
Graceful shutdown orchestration for GLaDOS.

Coordinates shutdown of all components in the correct order to prevent
data loss from in-flight operations.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from loguru import logger


class ShutdownPriority(IntEnum):
    """
    Shutdown priority levels.

    Lower values are shut down first. Components are grouped by priority
    and each group is fully stopped before proceeding to the next.
    """

    INPUT = 1  # Input listeners (ASR, text) - stop accepting new work
    PROCESSING = 2  # LLM processors, tool executors - complete in-flight work
    OUTPUT = 3  # TTS, audio player - complete pending output
    BACKGROUND = 4  # Autonomy, vision - can safely abandon
    CLEANUP = 5  # Final cleanup operations


@dataclass
class ComponentEntry:
    """Registered component for shutdown coordination."""

    name: str
    thread: threading.Thread
    queue: queue.Queue[Any] | None = None
    priority: ShutdownPriority = ShutdownPriority.BACKGROUND
    drain_timeout: float = 5.0
    daemon: bool = True


@dataclass
class ShutdownResult:
    """Result of a shutdown operation."""

    component: str
    success: bool
    duration: float
    items_drained: int = 0
    error: str | None = None


@dataclass
class ShutdownOrchestrator:
    """
    Coordinates graceful shutdown of GLaDOS components.

    Shutdown proceeds in phases:
    1. Signal shutdown event to all components
    2. Stop input components (prevent new work)
    3. Drain processing queues
    4. Stop processing components
    5. Drain output queues
    6. Stop output components
    7. Join all threads with timeout
    8. Force cleanup if needed

    Thread daemon settings:
    - True: Can be killed without waiting (pure input, stateless)
    - False: Must be joined (has in-flight state to preserve)
    """

    shutdown_event: threading.Event = field(default_factory=threading.Event)
    global_timeout: float = 30.0
    phase_timeout: float = 10.0
    _components: dict[str, ComponentEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _results: list[ShutdownResult] = field(default_factory=list)

    def register(
        self,
        name: str,
        thread: threading.Thread,
        queue: queue.Queue[Any] | None = None,
        priority: ShutdownPriority = ShutdownPriority.BACKGROUND,
        drain_timeout: float = 5.0,
    ) -> None:
        """
        Register a component for coordinated shutdown.

        Args:
            name: Unique component name.
            thread: The component's thread.
            queue: Optional input queue to drain before stopping.
            priority: Shutdown priority (lower = earlier).
            drain_timeout: Max time to wait for queue draining.
        """
        with self._lock:
            if name in self._components:
                logger.warning("Component '{}' already registered, replacing", name)
            self._components[name] = ComponentEntry(
                name=name,
                thread=thread,
                queue=queue,
                priority=priority,
                drain_timeout=drain_timeout,
                daemon=thread.daemon,
            )
            logger.debug("Registered component '{}' with priority {}", name, priority.name)

    def unregister(self, name: str) -> None:
        """Unregister a component."""
        with self._lock:
            if name in self._components:
                del self._components[name]
                logger.debug("Unregistered component '{}'", name)

    def initiate_shutdown(self) -> list[ShutdownResult]:
        """
        Initiate graceful shutdown of all registered components.

        Returns:
            List of shutdown results for each component.
        """
        start_time = time.time()
        logger.info("Initiating graceful shutdown...")

        # Phase 1: Signal global shutdown
        self.shutdown_event.set()
        logger.debug("Shutdown event signaled")

        # Get components sorted by priority
        with self._lock:
            components = list(self._components.values())

        # Group components by priority
        priority_groups: dict[ShutdownPriority, list[ComponentEntry]] = {}
        for component in components:
            if component.priority not in priority_groups:
                priority_groups[component.priority] = []
            priority_groups[component.priority].append(component)

        # Process each priority group in order
        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]
            elapsed = time.time() - start_time
            if elapsed >= self.global_timeout:
                logger.warning(
                    "Global timeout reached ({:.1f}s), abandoning remaining components",
                    elapsed,
                )
                break

            logger.info(
                "Shutting down {} components at priority {}",
                len(group),
                priority.name,
            )

            # Phase 2: Drain queues for this priority group
            for component in group:
                if component.queue is not None:
                    drained = self._drain_queue(component)
                    self._results.append(
                        ShutdownResult(
                            component=f"{component.name}_queue",
                            success=True,
                            duration=0,
                            items_drained=drained,
                        )
                    )

            # Phase 3: Join threads for this priority group
            remaining_timeout = min(
                self.phase_timeout,
                self.global_timeout - (time.time() - start_time),
            )
            self._join_group(group, remaining_timeout)

        # Final summary
        total_time = time.time() - start_time
        successful = sum(1 for r in self._results if r.success)
        logger.info(
            "Shutdown completed in {:.2f}s ({}/{} components successful)",
            total_time,
            successful,
            len(self._results),
        )

        return self._results

    def _drain_queue(self, component: ComponentEntry) -> int:
        """
        Drain a component's queue before shutdown.

        Args:
            component: The component entry with queue to drain.

        Returns:
            Number of items drained.
        """
        if component.queue is None:
            return 0

        drained = 0
        deadline = time.time() + component.drain_timeout

        while time.time() < deadline:
            try:
                component.queue.get_nowait()
                drained += 1
            except queue.Empty:
                break

        if drained > 0:
            logger.debug(
                "Drained {} items from {} queue",
                drained,
                component.name,
            )

        return drained

    def _join_group(
        self,
        group: list[ComponentEntry],
        timeout: float,
    ) -> None:
        """
        Join all threads in a priority group.

        Args:
            group: List of components to join.
            timeout: Maximum time to wait for all threads.
        """
        per_thread_timeout = timeout / max(len(group), 1)

        for component in group:
            start = time.time()
            result = self._join_thread(component, per_thread_timeout)
            self._results.append(result)

            # Adjust remaining timeout
            elapsed = time.time() - start
            per_thread_timeout = max(0.1, per_thread_timeout - elapsed)

    def _join_thread(
        self,
        component: ComponentEntry,
        timeout: float,
    ) -> ShutdownResult:
        """
        Join a single thread with timeout.

        Args:
            component: The component to join.
            timeout: Maximum time to wait.

        Returns:
            ShutdownResult indicating success/failure.
        """
        start = time.time()
        thread = component.thread

        if not thread.is_alive():
            return ShutdownResult(
                component=component.name,
                success=True,
                duration=0,
            )

        try:
            thread.join(timeout=timeout)
            duration = time.time() - start

            if thread.is_alive():
                # Thread didn't stop in time
                logger.warning(
                    "Component '{}' did not stop within {:.1f}s (daemon={})",
                    component.name,
                    timeout,
                    component.daemon,
                )
                return ShutdownResult(
                    component=component.name,
                    success=False,
                    duration=duration,
                    error="Timeout",
                )

            logger.debug("Component '{}' stopped in {:.2f}s", component.name, duration)
            return ShutdownResult(
                component=component.name,
                success=True,
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start
            logger.error("Error stopping component '{}': {}", component.name, e)
            return ShutdownResult(
                component=component.name,
                success=False,
                duration=duration,
                error=str(e),
            )

    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self.shutdown_event.is_set()

    def get_results(self) -> list[ShutdownResult]:
        """Get the results from the last shutdown."""
        return list(self._results)
