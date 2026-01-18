"""Tests for the ShutdownOrchestrator."""

import queue
import threading
import time

import pytest

from glados.core.shutdown import (
    ComponentEntry,
    ShutdownOrchestrator,
    ShutdownPriority,
    ShutdownResult,
)


class TestShutdownPriority:
    """Tests for ShutdownPriority enum."""

    def test_priority_ordering(self) -> None:
        """Test that priorities are ordered correctly."""
        assert ShutdownPriority.INPUT < ShutdownPriority.PROCESSING
        assert ShutdownPriority.PROCESSING < ShutdownPriority.OUTPUT
        assert ShutdownPriority.OUTPUT < ShutdownPriority.BACKGROUND
        assert ShutdownPriority.BACKGROUND < ShutdownPriority.CLEANUP


class TestComponentEntry:
    """Tests for ComponentEntry dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        thread = threading.Thread(target=lambda: None)
        entry = ComponentEntry(name="test", thread=thread)
        assert entry.queue is None
        assert entry.priority == ShutdownPriority.BACKGROUND
        assert entry.drain_timeout == 5.0
        assert entry.daemon is True


class TestShutdownOrchestrator:
    """Tests for ShutdownOrchestrator."""

    def test_register_component(self) -> None:
        """Test registering a component."""
        orchestrator = ShutdownOrchestrator()
        thread = threading.Thread(target=lambda: None, daemon=True)
        orchestrator.register("test", thread, priority=ShutdownPriority.PROCESSING)
        # No exception means success

    def test_register_duplicate_warns(self) -> None:
        """Test that registering duplicate component warns but succeeds."""
        orchestrator = ShutdownOrchestrator()
        thread1 = threading.Thread(target=lambda: None, daemon=True)
        thread2 = threading.Thread(target=lambda: None, daemon=True)
        orchestrator.register("test", thread1)
        # Should warn but not raise
        orchestrator.register("test", thread2)

    def test_unregister_component(self) -> None:
        """Test unregistering a component."""
        orchestrator = ShutdownOrchestrator()
        thread = threading.Thread(target=lambda: None, daemon=True)
        orchestrator.register("test", thread)
        orchestrator.unregister("test")
        # Unregistering again should be a no-op
        orchestrator.unregister("test")

    def test_is_shutting_down(self) -> None:
        """Test shutdown state tracking."""
        orchestrator = ShutdownOrchestrator()
        assert orchestrator.is_shutting_down() is False
        orchestrator.shutdown_event.set()
        assert orchestrator.is_shutting_down() is True

    def test_shutdown_sets_event(self) -> None:
        """Test that shutdown sets the shutdown event."""
        orchestrator = ShutdownOrchestrator()
        assert not orchestrator.shutdown_event.is_set()
        orchestrator.initiate_shutdown()
        assert orchestrator.shutdown_event.is_set()

    def test_shutdown_stops_running_threads(self) -> None:
        """Test that shutdown stops running threads."""
        orchestrator = ShutdownOrchestrator(global_timeout=5.0)
        stopped = threading.Event()

        def worker() -> None:
            while not orchestrator.shutdown_event.is_set():
                time.sleep(0.01)
            stopped.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        orchestrator.register("worker", thread, priority=ShutdownPriority.PROCESSING)

        results = orchestrator.initiate_shutdown()

        # Thread should have stopped
        assert stopped.is_set()
        # Should have results
        assert len(results) > 0

    def test_shutdown_drains_queues(self) -> None:
        """Test that shutdown drains component queues."""
        orchestrator = ShutdownOrchestrator(global_timeout=5.0)
        q: queue.Queue[str] = queue.Queue()
        q.put("item1")
        q.put("item2")
        q.put("item3")

        def worker() -> None:
            while not orchestrator.shutdown_event.is_set():
                time.sleep(0.01)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        orchestrator.register(
            "worker",
            thread,
            queue=q,
            priority=ShutdownPriority.PROCESSING,
        )

        results = orchestrator.initiate_shutdown()

        # Queue should be drained
        drained_results = [r for r in results if r.component == "worker_queue"]
        assert len(drained_results) == 1
        assert drained_results[0].items_drained == 3

    def test_shutdown_respects_priority_order(self) -> None:
        """Test that components are shut down in priority order."""
        orchestrator = ShutdownOrchestrator(global_timeout=10.0)

        # Track when each component's join was called (not when thread exits)
        join_order: list[str] = []
        lock = threading.Lock()

        def make_worker(name: str) -> callable:
            def worker() -> None:
                while not orchestrator.shutdown_event.is_set():
                    time.sleep(0.01)
                # Small delay to ensure join order is recorded correctly
                time.sleep(0.01)
            return worker

        # Register in reverse priority order
        for name, priority in [
            ("background", ShutdownPriority.BACKGROUND),
            ("processing", ShutdownPriority.PROCESSING),
            ("input", ShutdownPriority.INPUT),
        ]:
            thread = threading.Thread(target=make_worker(name), daemon=True)
            thread.start()
            orchestrator.register(name, thread, priority=priority)

        results = orchestrator.initiate_shutdown()

        # Get the order from results (they are added in processing order)
        result_order = [r.component for r in results if not r.component.endswith("_queue")]

        # Verify priority groups are processed in order by checking results
        # INPUT (1) < PROCESSING (2) < BACKGROUND (4)
        input_idx = result_order.index("input")
        processing_idx = result_order.index("processing")
        background_idx = result_order.index("background")

        assert input_idx < processing_idx, f"input ({input_idx}) should be before processing ({processing_idx})"
        assert processing_idx < background_idx, f"processing ({processing_idx}) should be before background ({background_idx})"

    def test_shutdown_timeout_handling(self) -> None:
        """Test that shutdown handles stuck threads gracefully."""
        orchestrator = ShutdownOrchestrator(
            global_timeout=1.0,
            phase_timeout=0.5,
        )

        def stuck_worker() -> None:
            # Never exits
            while True:
                time.sleep(0.1)

        thread = threading.Thread(target=stuck_worker, daemon=True)
        thread.start()
        orchestrator.register("stuck", thread, priority=ShutdownPriority.PROCESSING)

        start = time.time()
        results = orchestrator.initiate_shutdown()
        elapsed = time.time() - start

        # Should not hang forever
        assert elapsed < 5.0

        # Should have a failed result
        failed = [r for r in results if not r.success]
        assert len(failed) > 0

    def test_get_results(self) -> None:
        """Test that results can be retrieved after shutdown."""
        orchestrator = ShutdownOrchestrator()

        def worker() -> None:
            while not orchestrator.shutdown_event.is_set():
                time.sleep(0.01)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        orchestrator.register("worker", thread)

        orchestrator.initiate_shutdown()
        results = orchestrator.get_results()

        assert len(results) > 0
        assert all(isinstance(r, ShutdownResult) for r in results)

    def test_shutdown_with_no_components(self) -> None:
        """Test shutdown with no registered components."""
        orchestrator = ShutdownOrchestrator()
        results = orchestrator.initiate_shutdown()
        assert results == []
        assert orchestrator.is_shutting_down()


class TestShutdownResult:
    """Tests for ShutdownResult dataclass."""

    def test_successful_result(self) -> None:
        """Test creating a successful result."""
        result = ShutdownResult(
            component="test",
            success=True,
            duration=1.5,
        )
        assert result.component == "test"
        assert result.success is True
        assert result.duration == 1.5
        assert result.items_drained == 0
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test creating a failed result."""
        result = ShutdownResult(
            component="stuck",
            success=False,
            duration=5.0,
            error="Timeout",
        )
        assert result.success is False
        assert result.error == "Timeout"
