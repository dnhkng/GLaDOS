from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Any

from .events import ObservabilityEvent


class ObservabilityBus:
    """Thread-safe, multi-subscriber event bus with bounded history.

    Producers call :meth:`emit` / :meth:`publish`. Consumers either:

    * poll the built-in single-consumer FIFO via :meth:`drain` (used by the
      TUI's ``ObservabilityScreen``), or
    * call :meth:`subscribe` for a private queue when multiple independent
      consumers must share the bus without stealing events from each other
      (e.g. several webapp SSE streams).
    """

    def __init__(self, max_history: int = 500, subscriber_max: int = 100) -> None:
        if max_history < 1 or subscriber_max < 1:
            raise ValueError("Observability limits must be positive")
        self._queue: queue.Queue[ObservabilityEvent] = queue.Queue(maxsize=max_history)
        self._lock = threading.Lock()
        self._history: deque[ObservabilityEvent] = deque(maxlen=max_history)
        # Per-subscriber backlogs. Bounded so a slow consumer never blocks the
        # bus; turned over (oldest dropped) if it falls too far behind.
        self._subscriber_max = subscriber_max
        self._subscribers: list[queue.Queue[ObservabilityEvent]] = []

    def emit(
        self,
        source: str,
        kind: str,
        message: str,
        level: str = "info",
        meta: dict[str, Any] | None = None,
    ) -> ObservabilityEvent:
        event = ObservabilityEvent(
            timestamp=time.time(),
            source=source,
            kind=kind,
            message=message,
            level=level,
            meta=meta or {},
        )
        self.publish(event)
        return event

    def publish(self, event: ObservabilityEvent) -> None:
        """Append to history and fan out to every live subscriber."""
        with self._lock:
            self._history.append(event)
            self._enqueue(self._queue, event)
            for sub in self._subscribers:
                self._enqueue(sub, event)

    @staticmethod
    def _enqueue(target: queue.Queue[ObservabilityEvent], event: ObservabilityEvent) -> None:
        # Producers hold the bus lock; consumers may drain concurrently.
        try:
            target.put_nowait(event)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            target.put_nowait(event)

    def subscribe(self) -> queue.Queue[ObservabilityEvent]:
        """Register a private subscription queue for this consumer.

        The returned queue receives every future event (bounded; oldest dropped
        if the consumer lags). Call :meth:`unsubscribe` to release it.
        """
        sub: queue.Queue[ObservabilityEvent] = queue.Queue(maxsize=self._subscriber_max)
        with self._lock:
            self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: queue.Queue[Any]) -> None:
        """Remove a previously created subscription queue."""
        with self._lock:
            try:
                self._subscribers.remove(sub)
            except ValueError:
                pass

    def drain(self, max_items: int = 100) -> list[ObservabilityEvent]:
        events: list[ObservabilityEvent] = []
        for _ in range(max_items):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def snapshot(self, limit: int | None = None) -> list[ObservabilityEvent]:
        with self._lock:
            events = list(self._history)
        if limit is None or limit <= 0:
            return events
        return events[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
