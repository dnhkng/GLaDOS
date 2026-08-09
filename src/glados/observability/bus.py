"""Thread-safe observability event distribution."""

from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Any

from .events import ObservabilityEvent


class ObservabilityBus:
    """Thread-safe event bus with bounded history and independent consumers."""

    def __init__(self, max_history: int = 500, subscriber_max: int = 100) -> None:
        """Initialize bounded history and per-consumer queue capacity."""
        self._queue: queue.Queue[ObservabilityEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._history: deque[ObservabilityEvent] = deque(maxlen=max_history)
        self._subscriber_max = max(1, subscriber_max)
        self._subscribers: list[queue.Queue[ObservabilityEvent]] = []

    def emit(
        self,
        source: str,
        kind: str,
        message: str,
        level: str = "info",
        meta: dict[str, Any] | None = None,
    ) -> ObservabilityEvent:
        """Construct, publish, and return an observability event."""
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
        """Publish an event to history, the legacy queue, and all subscribers."""
        with self._lock:
            self._history.append(event)
            for subscriber in self._subscribers:
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    # A slow subscriber must not block producers. Keep its
                    # newest events by evicting one oldest item.
                    try:
                        subscriber.get_nowait()
                        subscriber.put_nowait(event)
                    except (queue.Empty, queue.Full):
                        pass
            self._queue.put(event)

    def subscribe(self) -> queue.Queue[ObservabilityEvent]:
        """Return a bounded queue which receives every future event."""
        subscriber: queue.Queue[ObservabilityEvent] = queue.Queue(maxsize=self._subscriber_max)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[ObservabilityEvent]) -> None:
        """Remove a queue previously returned by :meth:`subscribe`."""
        with self._lock:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

    def drain(self, max_items: int = 100) -> list[ObservabilityEvent]:
        """Remove up to ``max_items`` from the legacy single-consumer queue."""
        events: list[ObservabilityEvent] = []
        for _ in range(max_items):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def snapshot(self, limit: int | None = None) -> list[ObservabilityEvent]:
        """Return a stable copy of recent event history."""
        with self._lock:
            events = list(self._history)
        if limit is None or limit <= 0:
            return events
        return events[-limit:]

    def clear(self) -> None:
        """Clear history and the legacy queue."""
        with self._lock:
            self._history.clear()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
