from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Any

from .events import ObservabilityEvent


class ObservabilityBus:
    def __init__(self, max_history: int = 500) -> None:
        self._queue: queue.Queue[ObservabilityEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._history: deque[ObservabilityEvent] = deque(maxlen=max_history)

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
        with self._lock:
            self._history.append(event)
        self._queue.put(event)

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
