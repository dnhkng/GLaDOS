import queue
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()

    def publish(self, event: Any) -> None:
        self._queue.put(event)

    def get(self, timeout: float | None = None) -> Any:
        return self._queue.get(timeout=timeout)
