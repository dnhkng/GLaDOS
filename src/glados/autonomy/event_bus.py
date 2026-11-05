import queue


class EventBus:
    """Hands autonomy events (slot updates, clock ticks) to the loop; the loop dispatches on their type."""

    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue()

    def publish(self, event: object) -> None:
        self._queue.put(event)

    def get(self, timeout: float | None = None) -> object:
        return self._queue.get(timeout=timeout)
