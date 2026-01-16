from __future__ import annotations

import threading


class InFlightCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0

    def increment(self) -> None:
        with self._lock:
            self._count += 1

    def decrement(self) -> None:
        with self._lock:
            if self._count > 0:
                self._count -= 1

    def value(self) -> int:
        with self._lock:
            return self._count
