import threading
import time


class InteractionState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_user_ts: float | None = None
        self._last_assistant_ts: float | None = None

    def mark_user(self) -> None:
        with self._lock:
            self._last_user_ts = time.time()

    def mark_assistant(self) -> None:
        with self._lock:
            self._last_assistant_ts = time.time()

    def seconds_since_user(self) -> float | None:
        with self._lock:
            if self._last_user_ts is None:
                return None
            return time.time() - self._last_user_ts

    def seconds_since_assistant(self) -> float | None:
        with self._lock:
            if self._last_assistant_ts is None:
                return None
            return time.time() - self._last_assistant_ts
