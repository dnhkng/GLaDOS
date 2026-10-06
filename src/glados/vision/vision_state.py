from __future__ import annotations

import threading
import time
from typing import Any


class VisionState:
    """Thread-safe store for the latest vision description."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._description: str | None = None
        self._captured_at: float | None = None

    def update(self, description: str, captured_at: float | None = None) -> None:
        """Update the latest vision description."""
        with self._lock:
            self._description = description
            self._captured_at = time.time() if captured_at is None else captured_at

    def clear(self) -> None:
        with self._lock:
            self._description = None
            self._captured_at = None

    def snapshot(self) -> str | None:
        """Return the latest vision description, if available."""
        with self._lock:
            return self._description

    def as_message(self) -> dict[str, Any] | None:
        """Return the vision context as a system message or None if empty."""
        with self._lock:
            description, captured_at = self._description, self._captured_at
        if not description:
            return None
        age = max(0, time.time() - captured_at) if captured_at is not None else 0
        return {"role": "system", "content": f"[vision] Camera observation, {age:.1f} seconds old.\n{description}"}
