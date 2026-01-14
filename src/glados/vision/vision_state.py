from __future__ import annotations

import threading
from typing import Any


class VisionState:
    """Thread-safe store for the latest vision description."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._description: str | None = None

    def update(self, description: str) -> None:
        """Update the latest vision description."""
        with self._lock:
            self._description = description

    def snapshot(self) -> str | None:
        """Return the latest vision description, if available."""
        with self._lock:
            return self._description

    def as_message(self) -> dict[str, Any] | None:
        """Return the vision context as a system message or None if empty."""
        description = self.snapshot()
        if not description:
            return None
        return {"role": "system", "content": f"[vision] {description}"}
