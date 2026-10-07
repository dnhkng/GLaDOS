"""
Unified context builder for LLM requests.

Replaces the scattered context injection in _build_messages().
All context sources register with the builder, which produces
the final system messages for LLM requests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
from typing import Any

ContextSource = Callable[[], str | None]


@dataclass
class ContextEntry:
    """A registered context source."""
    name: str
    source: ContextSource
    priority: int = 0  # Higher = earlier within the same stability class.
    volatile: bool = False  # Live values go after stable instructions and history.


class ContextBuilder:
    """
    Builds LLM context from registered sources.

    Sources are functions that return:
    - A string to inject as a system message
    - None to skip (no content to inject)

    Usage:
        context = ContextBuilder()
        context.register("preferences", preferences_store.as_prompt, priority=10)
        context.register("emotion", emotion_state.to_prompt, priority=5, volatile=True)
        context.register("vision", vision_state.as_message, priority=0, volatile=True)

        # Build context for LLM request
        messages = [entry["message"] for entry in context.build_system_entries()]
        # Returns: [{"role": "system", "content": "..."}, ...]
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: list[ContextEntry] = []

    def register(
        self,
        name: str,
        source: ContextSource,
        priority: int = 0,
        *,
        volatile: bool = False,
    ) -> None:
        """
        Register a context source.

        Args:
            name: Identifier for this source (for debugging)
            source: Callable that returns prompt string or None
            priority: Higher values appear earlier within the same stability class
            volatile: Changes each request; keep out of the reusable prompt prefix
        """
        with self._lock:
            # Remove existing source with same name
            self._sources = [s for s in self._sources if s.name != name]
            self._sources.append(ContextEntry(name=name, source=source, priority=priority, volatile=volatile))
            # Stable configuration always precedes live data, even when a live
            # source has high priority for its position near the current input.
            self._sources.sort(key=lambda x: (x.volatile, -x.priority))

    def unregister(self, name: str) -> bool:
        """Remove a context source. Returns True if it existed."""
        with self._lock:
            before = len(self._sources)
            self._sources = [s for s in self._sources if s.name != name]
            return len(self._sources) < before


    def build_system_entries(self) -> list[dict[str, Any]]:
        """Resolve sources once, retaining provenance outside the model messages."""
        with self._lock:
            sources = list(self._sources)

        messages = []
        for entry in sources:
            try:
                content = entry.source()
                if content:
                    messages.append({"source": entry.name, "volatile": entry.volatile,
                                     "message": {"role": "system", "content": content}})
            except Exception:
                # Skip failed sources silently
                pass
        return messages


    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)
