"""
Unified context builder for LLM requests.

Replaces the scattered context injection in _build_messages().
All context sources register with the builder, which produces
the final system messages for LLM requests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

ContextSource = Callable[[], str | None]


@dataclass
class ContextEntry:
    """A registered context source."""
    name: str
    source: ContextSource
    priority: int = 0  # Higher = earlier in context


class ContextBuilder:
    """
    Builds LLM context from registered sources.

    Sources are functions that return:
    - A string to inject as a system message
    - None to skip (no content to inject)

    Usage:
        context = ContextBuilder()
        context.register("preferences", preferences_store.as_prompt, priority=10)
        context.register("emotion", emotion_state.to_prompt, priority=5)
        context.register("vision", vision_state.as_message, priority=0)

        # Build context for LLM request
        messages = context.build_system_messages()
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
    ) -> None:
        """
        Register a context source.

        Args:
            name: Identifier for this source (for debugging)
            source: Callable that returns prompt string or None
            priority: Higher values appear earlier in context
        """
        with self._lock:
            # Remove existing source with same name
            self._sources = [s for s in self._sources if s.name != name]
            self._sources.append(ContextEntry(name=name, source=source, priority=priority))
            # Sort by priority (descending)
            self._sources.sort(key=lambda x: x.priority, reverse=True)

    def unregister(self, name: str) -> bool:
        """Remove a context source. Returns True if it existed."""
        with self._lock:
            before = len(self._sources)
            self._sources = [s for s in self._sources if s.name != name]
            return len(self._sources) < before

    def build_system_messages(self) -> list[dict[str, str]]:
        """
        Build system messages from all registered sources.

        Returns list of {"role": "system", "content": "..."} dicts.
        Sources returning None are skipped.
        """
        with self._lock:
            sources = list(self._sources)

        messages = []
        for entry in sources:
            try:
                content = entry.source()
                if content:
                    messages.append({"role": "system", "content": content})
            except Exception:
                # Skip failed sources silently
                pass
        return messages

    def build_combined_prompt(self, separator: str = "\n\n") -> str | None:
        """
        Build a single combined prompt from all sources.

        Returns None if no sources have content.
        """
        with self._lock:
            sources = list(self._sources)

        parts = []
        for entry in sources:
            try:
                content = entry.source()
                if content:
                    parts.append(content)
            except Exception:
                pass

        return separator.join(parts) if parts else None

    def list_sources(self) -> list[str]:
        """Get names of all registered sources."""
        with self._lock:
            return [s.name for s in self._sources]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)
