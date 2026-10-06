"""
Generic thread-safe store with optional persistence.

Replaces: PreferencesStore, KnowledgeStore, TaskSlotStore, MindRegistry
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Generic, TypeVar

from loguru import logger

from .settings_files import read_settings, settings_source, write_settings

T = TypeVar("T")
Formatter = Callable[[dict[str, T]], str | None]


class Store(Generic[T]):
    """
    Thread-safe key-value store with optional YAML or legacy JSON persistence.

    Features:
    - Generic typing for values
    - Thread-safe via lock
    - Optional file persistence
    - Customizable prompt formatting via formatter function
    - Supports dataclass values (auto-serialized)

    Usage:
        # Simple key-value store
        prefs = Store[Any](path="preferences.yaml")
        prefs.set("theme", "dark")

        # Typed store with custom formatter
        def format_slots(data: dict[str, TaskSlot]) -> str:
            lines = ["[tasks]"]
            for slot in data.values():
                lines.append(f"- {slot.title}: {slot.status}")
            return "\\n".join(lines)

        slots = Store[TaskSlot](formatter=format_slots)
    """

    def __init__(
        self,
        path: Path | str | None = None,
        formatter: Formatter[T] | None = None,
        on_change: Callable[[str, T | None], None] | None = None,
    ) -> None:
        """
        Initialize the store.

        Args:
            path: Optional file path for settings persistence
            formatter: Function to format data as prompt string
            on_change: Callback invoked on set/delete with (key, new_value or None)
        """
        self._lock = threading.Lock()
        self._data: dict[str, T] = {}
        self._path = Path(path) if path else None
        self._formatter = formatter
        self._on_change = on_change

        if self._path and settings_source(self._path).exists():
            self._load()
        if self._path and self._path.suffix in {".yaml", ".yml"} and not self._path.exists():
            self._save()

    def _load(self) -> None:
        """Load data from disk."""
        if not self._path:
            return
        try:
            data = read_settings(self._path)
            if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
                raise ValueError("Preferences must be a mapping with string keys")
            self._data = data
            logger.debug("Store: Loaded {} entries from {}", len(self._data), self._path)
        except (ValueError, OSError) as e:
            logger.warning("Store: Failed to load from {}: {}", self._path, e)

    def _save(self) -> None:
        """Save data to disk."""
        if not self._path:
            return
        try:
            # Convert dataclasses to dicts for serialization
            serializable = {}
            for k, v in self._data.items():
                if is_dataclass(v) and not isinstance(v, type):
                    serializable[k] = asdict(v)
                else:
                    serializable[k] = v
            write_settings(self._path, serializable)
        except OSError as e:
            logger.warning("Store: Failed to save to {}: {}", self._path, e)

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get a value by key."""
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: T) -> None:
        """Set a value by key."""
        with self._lock:
            self._data[key] = value
            self._save()
        if self._on_change:
            self._on_change(key, value)

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._save()
                if self._on_change:
                    self._on_change(key, None)
                return True
            return False

    def update(self, key: str, **fields: Any) -> T | None:
        """
        Update specific fields of a value (if it's a dataclass or dict).
        Returns the updated value, or None if key doesn't exist.
        """
        with self._lock:
            existing = self._data.get(key)
            if existing is None:
                return None

            if is_dataclass(existing) and not isinstance(existing, type):
                # Create new dataclass with updated fields
                current = asdict(existing)
                current.update(fields)
                # Reconstruct - this requires the type to be available
                updated = type(existing)(**current)
                self._data[key] = updated
            elif isinstance(existing, dict):
                existing.update(fields)
                updated = existing
            else:
                return None

            self._save()
            if self._on_change:
                self._on_change(key, updated)
            return updated

    def all(self) -> dict[str, T]:
        """Get a copy of all data."""
        with self._lock:
            return dict(self._data)

    def values(self) -> list[T]:
        """Get all values as a list."""
        with self._lock:
            return list(self._data.values())

    def keys(self) -> list[str]:
        """Get all keys as a list."""
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> int:
        """Clear all data. Returns count of deleted items."""
        with self._lock:
            count = len(self._data)
            self._data.clear()
            self._save()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def as_prompt(self) -> str | None:
        """Format data for LLM context using the configured formatter."""
        if not self._formatter:
            return None
        with self._lock:
            if not self._data:
                return None
            return self._formatter(dict(self._data))

    def as_message(self) -> dict[str, str] | None:
        """Format data as a system message dict."""
        prompt = self.as_prompt()
        if prompt:
            return {"role": "system", "content": prompt}
        return None


# Common formatters

def format_preferences(data: dict[str, Any]) -> str | None:
    """Format preferences for LLM context."""
    if not data:
        return None
    lines = ["[preferences]"]
    for key, value in data.items():
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)
        lines.append(f"- {key}: {value_str}")
    return "\n".join(lines)


def format_knowledge(data: dict[str, Any]) -> str | None:
    """Format knowledge entries for LLM context."""
    if not data:
        return None
    lines = ["[knowledge]"]
    for entry_id, entry in data.items():
        if isinstance(entry, dict):
            text = entry.get("text", str(entry))
        else:
            text = str(entry)
        lines.append(f"- #{entry_id}: {text}")
    return "\n".join(lines)
