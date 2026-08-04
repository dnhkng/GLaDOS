"""
Per-subagent persistent memory using jsonlines.

Each subagent gets its own memory file storing entries as JSON lines.
Entries can be marked as "shown" when mentioned to the user, allowing
subagents to track what the user has already heard about.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Iterator

from loguru import logger

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt

    @contextmanager
    def _file_lock(f: IO, exclusive: bool = False) -> Iterator[None]:
        """Acquire a file lock on Windows."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)
            yield
        finally:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
else:
    import fcntl

    @contextmanager
    def _file_lock(f: IO, exclusive: bool = False) -> Iterator[None]:
        """Acquire a file lock on Unix."""
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@dataclass
class MemoryEntry:
    """A single memory entry."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    shown_at: float | None = None

    def is_shown(self) -> bool:
        return self.shown_at is not None


class SubagentMemory:
    """
    Fixed-size persistent memory for a subagent.

    Stores entries as jsonlines, evicts oldest when full.
    Thread-safe via file locking.
    """

    def __init__(
        self,
        agent_id: str,
        max_entries: int = 100,
        storage_dir: Path | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.max_entries = max_entries

        if storage_dir is None:
            storage_dir = Path.home() / ".glados" / "memory"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = self.storage_dir / f"{agent_id}.jsonl"
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    def get(self, key: str) -> MemoryEntry | None:
        """Get an entry by key."""
        return self._entries.get(key)

    def set(self, key: str, value: Any) -> MemoryEntry:
        """Store a value. Overwrites if key exists."""
        if key in self._entries:
            entry = self._entries[key]
            entry.value = value
        else:
            entry = MemoryEntry(key=key, value=value)
            self._entries[key] = entry

        self._prune_if_needed()
        self._save()
        return entry

    def mark_shown(self, key: str) -> bool:
        """Mark an entry as shown to the user. Returns True if found."""
        entry = self._entries.get(key)
        if entry is None:
            return False

        entry.shown_at = time.time()
        self._save()
        return True

    def list_unshown(self) -> list[MemoryEntry]:
        """Get all entries not yet shown to the user, oldest first."""
        unshown = [e for e in self._entries.values() if not e.is_shown()]
        return sorted(unshown, key=lambda e: e.created_at)

    def list_all(self) -> list[MemoryEntry]:
        """Get all entries, oldest first."""
        return sorted(self._entries.values(), key=lambda e: e.created_at)

    def delete(self, key: str) -> bool:
        """Delete an entry. Returns True if it existed."""
        if key not in self._entries:
            return False

        del self._entries[key]
        self._save()
        return True

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._save()

    def _prune_if_needed(self) -> None:
        """Remove oldest entries if over capacity."""
        if len(self._entries) <= self.max_entries:
            return

        # Sort by created_at, remove oldest
        by_age = sorted(self._entries.values(), key=lambda e: e.created_at)
        to_remove = len(self._entries) - self.max_entries

        for entry in by_age[:to_remove]:
            del self._entries[entry.key]
            logger.debug("Memory %s: evicted %s", self.agent_id, entry.key)

    def _load(self) -> None:
        """Load entries from disk."""
        if not self.file_path.exists():
            return

        try:
            with open(self.file_path, "r") as f:
                with _file_lock(f, exclusive=False):
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        entry = MemoryEntry(
                            key=data["key"],
                            value=data["value"],
                            created_at=data.get("created_at", time.time()),
                            shown_at=data.get("shown_at"),
                        )
                        self._entries[entry.key] = entry
        except Exception as exc:
            logger.warning("Failed to load memory for %s: %s", self.agent_id, exc)

    def _save(self) -> None:
        """Save entries to disk."""
        try:
            with open(self.file_path, "w") as f:
                with _file_lock(f, exclusive=True):
                    for entry in self._entries.values():
                        line = json.dumps(asdict(entry), ensure_ascii=False)
                        f.write(line + "\n")
        except Exception as exc:
            logger.warning("Failed to save memory for %s: %s", self.agent_id, exc)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries
