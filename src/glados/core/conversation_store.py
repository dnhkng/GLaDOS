"""
Thread-safe conversation history store.

This module provides a ConversationStore class that encapsulates all synchronization
for the shared conversation history, eliminating race conditions from conditional
lock usage patterns.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
import uuid

from loguru import logger


@dataclass
class HistoryRecord:
    """Metadata stays outside messages sent to inference APIs."""

    id: str
    message: dict[str, Any]
    start_at: float
    end_at: float
    summary_level: int | None = None
    voice_turn_id: str | None = None


class ConversationStore:
    """
    Thread-safe conversation history store.

    All operations are atomic and protected by internal locking.
    Consumers should NOT hold references to internal lists or mutate
    returned snapshots if they need isolation.

    This replaces the previous pattern of sharing a raw list with a
    threading.Lock that was conditionally acquired.
    """

    def __init__(self, initial_messages: list[dict[str, Any]] | None = None, path: Path | None = None) -> None:
        """
        Initialize the conversation store.

        Args:
            initial_messages: Optional initial messages (e.g., personality preprompt).
                            These are copied, not referenced.
        """
        self._lock = threading.RLock()  # RLock allows nested acquisition if needed
        self._messages: list[dict[str, Any]] = list(initial_messages or [])
        self._metadata = [HistoryRecord(uuid.uuid4().hex, m, time.time(), time.time()) for m in self._messages]
        self._path = path.expanduser() if path else None
        self._version: int = 0  # For change detection / optimistic concurrency
        if self._path and self._path.exists():
            try:
                rows = json.loads(self._path.read_text())["records"]
                restored = [HistoryRecord(**row) for row in rows]
                if any(
                    r.message.get("role") not in {"user", "assistant", "tool"} or r.end_at < r.start_at
                    for r in restored
                ):
                    raise ValueError("Invalid history records")
                self._metadata.extend(restored)
                self._messages.extend(r.message for r in restored)
            except (OSError, ValueError, TypeError, KeyError):
                logger.warning("Could not restore conversation history; leaving the saved file intact")
                self._path = None

    def _persist(self, *, strict: bool = False) -> None:
        if not self._path:
            return
        payload = {"records": [asdict(r) for r in self._metadata if r.message.get("role") != "system"]}
        temporary = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=self._path.parent, delete=False) as stream:
                temporary = stream.name  # Created with mode 0600.
                json.dump(payload, stream)
            os.replace(temporary, self._path)
        except (OSError, TypeError, ValueError):
            if strict:
                raise
            logger.warning("Could not persist conversation history")
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def append(self, message: dict[str, Any], timestamp: float | None = None) -> int:
        """
        Append a single message to the conversation history.

        Args:
            message: The message dict to append (role, content, etc.)

        Returns:
            The new length of the conversation history.
        """
        with self._lock:
            self._messages.append(message)
            now = time.time() if timestamp is None else timestamp
            self._metadata.append(HistoryRecord(uuid.uuid4().hex, message, now, now))
            self._version += 1
            self._persist()
            return len(self._messages)

    def remove_voice_input(self, turn_id: str) -> None:
        """Withdraw the earlier user segment when an unanswered utterance continues."""
        with self._lock:
            kept = [r for r in self._metadata if not (
                r.voice_turn_id == turn_id and r.message.get("role") == "user")]
            if len(kept) != len(self._metadata):
                self._metadata = kept
                self._messages = [r.message for r in kept]
                self._version += 1
                self._persist()

    def append_voice_input(self, message: dict[str, Any], turn_id: str) -> None:
        """One history entry per complete utterance; metadata never enters model messages."""
        with self._lock:
            self.remove_voice_input(turn_id)
            now = time.time()
            self._messages.append(message)
            self._metadata.append(HistoryRecord(uuid.uuid4().hex, message, now, now, voice_turn_id=turn_id))
            self._version += 1
            self._persist()

    def append_multiple(self, messages: list[dict[str, Any]]) -> int:
        """
        Atomically append multiple messages to the conversation history.

        This is useful for operations that need to add several related messages
        (e.g., user message + interrupted assistant partial response).

        Args:
            messages: List of message dicts to append.

        Returns:
            The new length of the conversation history.
        """
        with self._lock:
            self._messages.extend(messages)
            now = time.time()
            self._metadata.extend(HistoryRecord(uuid.uuid4().hex, m, now, now) for m in messages)
            self._version += 1
            self._persist()
            return len(self._messages)

    def snapshot(self) -> list[dict[str, Any]]:
        """
        Return a shallow copy of all messages.

        The returned list is a new list object, but the message dicts
        inside are the same objects. This is safe for reading but callers
        should not mutate the individual message dicts.

        Returns:
            A shallow copy of the conversation history.
        """
        with self._lock:
            return list(self._messages)

    def deep_snapshot(self) -> list[dict[str, Any]]:
        """
        Return a deep copy of all messages for safe mutation.

        Use this when you need to modify messages without affecting
        the original store.

        Returns:
            A deep copy of the conversation history.
        """
        with self._lock:
            return deepcopy(self._messages)

    def replace_all(self, new_messages: list[dict[str, Any]]) -> None:
        """
        Atomically replace the entire conversation history.

        This is used by the compaction agent to swap in a compacted
        history without race conditions.

        Args:
            new_messages: The new message list to replace with (copied).
        """
        with self._lock:
            self._messages.clear()
            self._messages.extend(new_messages)
            old = {id(r.message): r for r in self._metadata}
            now = time.time()
            self._metadata = [old.get(id(m), HistoryRecord(uuid.uuid4().hex, m, now, now)) for m in new_messages]
            self._version += 1
            self._persist()

    def records(self) -> list[HistoryRecord]:
        """Isolated timestamped snapshot for a background compaction pass."""
        with self._lock:
            return deepcopy(self._metadata)

    def edit_summary(self, record_id: str, expected_revision: str, content: str | None) -> None:
        from .memory_records import revision
        with self._lock:
            record = next((r for r in self._metadata if r.id == record_id and r.summary_level is not None), None)
            if record is None or revision(str(record.message.get("content", ""))) != expected_revision:
                raise ValueError("Summary changed or no longer exists; refresh before editing")
            previous = deepcopy(self._metadata)
            if content is None:
                self._metadata = [r for r in self._metadata if r.id != record_id]
            else:
                record.message = {**record.message, "content": content}
            self._messages = [r.message for r in self._metadata]
            self._version += 1
            try:
                self._persist(strict=True)
            except (OSError, TypeError, ValueError):
                self._metadata = previous
                self._messages = [r.message for r in previous]
                self._version += 1
                raise

    def compact(self, records: list[HistoryRecord], content: str, level: int) -> bool:
        """Replace unchanged records atomically, preserving concurrent appends."""
        if not records:
            return False
        with self._lock:
            current = {r.id: r for r in self._metadata}
            if any(current.get(r.id) != r for r in records):
                return False
            if any(r.message.get("role") == "system" for r in records):
                return False
            selected = {r.id for r in records}
            replacement = HistoryRecord(
                uuid.uuid4().hex,
                {"role": "assistant", "content": content},
                min(r.start_at for r in records),
                max(r.end_at for r in records),
                level,
            )
            result = []
            inserted = False
            for record in self._metadata:
                if record.id in selected:
                    if not inserted:
                        result.append(replacement)
                        inserted = True
                else:
                    result.append(record)
            self._metadata = result
            self._messages = [r.message for r in result]
            self._version += 1
            self._persist()
            return True

    def modify_message(
        self,
        index: int,
        modifier: Any,
    ) -> bool:
        """
        Modify a message at a specific index atomically.

        Args:
            index: The index of the message to modify.
            modifier: Either a dict to update with, or a callable that
                     takes the message and returns the modified message.

        Returns:
            True if modification succeeded, False if index out of range.
        """
        with self._lock:
            if index < 0 or index >= len(self._messages):
                return False
            if callable(modifier):
                self._messages[index] = modifier(self._messages[index])
            else:
                self._messages[index].update(modifier)
            self._metadata[index].message = self._messages[index]
            self._version += 1
            self._persist()
            return True

    def __len__(self) -> int:
        """Return the number of messages in the store."""
        with self._lock:
            return len(self._messages)

    @property
    def version(self) -> int:
        """
        Current version number for change detection.

        Incremented on every modification. Can be used for optimistic
        concurrency checks or cache invalidation.
        """
        with self._lock:
            return self._version

    def iter_messages(self) -> list[dict[str, Any]]:
        """
        Return a snapshot for iteration.

        This is equivalent to snapshot() but named explicitly for
        iteration use cases.

        Returns:
            A shallow copy suitable for iteration.
        """
        return self.snapshot()
