"""Tests for the thread-safe ConversationStore."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from glados.core.conversation_store import ConversationStore


class TestConversationStore:
    """Unit tests for ConversationStore."""

    def test_init_empty(self) -> None:
        """Test initialization with no messages."""
        store = ConversationStore()
        assert len(store) == 0
        assert store.snapshot() == []
        assert store.version == 0

    def test_init_with_messages(self) -> None:
        """Test initialization with initial messages."""
        initial = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        store = ConversationStore(initial_messages=initial)
        assert len(store) == 2
        assert store.snapshot() == initial

    def test_append_single_message(self) -> None:
        """Test appending a single message."""
        store = ConversationStore()
        length = store.append({"role": "user", "content": "Hello"})
        assert length == 1
        assert len(store) == 1
        assert store.version == 1

    def test_append_multiple_messages(self) -> None:
        """Test atomically appending multiple messages."""
        store = ConversationStore()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        length = store.append_multiple(messages)
        assert length == 2
        assert len(store) == 2
        assert store.version == 1  # Single version increment

    def test_snapshot_returns_copy(self) -> None:
        """Test that snapshot returns a copy, not a reference."""
        store = ConversationStore([{"role": "user", "content": "Hello"}])
        snapshot1 = store.snapshot()
        snapshot2 = store.snapshot()

        # Should be equal but not the same object
        assert snapshot1 == snapshot2
        assert snapshot1 is not snapshot2

        # Modifying snapshot should not affect store
        snapshot1.append({"role": "assistant", "content": "Hi"})
        assert len(store) == 1

    def test_deep_snapshot(self) -> None:
        """Test deep snapshot returns fully independent copy."""
        store = ConversationStore([{"role": "user", "content": "Hello"}])
        deep = store.deep_snapshot()

        # Modifying message in deep snapshot should not affect store
        deep[0]["content"] = "Modified"
        assert store.snapshot()[0]["content"] == "Hello"

    def test_replace_all(self) -> None:
        """Test atomic replacement of all messages."""
        store = ConversationStore([{"role": "user", "content": "Old"}])
        new_messages = [
            {"role": "system", "content": "New system"},
            {"role": "user", "content": "New user"},
        ]
        store.replace_all(new_messages)
        assert len(store) == 2
        assert store.snapshot() == new_messages

    def test_modify_message_dict(self) -> None:
        """Test modifying a message with a dict."""
        store = ConversationStore([{"role": "user", "content": "Hello"}])
        result = store.modify_message(0, {"content": "Modified"})
        assert result is True
        assert store.snapshot()[0]["content"] == "Modified"
        assert store.snapshot()[0]["role"] == "user"

    def test_modify_message_callable(self) -> None:
        """Test modifying a message with a callable."""
        store = ConversationStore([{"role": "user", "content": "hello"}])

        def uppercase(msg: dict) -> dict:
            return {**msg, "content": msg["content"].upper()}

        result = store.modify_message(0, uppercase)
        assert result is True
        assert store.snapshot()[0]["content"] == "HELLO"

    def test_modify_message_invalid_index(self) -> None:
        """Test that modifying an invalid index returns False."""
        store = ConversationStore([{"role": "user", "content": "Hello"}])
        assert store.modify_message(-1, {}) is False
        assert store.modify_message(5, {}) is False

    def test_version_tracking(self) -> None:
        """Test that version increments on modifications."""
        store = ConversationStore()
        assert store.version == 0

        store.append({"role": "user", "content": "1"})
        assert store.version == 1

        store.append_multiple([{"role": "user", "content": "2"}])
        assert store.version == 2

        store.replace_all([])
        assert store.version == 3

        store.modify_message(0, {})  # Invalid, no change
        assert store.version == 3


class TestConversationStoreThreadSafety:
    """Thread-safety stress tests for ConversationStore."""

    def test_concurrent_appends(self) -> None:
        """Test that concurrent appends don't lose data."""
        store = ConversationStore()
        num_threads = 10
        messages_per_thread = 100

        def append_messages(thread_id: int) -> None:
            for i in range(messages_per_thread):
                store.append({"role": "user", "content": f"T{thread_id}-M{i}"})

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(append_messages, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        # All messages should be present
        assert len(store) == num_threads * messages_per_thread

    def test_concurrent_reads_and_writes(self) -> None:
        """Test concurrent reading and writing."""
        store = ConversationStore([{"role": "system", "content": "Initial"}])
        errors: list[Exception] = []
        stop_flag = threading.Event()

        def writer() -> None:
            for i in range(100):
                try:
                    store.append({"role": "user", "content": f"Message {i}"})
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            while not stop_flag.is_set():
                try:
                    snapshot = store.snapshot()
                    # Verify snapshot is consistent
                    for msg in snapshot:
                        assert "role" in msg
                        assert "content" in msg
                except Exception as e:
                    errors.append(e)

        # Start multiple readers and writers
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=reader))
        threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()

        # Wait for writer to finish
        threads[-1].join()
        stop_flag.set()

        for t in threads[:-1]:
            t.join(timeout=1.0)

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

    def test_snapshot_isolation(self) -> None:
        """Test that snapshots are isolated from concurrent modifications."""
        store = ConversationStore([{"role": "user", "content": "Initial"}])

        # Take a snapshot
        snapshot = store.snapshot()

        # Modify store in another thread
        def modifier() -> None:
            store.replace_all([{"role": "system", "content": "Replaced"}])

        thread = threading.Thread(target=modifier)
        thread.start()
        thread.join()

        # Original snapshot should be unchanged
        assert snapshot == [{"role": "user", "content": "Initial"}]
        # Store should be updated
        assert store.snapshot() == [{"role": "system", "content": "Replaced"}]
