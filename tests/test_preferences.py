"""Tests for PreferencesStore."""

import json
import tempfile
from pathlib import Path

import pytest

from glados.autonomy.preferences import PreferencesStore


class TestPreferencesStore:
    """Tests for the PreferencesStore class."""

    def test_get_set_basic(self):
        """Test basic get/set operations."""
        store = PreferencesStore()

        assert store.get("nonexistent") is None
        assert store.get("nonexistent", "default") == "default"

        store.set("key1", "value1")
        assert store.get("key1") == "value1"

        store.set("key2", 42)
        assert store.get("key2") == 42

        store.set("key3", ["a", "b", "c"])
        assert store.get("key3") == ["a", "b", "c"]

    def test_delete(self):
        """Test delete operation."""
        store = PreferencesStore()

        store.set("key", "value")
        assert store.get("key") == "value"

        result = store.delete("key")
        assert result is True
        assert store.get("key") is None

        # Delete nonexistent key
        result = store.delete("nonexistent")
        assert result is False

    def test_all(self):
        """Test getting all preferences."""
        store = PreferencesStore()

        assert store.all() == {}

        store.set("a", 1)
        store.set("b", 2)

        all_prefs = store.all()
        assert all_prefs == {"a": 1, "b": 2}

        # Ensure it returns a copy
        all_prefs["c"] = 3
        assert "c" not in store.all()

    def test_as_prompt_empty(self):
        """Test as_prompt with no preferences."""
        store = PreferencesStore()
        assert store.as_prompt() is None

    def test_as_prompt_with_data(self):
        """Test as_prompt formatting."""
        store = PreferencesStore()
        store.set("theme", "dark")
        store.set("topics", ["AI", "science"])

        prompt = store.as_prompt()
        assert prompt is not None
        assert "[preferences]" in prompt
        assert "theme: dark" in prompt
        assert "topics: AI, science" in prompt

    def test_persistence(self):
        """Test that preferences persist to file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Write preferences
            store1 = PreferencesStore(path)
            store1.set("key1", "value1")
            store1.set("key2", [1, 2, 3])

            # Read back with new instance
            store2 = PreferencesStore(path)
            assert store2.get("key1") == "value1"
            assert store2.get("key2") == [1, 2, 3]

            # Verify file contents
            with open(path) as f:
                data = json.load(f)
            assert data == {"key1": "value1", "key2": [1, 2, 3]}
        finally:
            path.unlink(missing_ok=True)

    def test_persistence_handles_missing_file(self):
        """Test that missing file is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent" / "prefs.json"
            store = PreferencesStore(path)

            # Should work without error
            store.set("key", "value")
            assert store.get("key") == "value"

            # File should be created
            assert path.exists()

    def test_persistence_handles_corrupt_file(self):
        """Test that corrupt JSON file is handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json {{{")
            path = Path(f.name)

        try:
            # Should not raise, just log warning
            store = PreferencesStore(path)
            assert store.all() == {}

            # Should still work
            store.set("key", "value")
            assert store.get("key") == "value"
        finally:
            path.unlink(missing_ok=True)

    def test_thread_safety(self):
        """Test that operations are thread-safe."""
        import threading

        store = PreferencesStore()
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    store.set(f"key_{n}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    store.all()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
