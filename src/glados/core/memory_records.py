"""Atomic JSONL memory edits shared by the in-process core and MCP writer."""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, BinaryIO

# The lock strategy is chosen once at import, under sys.platform so type checkers see which module exists.
if sys.platform == "win32":
    import msvcrt

    def _lock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle: BinaryIO) -> None:
        fcntl.flock(handle, fcntl.LOCK_EX)

    def _unlock(handle: BinaryIO) -> None:
        fcntl.flock(handle, fcntl.LOCK_UN)


def revision(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:24]


@contextmanager
def locked(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".memory.lock").open("a+b") as lock:
        _lock(lock)
        try:
            yield
        finally:
            _unlock(lock)


def append_record(path: Path, record: dict[str, Any]) -> None:
    with locked(path.parent), path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def edit_record(path: Path, entry_id: str, expected_revision: str, content: str | None) -> None:
    with locked(path.parent):
        lines = path.read_text().splitlines()
        matches = []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get("id") == entry_id:
                    matches.append((index, row))
            except ValueError:
                continue
        if not matches:
            raise ValueError("Memory not found; refresh the catalogue")
        index, row = matches[-1]
        if revision(row.get("content", "")) != expected_revision:
            raise ValueError("Memory changed; refresh before editing")
        removed = {i for i, _ in matches}
        output = []
        for i, line in enumerate(lines):
            if i == index and content is not None:
                row["content"] = content
                output.append(json.dumps(row, ensure_ascii=False))
            elif i not in removed:
                output.append(line)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
                temporary = stream.name
                stream.write("\n".join(output) + ("\n" if output else ""))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
