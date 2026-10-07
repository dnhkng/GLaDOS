"""Atomic JSONL memory edits shared by the in-process core and MCP writer."""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

_WINDOWS = sys.platform == "win32"
if _WINDOWS:
    import msvcrt
else:
    import fcntl


def revision(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:24]


@contextmanager
def locked(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".memory.lock").open("a+b") as lock:
        if _WINDOWS:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _WINDOWS:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def append_record(path: Path, record: dict) -> None:
    with locked(path.parent):
        with path.open("a") as stream:
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
