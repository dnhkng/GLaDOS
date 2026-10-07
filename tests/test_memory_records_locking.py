"""Memory locks protect writers and import without Unix modules on Windows."""

import builtins
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from glados.core import memory_records


@pytest.mark.parametrize("fail_body", [False, True])
def test_windows_import_and_lock_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail_body: bool) -> None:
    operations = []

    def locking(fd: int, mode: int, length: int) -> None:
        operations.append((mode, length, os.lseek(fd, 0, os.SEEK_CUR)))

    windows = SimpleNamespace(locking=locking, LK_LOCK=1, LK_UNLCK=0)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "fcntl":
            raise ModuleNotFoundError("fcntl unavailable on Windows")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("windows_memory_records", memory_records.__file__)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as platform:
        platform.setattr(sys, "platform", "win32")
        platform.setitem(sys.modules, "msvcrt", windows)
        platform.setattr(builtins, "__import__", guarded_import)
        spec.loader.exec_module(module)

    def write() -> None:
        with module.locked(tmp_path):
            if fail_body:
                raise RuntimeError("write failed")

    if fail_body:
        with pytest.raises(RuntimeError, match="write failed"):
            write()
    else:
        write()
    assert operations == [(1, 1, 0), (0, 1, 0)]


@pytest.mark.skipif(sys.platform == "win32", reason="Unix flock behavior")
def test_unix_lock_excludes_other_writers_and_releases_on_error(tmp_path: Path) -> None:
    import fcntl

    with pytest.raises(RuntimeError, match="write failed"), memory_records.locked(tmp_path):
        with (tmp_path / ".memory.lock").open("a+b") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise RuntimeError("write failed")
    with (tmp_path / ".memory.lock").open("a+b") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)
