"""Regression checks for idle event consumers and bounded process logs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import pytest

from glados.autonomy.slots import TaskSlotStore
from glados.observability import ObservabilityBus


def test_idle_consumers_keep_latest_events() -> None:
    bus = ObservabilityBus(max_history=5, subscriber_max=3)
    sub = bus.subscribe()
    for i in range(1000):
        bus.emit("test", "tick", str(i))
    assert [e.message for e in bus.snapshot()] == [str(i) for i in range(995, 1000)]
    assert [e.message for e in bus.drain()] == [str(i) for i in range(995, 1000)]
    assert [sub.get_nowait().message for _ in range(3)] == ["997", "998", "999"]
    assert sub.empty()


def test_concurrent_producers_do_not_block_on_full_queues() -> None:
    bus = ObservabilityBus(max_history=10, subscriber_max=2)
    sub = bus.subscribe()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: bus.emit("test", "tick", str(i)), range(1000)))
    assert len(bus.snapshot()) == len(bus.drain()) == 10
    assert sub.qsize() == 2


@pytest.mark.parametrize("kwargs", [{"max_history": 0}, {"subscriber_max": 0}])
def test_unbounded_observability_limits_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ObservabilityBus(**kwargs)


def test_slot_refreshes_preserve_state_without_spamming_events() -> None:
    bus = ObservabilityBus()
    slots = TaskSlotStore(bus)
    slots.update_slot("vision", "Vision", "active", "scene", notify_user=False, updated_at=1)
    assert bus.drain()[0].level == "debug"
    slots.update_slot("vision", "Vision", "active", "scene", notify_user=False, updated_at=2)
    assert slots.get_slot("vision").updated_at == 2
    assert bus.drain() == []
    slots.update_slot("vision", "Vision", "error", "Camera unavailable", notify_user=False)
    assert bus.drain()[0].level == "warning"
    slots.update_slot("task", "Task", "done", "Useful result", notify_user=True)
    assert bus.drain()[0].level == "info"


def test_process_log_rotation_and_exit_status(tmp_path: Path) -> None:
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_with_logs.py"
    log = tmp_path / "capture.log"
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--log-file",
            str(log),
            "--max-bytes",
            "65536",
            "--backups",
            "2",
            "--",
            sys.executable,
            "-c",
            "import os; os.write(1, b'x' * 1000000); os.write(2, b'FINAL'); raise SystemExit(7)",
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 7, result.stderr
    files = list(tmp_path.glob("capture.log*"))
    assert len(files) == 3
    assert all(file.stat().st_size <= 65536 for file in files)
    assert log.read_text().endswith("FINAL")
