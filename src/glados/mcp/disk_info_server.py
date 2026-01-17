from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import logging

from loguru import logger
from mcp.server.fastmcp import FastMCP

logger.remove()
logging.getLogger().setLevel(logging.CRITICAL)

mcp = FastMCP("disk_info")


def _read_mounts() -> list[dict[str, Any]]:
    mounts_path = Path("/proc/mounts")
    if not mounts_path.exists():
        return []
    mounts: list[dict[str, Any]] = []
    for line in mounts_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mounts.append({"device": parts[0], "mount": parts[1], "fstype": parts[2]})
    return mounts


@mcp.tool()
def disk_usage(path: str = "/") -> str:
    """Return disk usage for a path."""
    try:
        usage = Path(path)
        stats = usage.stat()
        _ = stats  # ensure path exists
        total, used, free = shutil.disk_usage(path)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    percent = round((used / total * 100.0) if total else 0.0, 2)
    return json.dumps(
        {
            "path": path,
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "percent": percent,
        }
    )


@mcp.tool()
def mounts() -> str:
    """Return mounted filesystems from /proc/mounts."""
    data = _read_mounts()
    return json.dumps({"mounts": data})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
