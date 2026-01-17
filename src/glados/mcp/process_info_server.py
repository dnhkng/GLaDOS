from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logging

from loguru import logger
from mcp.server.fastmcp import FastMCP

logger.remove()
logging.getLogger().setLevel(logging.CRITICAL)

mcp = FastMCP("process_info")


def _iter_proc_status() -> list[dict[str, Any]]:
    proc_root = Path("/proc")
    items: list[dict[str, Any]] = []
    if not proc_root.exists():
        return items
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        status_path = entry / "status"
        if not status_path.exists():
            continue
        data: dict[str, Any] = {"pid": int(entry.name)}
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                value = rest.strip()
                if key in {"Name", "State", "VmRSS"}:
                    data[key] = value
            items.append(data)
        except OSError:
            continue
    return items


@mcp.tool()
def process_count() -> str:
    """Return the number of processes visible in /proc."""
    proc_root = Path("/proc")
    if not proc_root.exists():
        return json.dumps({"error": "/proc unavailable"})
    count = sum(1 for entry in proc_root.iterdir() if entry.name.isdigit())
    return json.dumps({"process_count": count})


@mcp.tool()
def top_memory(limit: int = 5) -> str:
    """Return top processes by resident memory from /proc."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 20))
    items = _iter_proc_status()
    for item in items:
        rss_kb = 0
        raw = item.get("VmRSS", "0 kB")
        try:
            rss_kb = int(str(raw).split()[0])
        except (ValueError, IndexError):
            rss_kb = 0
        item["rss_bytes"] = rss_kb * 1024
        item.pop("VmRSS", None)
    items.sort(key=lambda row: row.get("rss_bytes", 0), reverse=True)
    return json.dumps({"top_memory": items[:limit]})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
