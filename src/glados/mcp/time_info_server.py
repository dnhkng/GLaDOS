from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import logging

from loguru import logger
from mcp.server.fastmcp import FastMCP

logger.remove()
logging.getLogger().setLevel(logging.CRITICAL)

mcp = FastMCP("time_info")


def _read_uptime() -> float | None:
    uptime_path = Path("/proc/uptime")
    if not uptime_path.exists():
        return None
    try:
        raw = uptime_path.read_text(encoding="utf-8").split()[0]
        return float(raw)
    except (OSError, ValueError, IndexError):
        return None


@mcp.tool()
def now_iso() -> str:
    """Return current time in ISO-8601 format (UTC and local)."""
    utc = datetime.now(timezone.utc).isoformat()
    local = datetime.now().astimezone().isoformat()
    return json.dumps({"utc": utc, "local": local})


@mcp.tool()
def uptime_seconds() -> str:
    """Return system uptime in seconds when available."""
    uptime = _read_uptime()
    if uptime is None:
        uptime = time.monotonic()
    return json.dumps({"uptime_seconds": round(uptime, 3)})


@mcp.tool()
def boot_time() -> str:
    """Return estimated boot time as an ISO-8601 timestamp."""
    uptime = _read_uptime()
    if uptime is None:
        uptime = time.monotonic()
    boot_ts = time.time() - uptime
    return json.dumps({"boot_time": datetime.fromtimestamp(boot_ts).astimezone().isoformat()})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
