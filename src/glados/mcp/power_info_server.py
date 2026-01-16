from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("power_info")


def _read_supply(path: Path) -> dict[str, Any]:
    def read_value(name: str) -> str | None:
        file_path = path / name
        if not file_path.exists():
            return None
        return file_path.read_text(encoding="utf-8").strip()

    data: dict[str, Any] = {
        "name": path.name,
        "type": read_value("type"),
        "status": read_value("status"),
        "capacity": read_value("capacity"),
        "present": read_value("present"),
    }
    for key in ("energy_now", "energy_full", "charge_now", "charge_full", "voltage_now", "current_now"):
        raw = read_value(key)
        if raw is None:
            continue
        try:
            data[key] = int(raw)
        except ValueError:
            data[key] = raw
    return data


@mcp.tool()
def batteries() -> str:
    """Return battery information from /sys/class/power_supply."""
    root = Path("/sys/class/power_supply")
    if not root.exists():
        return json.dumps({"error": "power_supply unavailable"})
    entries = [_read_supply(path) for path in root.iterdir() if path.is_dir()]
    batteries_only = [entry for entry in entries if entry.get("type") == "Battery"]
    return json.dumps({"batteries": batteries_only})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
