from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("system_info")


def _read_meminfo() -> dict[str, int] | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    data: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        value = rest.strip().split()[0]
        try:
            data[key] = int(value) * 1024
        except ValueError:
            continue
    return data


def _read_temps_sysfs() -> list[dict[str, Any]]:
    readings: list[dict[str, Any]] = []
    thermal_root = Path("/sys/class/thermal")
    if thermal_root.exists():
        for zone in thermal_root.glob("thermal_zone*"):
            temp_path = zone / "temp"
            if not temp_path.exists():
                continue
            try:
                milli_c = int(temp_path.read_text(encoding="utf-8").strip())
            except ValueError:
                continue
            label = (zone / "type").read_text(encoding="utf-8").strip() if (zone / "type").exists() else zone.name
            readings.append({"sensor": label, "celsius": milli_c / 1000.0})

    hwmon_root = Path("/sys/class/hwmon")
    if hwmon_root.exists():
        for hwmon in hwmon_root.glob("hwmon*"):
            name = (hwmon / "name").read_text(encoding="utf-8").strip() if (hwmon / "name").exists() else hwmon.name
            for temp_input in hwmon.glob("temp*_input"):
                try:
                    milli_c = int(temp_input.read_text(encoding="utf-8").strip())
                except ValueError:
                    continue
                sensor_id = temp_input.stem.replace("_input", "")
                label_path = hwmon / f"{sensor_id}_label"
                label = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else sensor_id
                readings.append({"sensor": f"{name}:{label}", "celsius": milli_c / 1000.0})

    return readings


@mcp.tool()
def cpu_load() -> str:
    """Return system load averages."""
    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):
        return json.dumps({"error": "loadavg unavailable"})
    return json.dumps({"load_1m": one, "load_5m": five, "load_15m": fifteen})


@mcp.tool()
def memory_usage() -> str:
    """Return memory usage statistics."""
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None

    if psutil:
        mem = psutil.virtual_memory()
        return json.dumps(
            {
                "total_bytes": mem.total,
                "used_bytes": mem.used,
                "available_bytes": mem.available,
                "percent": mem.percent,
            }
        )

    meminfo = _read_meminfo()
    if not meminfo:
        return json.dumps({"error": "meminfo unavailable"})
    total = meminfo.get("MemTotal", 0)
    available = meminfo.get("MemAvailable", 0)
    used = max(total - available, 0)
    percent = (used / total * 100.0) if total else 0.0
    return json.dumps(
        {
            "total_bytes": total,
            "used_bytes": used,
            "available_bytes": available,
            "percent": round(percent, 2),
        }
    )


@mcp.tool()
def temperatures() -> str:
    """Return temperature sensor readings when available."""
    readings = _read_temps_sysfs()
    if not readings:
        return json.dumps({"error": "temperature sensors unavailable"})
    return json.dumps({"readings": readings})


@mcp.tool()
def system_overview() -> str:
    """Return a combined snapshot of load, memory, and temperature."""
    payload = {"load": json.loads(cpu_load()), "memory": json.loads(memory_usage())}
    payload["temperatures"] = json.loads(temperatures())
    return json.dumps(payload)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
