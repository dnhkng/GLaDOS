from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("network_info")


def _list_interfaces() -> list[dict[str, Any]]:
    interfaces: list[dict[str, Any]] = []
    sysfs = Path("/sys/class/net")
    if not sysfs.exists():
        return interfaces
    for iface in sysfs.iterdir():
        if not iface.is_dir():
            continue
        state_path = iface / "operstate"
        mtu_path = iface / "mtu"
        state = state_path.read_text(encoding="utf-8").strip() if state_path.exists() else "unknown"
        mtu = None
        if mtu_path.exists():
            try:
                mtu = int(mtu_path.read_text(encoding="utf-8").strip())
            except ValueError:
                mtu = None
        interfaces.append({"name": iface.name, "state": state, "mtu": mtu})
    return interfaces


def _resolve_addresses(hostname: str) -> list[str]:
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            addresses.add(addr)
    except socket.gaierror:
        pass
    return sorted(addresses)


@mcp.tool()
def host_info() -> str:
    """Return hostname and resolved addresses."""
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    return json.dumps(
        {
            "hostname": hostname,
            "fqdn": fqdn,
            "addresses": _resolve_addresses(hostname),
        }
    )


@mcp.tool()
def interfaces() -> str:
    """Return network interface metadata from sysfs."""
    return json.dumps({"interfaces": _list_interfaces()})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
