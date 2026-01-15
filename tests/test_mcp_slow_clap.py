import sys
import time

import pytest

from glados.mcp import MCPManager, MCPServerConfig

pytest.importorskip("mcp")


def test_mcp_slow_clap_stdio():
    config = MCPServerConfig(
        name="slow_clap_demo",
        transport="stdio",
        command=sys.executable,
        args=["-m", "glados.mcp.slow_clap_server"],
    )
    manager = MCPManager([config], tool_timeout=5.0)
    manager.start()
    try:
        deadline = time.time() + 5.0
        tool_name = "mcp.slow_clap_demo.slow_clap"
        tools = []
        while time.time() < deadline:
            tools = manager.get_tool_definitions()
            if any(tool.get("function", {}).get("name") == tool_name for tool in tools):
                break
            time.sleep(0.1)
        assert any(tool.get("function", {}).get("name") == tool_name for tool in tools)
        result = manager.call_tool(tool_name, {"claps": 2})
        assert result == "clap clap"
    finally:
        manager.shutdown()
