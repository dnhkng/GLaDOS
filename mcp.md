# MCP Integration

This project supports Model Context Protocol (MCP) servers. MCP tools are namespaced as `mcp.<server>.<tool>`.

## Demo Server (slow_clap)
Start the demo MCP server with stdio:

```
python -m glados.mcp.slow_clap_server
```

Configure it in `configs/glados_config.yaml`:

```
mcp_servers:
  - name: "slow_clap_demo"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.slow_clap_server"]
```

## Local System MCP Servers
The project ships with several lightweight local MCP servers (stdio transport).
Enable any of these in `configs/glados_config.yaml` as needed:

```
mcp_servers:
  - name: "system_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.system_info_server"]
  - name: "time_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.time_info_server"]
  - name: "disk_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.disk_info_server"]
  - name: "network_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.network_info_server"]
  - name: "process_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.process_info_server"]
  - name: "power_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.power_info_server"]
```

Available tools (summary):
- `system_info`: `cpu_load`, `memory_usage`, `temperatures`, `system_overview`
- `time_info`: `now_iso`, `uptime_seconds`, `boot_time`
- `disk_info`: `disk_usage`, `mounts`
- `network_info`: `host_info`, `interfaces`
- `process_info`: `process_count`, `top_memory`
- `power_info`: `batteries`

## Home Assistant
If Home Assistant runs on another machine, use HTTP or SSE transport:

```
mcp_servers:
  - name: "home_assistant"
    transport: "http"
    url: "http://homeassistant.local:8123/mcp"
    token: "YOUR_LONG_LIVED_TOKEN"
```

You can optionally limit MCP tools using `allowed_tools` or `blocked_tools` patterns.

To add MCP resources as context messages:

```
mcp_servers:
  - name: "home_assistant"
    transport: "http"
    url: "http://homeassistant.local:8123/mcp"
    token: "YOUR_LONG_LIVED_TOKEN"
    context_resources:
      - "ha://config"
    resource_ttl_s: 300
```
