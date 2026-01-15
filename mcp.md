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
