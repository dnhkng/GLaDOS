# MCP Integration

GLaDOS supports Model Context Protocol (MCP) servers for extensible tool access. MCP tools are namespaced as `mcp.<server>.<tool>`.

## Architecture

MCP forms the **Tool Layer** at the bottom of the GLaDOS architecture:

```mermaid
flowchart TB
    A[Main Agent] -->|tool calls| B[Tool Executor]
    B --> C[MCP Tool Layer]
    C --> D[Local Servers]
    C --> E[Remote Servers]
    C --> F[Custom Servers]
```

Both the main agent and subagents can invoke MCP tools. Tools are discovered at startup and registered with the LLM.

## How MCP Servers Work

1. **Startup**: GLaDOS launches configured MCP servers as subprocesses
2. **Discovery**: Each server reports its available tools via the MCP protocol
3. **Namespacing**: Tools are prefixed with `mcp.<server_name>.` to avoid conflicts
4. **Invocation**: When the LLM calls a tool, GLaDOS routes it to the appropriate server
5. **Response**: The server executes the tool and returns results to the LLM

### Transport Types

| Transport | Use Case | Configuration |
|-----------|----------|---------------|
| `stdio` | Local servers | `command` + `args` |
| `http` | Remote servers | `url` + optional `token` |
| `sse` | Server-sent events | `url` + optional `token` |

## Local System Servers

GLaDOS ships with lightweight local MCP servers for system monitoring:

```yaml
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

  - name: "memory"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.memory_server"]
```

### Available Tools

| Server | Tools | Description |
|--------|-------|-------------|
| `system_info` | `cpu_load`, `memory_usage`, `temperatures`, `system_overview` | System metrics |
| `time_info` | `now_iso`, `uptime_seconds`, `boot_time` | Time information |
| `disk_info` | `disk_usage`, `mounts` | Storage status |
| `network_info` | `host_info`, `interfaces` | Network details |
| `process_info` | `process_count`, `top_memory` | Process monitoring |
| `power_info` | `batteries` | Battery status |
| `memory` | `store_fact`, `search_memory`, `list_facts`, `store_summary`, `get_summaries`, `memory_stats` | Long-term memory |

## Demo Server

Test MCP integration with the included demo server:

```bash
python -m glados.mcp.slow_clap_server
```

Configure in `glados_config.yaml`:

```yaml
mcp_servers:
  - name: "slow_clap_demo"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.slow_clap_server"]
```

The LLM can then call `mcp.slow_clap_demo.slow_clap(claps=3)` to get "clap clap clap".

## Creating Custom MCP Servers

### Basic Template

```python
from mcp.server.fastmcp import FastMCP

# Create server with a unique name
mcp = FastMCP("my_custom_server")


@mcp.tool()
def my_tool(param: str) -> str:
    """Description shown to the LLM."""
    return f"Result: {param}"


@mcp.tool()
def another_tool(count: int = 1) -> str:
    """Another tool with a default parameter."""
    return "done " * count


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
```

### Registration

Add to `glados_config.yaml`:

```yaml
mcp_servers:
  - name: "my_custom"
    transport: "stdio"
    command: "python"
    args: ["path/to/my_server.py"]
```

Tools will be available as:
- `mcp.my_custom.my_tool`
- `mcp.my_custom.another_tool`

### Best Practices

1. **Return JSON**: For complex data, return JSON strings
2. **Handle errors**: Return error objects rather than raising exceptions
3. **Docstrings**: Write clear docstrings - these are shown to the LLM
4. **Type hints**: Use type hints for parameters - they inform the LLM

## Remote Servers

### Home Assistant

```yaml
mcp_servers:
  - name: "home_assistant"
    transport: "http"
    url: "http://homeassistant.local:8123/mcp"
    token: "YOUR_LONG_LIVED_TOKEN"
```

### With Headers

```yaml
mcp_servers:
  - name: "custom_api"
    transport: "http"
    url: "https://api.example.com/mcp"
    headers:
      Authorization: "Bearer YOUR_TOKEN"
      X-Custom-Header: "value"
```

## Tool Filtering

Control which tools are exposed to the LLM:

### Allow List

Only expose specific tools:

```yaml
mcp_servers:
  - name: "home_assistant"
    transport: "http"
    url: "http://homeassistant.local:8123/mcp"
    token: "YOUR_TOKEN"
    allowed_tools:
      - "light.*"        # All light tools
      - "climate.set_*"  # Climate setters only
```

### Block List

Hide specific tools:

```yaml
mcp_servers:
  - name: "system_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.system_info_server"]
    blocked_tools:
      - "temperatures"   # Hide temperature tool
```

Patterns support `*` wildcards.

## MCP Resources

MCP servers can provide **resources** - contextual data injected into the LLM context.

```yaml
mcp_servers:
  - name: "home_assistant"
    transport: "http"
    url: "http://homeassistant.local:8123/mcp"
    token: "YOUR_TOKEN"
    context_resources:
      - "ha://config"
      - "ha://states/light.*"
    resource_ttl_s: 300    # Cache for 5 minutes
```

Resources are refreshed at the specified TTL and included as system messages.

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | required | Server identifier (used in namespacing) |
| `transport` | string | `"stdio"` | `"stdio"`, `"http"`, or `"sse"` |
| `command` | string | null | Command to run (stdio transport) |
| `args` | list | `[]` | Command arguments (stdio transport) |
| `env` | dict | null | Environment variables for subprocess |
| `url` | string | null | Server URL (http/sse transport) |
| `headers` | dict | null | HTTP headers |
| `token` | string | null | Authentication token (added as Bearer) |
| `allowed_tools` | list | null | Tool allow patterns (null = all) |
| `blocked_tools` | list | null | Tool block patterns |
| `context_resources` | list | `[]` | Resource URIs to inject as context |
| `resource_ttl_s` | float | `300.0` | Resource cache TTL in seconds |

## Memory MCP Server

Long-term memory is enabled by default. The memory server provides:

- **Fact storage** with source tracking and importance levels
- **Keyword-based search** with importance boost and recency decay
- **Conversation summaries** (session/daily/weekly)
- **Automatic fact extraction** via CompactionAgent during context compaction

### Memory Tools

| Tool | Description |
|------|-------------|
| `store_fact` | Store a fact with source and importance (0.0-1.0) |
| `search_memory` | Search facts by keyword (LLM handles semantic interpretation) |
| `list_facts` | List facts filtered by minimum importance |
| `store_summary` | Store a conversation summary with time period |
| `get_summaries` | Retrieve summaries by period |
| `memory_stats` | Get statistics about stored memories |

### Storage

All memory data is stored in `~/.glados/memory/`:
- `facts.jsonl` - Stored facts
- `summaries.jsonl` - Conversation summaries

### Philosophy

The memory system follows the "LLM-first" principle: search is simple keyword matching, and the main agent handles semantic interpretation of results. This keeps infrastructure lightweight while leveraging the LLM's understanding capabilities.

## See Also

- [README](../README.md) - Full architecture diagram
- [autonomy.md](./autonomy.md) - How autonomy uses MCP tools
