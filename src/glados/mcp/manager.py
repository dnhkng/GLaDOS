import asyncio
import fnmatch
import threading
import time
from collections.abc import Iterable
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .config import MCPServerConfig
from ..observability import ObservabilityBus, trim_message

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # pragma: no cover - handled in runtime checks
    ClientSession = None  # type: ignore[assignment]
    sse_client = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    streamable_http_client = None  # type: ignore[assignment]


class MCPError(RuntimeError):
    pass


class MCPToolError(MCPError):
    pass


@dataclass(frozen=True)
class MCPToolEntry:
    server: str
    name: str
    description: str | None
    input_schema: dict[str, Any] | None


@dataclass
class _ResourceCacheEntry:
    message: dict[str, str]
    expires_at: float


class MCPManager:
    def __init__(
        self,
        servers: Iterable[MCPServerConfig],
        tool_timeout: float = 30.0,
        observability_bus: ObservabilityBus | None = None,
    ) -> None:
        self._servers = {server.name: server for server in servers}
        self._tool_timeout = tool_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run_loop, name="MCPManager", daemon=True)
        self._ready = threading.Event()
        self._shutdown_event = threading.Event()
        self._shutdown_async: asyncio.Event | None = None
        self._observability_bus = observability_bus

        self._tool_lock = threading.Lock()
        self._tool_registry: dict[str, MCPToolEntry] = {}

        self._resource_lock = threading.Lock()
        self._resource_cache: dict[tuple[str, str], _ResourceCacheEntry] = {}
        self._resource_refreshing: set[tuple[str, str]] = set()

        self._session_tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, ClientSession] = {}

    def start(self) -> None:
        if not self._servers:
            return
        if ClientSession is None:
            raise MCPError("MCP client library is not installed. Install the 'mcp' package to enable MCP support.")
        if self._thread.is_alive():
            return
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise MCPError("MCP manager failed to initialize in time.")

    def shutdown(self) -> None:
        if not self._thread.is_alive() or self._loop is None:
            return
        self._shutdown_event.set()
        if self._shutdown_async:
            self._loop.call_soon_threadsafe(self._shutdown_async.set)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        with self._tool_lock:
            entries = list(self._tool_registry.items())
        entries.sort(key=lambda item: item[0])
        return [self._tool_entry_to_definition(tool_name, entry) for tool_name, entry in entries]

    def get_context_messages(self, timeout: float = 5.0, block: bool = True) -> list[dict[str, str]]:
        if not self._servers:
            return []
        messages: list[dict[str, str]] = []
        for server in self._servers.values():
            if not server.context_resources:
                continue
            for uri in server.context_resources:
                cached = self._get_cached_resource(server.name, uri, allow_expired=not block)
                if cached:
                    messages.append(cached.message)
                    if cached.expires_at < time.time():
                        self._schedule_resource_refresh(server.name, uri, server.resource_ttl_s)
                    continue
                if not block:
                    self._schedule_resource_refresh(server.name, uri, server.resource_ttl_s)
                    continue
                fetched = self._fetch_resource(server.name, uri, timeout)
                if fetched:
                    self._cache_resource(server.name, uri, fetched, server.resource_ttl_s)
                    messages.append(fetched)
        return messages

    def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: float | None = None) -> str:
        server_name, local_tool = self._parse_tool_name(tool_name)
        if self._loop is None:
            raise MCPError("MCP manager is not running.")
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(server_name, local_tool, arguments),
            self._loop,
        )
        try:
            return future.result(timeout=timeout or self._tool_timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise MCPError(f"MCP tool '{tool_name}' timed out.") from exc
        except MCPError:
            raise
        except Exception as exc:
            future.cancel()
            raise MCPError(str(exc)) from exc

    def status_snapshot(self) -> list[dict[str, Any]]:
        with self._tool_lock:
            tools = list(self._tool_registry.values())
        with self._resource_lock:
            resource_counts: dict[str, int] = {}
            for (server_name, _uri) in self._resource_cache.keys():
                resource_counts[server_name] = resource_counts.get(server_name, 0) + 1
        connected = set(self._sessions.keys())
        tool_counts: dict[str, int] = {}
        for entry in tools:
            tool_counts[entry.server] = tool_counts.get(entry.server, 0) + 1
        entries: list[dict[str, Any]] = []
        for server in self._servers.values():
            entries.append(
                {
                    "name": server.name,
                    "connected": server.name in connected,
                    "tools": tool_counts.get(server.name, 0),
                    "resources": resource_counts.get(server.name, 0),
                    "context_resources": len(server.context_resources or []),
                }
            )
        entries.sort(key=lambda item: item["name"])
        return entries

    def _run_loop(self) -> None:
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._shutdown_async = asyncio.Event()
        self._loop.create_task(self._initialize_servers())
        self._ready.set()
        self._loop.run_forever()

        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    async def _initialize_servers(self) -> None:
        for server in self._servers.values():
            self._session_tasks[server.name] = asyncio.create_task(self._session_runner(server))

    async def _session_runner(self, config: MCPServerConfig) -> None:
        retry_delay = 2.0
        while not self._shutdown_event.is_set() and self._shutdown_async:
            try:
                async with self._open_transport(config) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        self._sessions[config.name] = session
                        if self._observability_bus:
                            self._observability_bus.emit(
                                source="mcp",
                                kind="connect",
                                message=f"{config.name} connected",
                                meta={"transport": config.transport},
                            )
                        await self._refresh_tools(config, session)
                        await self._refresh_resources(config, session)
                        await self._shutdown_async.wait()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"MCP: server '{config.name}' connection failed: {exc}")
                if self._observability_bus:
                    self._observability_bus.emit(
                        source="mcp",
                        kind="error",
                        message=trim_message(f"{config.name} failed: {exc}"),
                        level="warning",
                    )
                await asyncio.sleep(retry_delay)
            finally:
                self._sessions.pop(config.name, None)
                self._remove_tools_for_server(config.name)
                self._clear_resource_cache(config.name)
                if self._observability_bus:
                    self._observability_bus.emit(
                        source="mcp",
                        kind="disconnect",
                        message=f"{config.name} disconnected",
                    )

    def _open_transport(self, config: MCPServerConfig):
        if config.transport == "stdio":
            if not config.command:
                raise MCPError(f"MCP server '{config.name}' requires a command for stdio transport.")
            params = StdioServerParameters(command=config.command, args=config.args, env=config.env)
            return stdio_client(params)

        if not config.url:
            raise MCPError(f"MCP server '{config.name}' requires a URL for {config.transport} transport.")

        headers = dict(config.headers or {})
        if config.token:
            headers.setdefault("Authorization", f"Bearer {config.token}")

        if config.transport == "http":
            return streamable_http_client(str(config.url), headers=headers)
        if config.transport == "sse":
            return sse_client(str(config.url), headers=headers)

        raise MCPError(f"MCP server '{config.name}' has unsupported transport '{config.transport}'.")

    async def _refresh_tools(self, config: MCPServerConfig, session: ClientSession) -> None:
        response = await session.list_tools()
        tools = self._extract_list(response, "tools")
        entries: dict[str, MCPToolEntry] = {}
        for tool in tools:
            tool_name = self._get_field(tool, "name")
            if not tool_name:
                continue
            if not self._tool_allowed(tool_name, config):
                continue
            description = self._get_field(tool, "description")
            input_schema = self._coerce_dict(self._get_field(tool, "inputSchema", "input_schema"))
            entry = MCPToolEntry(
                server=config.name,
                name=tool_name,
                description=description,
                input_schema=input_schema,
            )
            full_name = self._build_tool_name(config.name, tool_name)
            entries[full_name] = entry

        with self._tool_lock:
            self._tool_registry = {
                tool_name: entry
                for tool_name, entry in self._tool_registry.items()
                if entry.server != config.name
            }
            self._tool_registry.update(entries)
        if self._observability_bus:
            self._observability_bus.emit(
                source="mcp",
                kind="tools",
                message=f"{config.name} tools refreshed",
                meta={"count": len(entries)},
            )

    async def _refresh_resources(self, config: MCPServerConfig, session: ClientSession) -> None:
        if not config.context_resources:
            return
        for uri in config.context_resources:
            message = await self._read_resource(session, config.name, uri)
            if message:
                self._cache_resource(config.name, uri, message, config.resource_ttl_s)

    async def _call_tool_async(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        session = self._sessions.get(server_name)
        if not session:
            raise MCPError(f"MCP server '{server_name}' is not connected.")
        result = await session.call_tool(tool_name, arguments)
        error_flag = self._get_field(result, "isError", "is_error")
        content = self._render_contents(self._get_field(result, "content") or [])
        if error_flag:
            raise MCPToolError(content or "MCP tool reported an error.")
        return content or "success"

    async def _read_resource(self, session: ClientSession, server_name: str, uri: str) -> dict[str, str] | None:
        response = await session.read_resource(uri)
        contents = self._extract_list(response, "contents")
        text = self._render_contents(contents)
        if not text:
            return None
        return {
            "role": "system",
            "content": f"[mcp:{server_name}] Resource {uri}\n{text}",
        }

    def _fetch_resource(self, server_name: str, uri: str, timeout: float) -> dict[str, str] | None:
        if self._loop is None:
            return None
        future = asyncio.run_coroutine_threadsafe(self._fetch_resource_async(server_name, uri), self._loop)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            logger.warning(f"MCP: resource '{uri}' from {server_name} timed out.")
            return None
        except Exception as exc:
            future.cancel()
            logger.warning(f"MCP: failed to fetch resource '{uri}' from {server_name}: {exc}")
            return None

    async def _fetch_resource_async(self, server_name: str, uri: str) -> dict[str, str] | None:
        session = self._sessions.get(server_name)
        if not session:
            return None
        return await self._read_resource(session, server_name, uri)

    async def _refresh_resource_async(self, server_name: str, uri: str, ttl: float) -> None:
        try:
            message = await self._fetch_resource_async(server_name, uri)
            if message:
                self._cache_resource(server_name, uri, message, ttl)
        except Exception as exc:
            logger.warning(f"MCP: failed to refresh resource '{uri}' from {server_name}: {exc}")
        finally:
            self._mark_resource_refresh_complete(server_name, uri)

    def _schedule_resource_refresh(self, server_name: str, uri: str, ttl: float) -> None:
        if self._loop is None:
            return
        key = (server_name, uri)
        with self._resource_lock:
            if key in self._resource_refreshing:
                return
            self._resource_refreshing.add(key)
        try:
            asyncio.run_coroutine_threadsafe(
                self._refresh_resource_async(server_name, uri, ttl),
                self._loop,
            )
        except RuntimeError as exc:
            logger.warning(f"MCP: failed to schedule refresh for '{uri}' from {server_name}: {exc}")
            self._mark_resource_refresh_complete(server_name, uri)

    def _mark_resource_refresh_complete(self, server_name: str, uri: str) -> None:
        with self._resource_lock:
            self._resource_refreshing.discard((server_name, uri))

    def _cache_resource(self, server_name: str, uri: str, message: dict[str, str], ttl: float) -> None:
        expires_at = time.time() + ttl if ttl > 0 else time.time()
        with self._resource_lock:
            self._resource_cache[(server_name, uri)] = _ResourceCacheEntry(message=message, expires_at=expires_at)

    def _get_cached_resource(
        self,
        server_name: str,
        uri: str,
        allow_expired: bool = False,
    ) -> _ResourceCacheEntry | None:
        with self._resource_lock:
            entry = self._resource_cache.get((server_name, uri))
        if not entry:
            return None
        if allow_expired or entry.expires_at >= time.time():
            return entry
        return None

    def _clear_resource_cache(self, server_name: str) -> None:
        with self._resource_lock:
            keys = [key for key in self._resource_cache if key[0] == server_name]
            for key in keys:
                self._resource_cache.pop(key, None)
            refreshing = {key for key in self._resource_refreshing if key[0] == server_name}
            for key in refreshing:
                self._resource_refreshing.discard(key)

    def _remove_tools_for_server(self, server_name: str) -> None:
        with self._tool_lock:
            self._tool_registry = {
                tool_name: entry
                for tool_name, entry in self._tool_registry.items()
                if entry.server != server_name
            }

    def _tool_entry_to_definition(self, tool_name: str, entry: MCPToolEntry) -> dict[str, Any]:
        schema = entry.input_schema or {"type": "object", "properties": {}}
        description = entry.description or f"MCP tool '{entry.name}' from server '{entry.server}'."
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": schema,
            },
        }

    def _parse_tool_name(self, tool_name: str) -> tuple[str, str]:
        if not tool_name.startswith("mcp."):
            raise MCPError(f"Tool name '{tool_name}' is not an MCP tool.")
        parts = tool_name.split(".")
        if len(parts) < 3:
            raise MCPError(f"Tool name '{tool_name}' is missing server or tool name.")
        server_name = parts[1]
        local_tool = ".".join(parts[2:])
        return server_name, local_tool

    @staticmethod
    def _build_tool_name(server_name: str, tool_name: str) -> str:
        return f"mcp.{server_name}.{tool_name}"

    @staticmethod
    def _tool_allowed(tool_name: str, config: MCPServerConfig) -> bool:
        if config.allowed_tools:
            return any(fnmatch.fnmatch(tool_name, pattern) for pattern in config.allowed_tools)
        if config.blocked_tools:
            return not any(fnmatch.fnmatch(tool_name, pattern) for pattern in config.blocked_tools)
        return True

    @staticmethod
    def _get_field(obj: Any, *fields: str) -> Any:
        if isinstance(obj, dict):
            for field in fields:
                if field in obj:
                    return obj[field]
            return None
        for field in fields:
            if hasattr(obj, field):
                return getattr(obj, field)
        return None

    @staticmethod
    def _extract_list(obj: Any, field: str) -> list[Any]:
        value = MCPManager._get_field(obj, field)
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, list):
                return dumped
        return list(value) if isinstance(value, Iterable) else []

    @staticmethod
    def _coerce_dict(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(value, "dict"):
            dumped = value.dict()
            if isinstance(dumped, dict):
                return dumped
        return None

    @staticmethod
    def _render_contents(contents: Iterable[Any]) -> str:
        parts: list[str] = []
        for item in contents:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                    continue
                if "data" in item:
                    parts.append(str(item["data"]))
                    continue
            text = MCPManager._get_field(item, "text")
            if text:
                parts.append(str(text))
                continue
            if item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
