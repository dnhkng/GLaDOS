"""In-process webapp observability console server.

Runs inside the Glados engine process (mirroring the websocket-audio server
pattern) so it can read live thread-safe state directly. A single stdlib
:class:`http.server.ThreadingHTTPServer` serves the static console, a JSON
snapshot API, and a Server-Sent-Events (SSE) stream that pushes observability
events plus periodic state pings to every connected browser.

Endpoints
---------
     GET  /                       static console (``static/index.html``)
     GET  /api/snapshot          aggregate JSON snapshot
     GET  /api/state             lightweight state JSON
     GET  /api/stream            SSE: "obs" events + "state" pings
     GET  /api/minds             registered mind statuses
     GET  /api/minds/{id}         single mind status
     GET  /api/minds/{id}/memory  that agent's jsonlines memory entries
     GET  /api/slots             task slots (summary fields)
     GET  /api/slots/{id}         full slot incl. on-demand report
     GET  /api/agents            registered subagent statuses
"""
# The HTTP boundary intentionally accepts the live engine and its optional
# components structurally; serializers isolate those dynamic reads.
# ruff: noqa: ANN401

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any, cast
from urllib.parse import unquote, urlparse

from loguru import logger

from .serializers import (
    build_snapshot,
    build_state,
    dumps,
    serialize_agent,
    serialize_event,
    serialize_memory_entry,
    serialize_mind,
    serialize_slot,
    serialize_slot_full,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _serialize_or(serializer: Any, item: Any) -> dict[str, Any]:
    """Serialize one API item, failing closed to an empty object."""
    try:
        return cast(dict[str, Any], serializer(item))
    except Exception:  # pragma: no cover
        return {}


# --------------------------------------------------------------------------- helpers


def _content_type(name: str) -> str:
    """Map a static asset suffix to its HTTP content type."""
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
        ".wasm": "application/wasm",
    }.get(Path(name).suffix.lower(), "application/octet-stream")


def _find_mind(engine: Any, mind_id: str) -> Any | None:
    """Find a mind status by identifier in a registry snapshot."""
    for mind in engine.mind_registry.snapshot():
        if mind.mind_id == mind_id:
            return mind
    return None


def _find_store(engine: Any) -> Any | None:
    """Return the optional task-slot store."""
    return getattr(engine, "autonomy_slots", None)


def _agent_manager(engine: Any) -> Any | None:
    """Return the optional subagent manager."""
    return getattr(engine, "subagent_manager", None)


class _EngineHTTPServer(ThreadingHTTPServer):
    """Threading server carrying the live engine reference to handlers."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], engine: Any) -> None:
        """Attach the live engine to a standard threaded HTTP server."""
        self.engine = engine
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    """Stateless handler; reads the engine reference from ``self.server.engine``."""

    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------ utilities
    def _path(self) -> str:
        """Return the decoded request path without its query string."""
        return urlparse(self.path).path

    def _json(self, code: int, payload: Any) -> None:
        """Send one length-delimited JSON response."""
        body = dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code: int, text: str) -> None:
        """Send one UTF-8 plain-text response."""
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name: str) -> None:
        """Serve a file confined to the packaged static directory."""
        target = (STATIC_DIR / name).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            return self._text(404, "Not found")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _content_type(name))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Route request logging through Loguru."""
        logger.debug("[webapp] {} {}", self.address_string(), fmt % args)

    # --------------------------------------------------------------- do_GET
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        """Serve the console, its assets, or a read-only API endpoint."""
        if not self._same_origin():
            return self._json(403, {"error": "cross-origin requests are not allowed"})
        path = self._path()
        if path in ("/", "/index.html"):
            return self._file("index.html")
        if path.startswith("/static/"):
            return self._file(path[len("/static/") :])
        if path.startswith("/api/"):
            try:
                return self._route_api(path)
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                return
            except OSError:  # pragma: no cover
                return
            except Exception:  # pragma: no cover - keep request failures isolated
                logger.exception("webapp API request failed")
                return self._json(500, {"error": "internal server error"})
        return self._text(404, "Not found")

    def _same_origin(self) -> bool:
        """Allow direct clients and same-origin browsers, never cross-origin pages."""
        request_host = self.headers.get("Host", "")
        hostname = urlparse(f"//{request_host}").hostname
        if hostname not in {"127.0.0.1", "localhost"}:
            return False
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed = urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc.lower() == request_host.lower()

    # -------------------------------------------------------------- routing
    def _route_api(self, path: str) -> None:
        """Dispatch a read-only API path."""
        engine = cast(_EngineHTTPServer, self.server).engine
        if path == "/api/stream":
            return self._stream(engine)
        if path == "/api/snapshot":
            return self._json(200, build_snapshot(engine))
        if path == "/api/state":
            return self._json(200, build_state(engine))
        if path == "/api/minds":
            minds = [_serialize_or(serialize_mind, m) for m in engine.mind_registry.snapshot()]
            return self._json(200, {"minds": minds})
        if path == "/api/agents":
            return self._json(200, {"agents": self._agent_list(engine)})
        if path == "/api/slots":
            slots = [_serialize_or(serialize_slot, s) for s in self._slots(engine)]
            return self._json(200, {"slots": slots})

        mind_rest = _sub_path(path, "/api/minds/")
        if mind_rest is not None:
            parts = mind_rest.split("/", 1)
            mind_id = unquote(parts[0])
            sub = parts[1] if len(parts) > 1 else ""
            if sub == "memory":
                entries = self._memory(engine, mind_id)
                if entries is None:
                    return self._json(404, {"error": "No subagent memory for that mind"})
                return self._json(200, {"agent_id": mind_id, "memory": entries})
            if sub:
                return self._json(404, {"error": "Unknown sub-path"})
            mind = _find_mind(engine, mind_id)
            if mind is None:
                return self._json(404, {"error": "mind not found"})
            return self._json(200, serialize_mind(mind))

        slot_id = _sub_path(path, "/api/slots/")
        if slot_id is not None:
            store = _find_store(engine)
            slot = store.get_slot(unquote(slot_id)) if store is not None else None
            if slot is None:
                return self._json(404, {"error": "slot not found"})
            return self._json(200, serialize_slot_full(slot))

        self._json(404, {"error": "not found"})

    # --------------------------------------------------------------- SSE
    def _stream(self, engine: Any) -> None:
        """SSE stream: replay history then push live events + periodic state.

        Uses a private :meth:`ObservabilityBus.subscribe` queue so each browser
        gets its own copy of the stream instead of competing over the TUI's
        single-consumer ``drain()`` queue.
        """
        import queue as _queue
        import time

        bus = engine.observability_bus
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        sub = bus.subscribe()
        try:
            # Subscribe before taking history so an event cannot land in the
            # gap between replay and live delivery. Skip the same event object
            # if it appears in both history and the private queue.
            history = bus.snapshot(limit=100)[-100:]
            replayed = {id(event) for event in history}
            for event in history:
                self._send_sse("obs", serialize_event(event))
            last_state_time = time.time()
            while not engine.shutdown_event.is_set():
                try:
                    event = sub.get(timeout=0.5)
                except _queue.Empty:
                    self._send_sse("state", build_state(engine))
                    last_state_time = time.time()
                    continue
                if id(event) in replayed:
                    replayed.remove(id(event))
                    continue
                self._send_sse("obs", serialize_event(event))
                # Emit state frame if 0.5 seconds have elapsed since last state send
                if time.time() - last_state_time >= 0.5:
                    self._send_sse("state", build_state(engine))
                    last_state_time = time.time()
        except (BrokenPipeError, ConnectionResetError, OSError):  # pragma: no cover
            pass
        except Exception:  # pragma: no cover - stream already has HTTP headers
            logger.exception("webapp event stream failed")
        finally:
            bus.unsubscribe(sub)

    def _send_sse(self, event_type: str, payload: Any) -> None:
        """Write and flush one Server-Sent Event frame."""
        data = dumps(payload)
        frame = (f"event: {event_type}\ndata: {data}\n\n").encode()
        self.wfile.write(frame)
        self.wfile.flush()

    # -------------------------------------------------------------- helpers
    def _slots(self, engine: Any) -> list[Any]:
        """Return task slots without propagating telemetry read failures."""
        store = _find_store(engine)
        if store is None:
            return []
        try:
            return cast(list[Any], store.list_slots())
        except Exception:  # pragma: no cover
            return []

    def _agent_list(self, engine: Any) -> list[dict[str, Any]]:
        """Return serialized subagent statuses."""
        manager = _agent_manager(engine)
        if manager is None:
            return []
        try:
            return [_serialize_or(serialize_agent, a) for a in manager.list_agents()]
        except Exception:  # pragma: no cover
            return []

    def _memory(self, engine: Any, agent_id: str) -> list[dict[str, Any]] | None:
        """Return serialized private memory for a known subagent."""
        manager = _agent_manager(engine)
        if manager is None:
            return None
        try:
            subagent = manager.get(agent_id)
        except Exception:  # pragma: no cover
            return None
        if subagent is None:
            return None
        try:
            entries = subagent.memory.list_all()
        except Exception:  # pragma: no cover
            return None
        return [_serialize_or(serialize_memory_entry, e) for e in entries]


def _sub_path(path: str, prefix: str) -> str | None:
    """Extract a non-empty, non-directory suffix beneath an API prefix."""
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix) :]
    if not rest or rest.endswith("/"):
        return None
    return rest


class WebappServer:
    """Lifecycle wrapper: start/stop the console server on a background thread."""

    def __init__(self, engine: Any, host: str = "127.0.0.1", port: int = 8050) -> None:
        """Configure a loopback-only server without binding it yet."""
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("The unauthenticated webapp console is loopback-only")
        self.engine = engine
        self.host = host
        self.port = port
        self._server: _EngineHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.bound_port: int | None = None

    def start(self) -> None:
        """Bind the socket and start serving on a daemon thread."""
        if self._server is not None:
            return
        try:
            self._server = _EngineHTTPServer((self.host, self.port), self.engine)
            self.bound_port = self._server.server_address[1]
        except OSError as exc:
            logger.error("webapp: failed to bind {}:{} - {}", self.host, self.port, exc)
            self._server = None
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="GladosWebappServer",
            daemon=True,
        )
        self._thread.start()
        logger.success("Webapp console live: http://{}:{}/", self.host, self.bound_port)

    def shutdown(self) -> None:
        """Stop serving, close the socket, and join the server thread."""
        server = self._server
        if server is not None:
            server.shutdown()
            server.server_close()
            self._server = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.bound_port = None

    @property
    def url(self) -> str:
        """Return the effective console URL, including an ephemeral port."""
        port = self.bound_port or self.port
        return f"http://{self.host}:{port}/"

    @property
    def is_running(self) -> bool:
        """Report whether a server socket is currently active."""
        return self._server is not None


__all__ = ["WebappServer"]
