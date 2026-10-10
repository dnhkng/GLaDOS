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
     GET  /api/context           ordered live preview or last submitted inference context
     GET  /api/memory            saved facts/summaries and current Memory Core recall
     GET  /api/stream            SSE: "obs" events + "state" pings
     GET  /api/minds             registered mind statuses
     GET  /api/minds/{id}         single mind status
     GET  /api/minds/{id}/memory  that agent's jsonlines memory entries
     GET  /api/slots             task slots (summary fields)
     GET  /api/slots/{id}         full slot incl. on-demand report
     GET  /api/agents            registered subagent statuses
     POST /api/command           run an engine command ({"command": "/agents"})
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from loguru import logger

from ..tools.manage_slot import save_task
from .config import console_hosts
from .devices import device_snapshot, select_device
from .serializers import (
    build_audio_state,
    build_context,
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
    try:
        return serializer(item)
    except Exception:  # pragma: no cover
        return {}


# --------------------------------------------------------------------------- helpers


def _content_type(name: str) -> str:
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
    for mind in engine.mind_registry.snapshot():
        if mind.mind_id == mind_id:
            return mind
    return None


def _find_store(engine: Any) -> Any | None:
    return getattr(engine, "autonomy_slots", None)


def _agent_manager(engine: Any) -> Any | None:
    return getattr(engine, "subagent_manager", None)


class _EngineHTTPServer(ThreadingHTTPServer):
    """Threading server carrying the live engine reference to handlers."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], engine: Any, allowed_hosts: list[str]):
        self.engine = engine
        self.allowed_hosts = console_hosts(address[0], allowed_hosts)
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    """Stateless handler; reads the engine reference from ``self.server.engine``."""

    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------ utilities
    def _path(self) -> str:
        return urlparse(self.path).path

    def _query(self) -> dict[str, str]:
        parsed = urlparse(self.path)
        return {k: v[0] for k, v in parse_qs(parsed.query).items()}

    def _host_allowed(self) -> bool:
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or any(char.isspace() for char in hosts[0]):
            return False
        try:
            parsed = urlparse("//" + hosts[0])
            # Accessing port validates it, even though the allowlist uses hostnames.
            _ = parsed.port
            return bool(
                parsed.hostname
                and parsed.hostname.lower() in self.server.allowed_hosts
                and parsed.username is None
                and parsed.password is None
                and not (parsed.path or parsed.query or parsed.fragment)
            )
        except ValueError:
            return False

    def _json(self, code: int, payload: Any) -> None:
        body = dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if self._path() in {"/api/context", "/api/memory"}:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name: str) -> None:
        target = (STATIC_DIR / name).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return self._text(404, "Not found")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _content_type(name))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("[webapp] {} {}", self.address_string(), fmt % args)

    # --------------------------------------------------------------- do_GET
    def do_GET(self) -> None:
        if not self._host_allowed():
            self.close_connection = True
            return self._json(421, {"error": "Unexpected Host header"})
        path = self._path()
        if path in ("/", "/index.html"):
            return self._file("index.html")
        # Sibling URLs keep index.html usable as a local-file demo too.
        if path in ("/glados-rig.js", "/glados-avatar.js", "/glados-avatar.css", "/glados-vision.js", "/glados-routing.js", "/glados-context.js", "/glados-devices.js", "/glados-memory.js"):
            return self._file(path[1:])
        if path.startswith("/static/"):
            return self._file(path[len("/static/"):])
        if path in ("/api/vision/frame", "/api/vision/live"):
            origin = self.headers.get("Origin")
            if (origin and urlparse(origin).netloc != self.headers.get("Host")) or self.headers.get(
                "Sec-Fetch-Site"
            ) == "cross-site":
                return self._json(403, {"error": "Cross-origin camera access is not allowed"})
            agent = getattr(self.server.engine, "vision_agent", None)
            if path == "/api/vision/live":
                return self._camera_stream(agent)
            frame = agent.preview() if agent else None
            if frame is None:
                return self._json(404, {"error": "No camera observation available"})
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            self.wfile.write(frame)
            return
        if path.startswith("/api/"):
            try:
                return self._route_api(path)
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                return
            except OSError:  # pragma: no cover
                return
        return self._text(404, "Not found")

    def _camera_stream(self, agent: Any) -> None:
        """Stream CPU-tracked webcam frames independently of E4B captions."""
        overlay = self._query().get("overlay", "1") != "0"
        if agent is None or agent.live_preview(overlay=overlay) is None:
            return self._json(404, {"error": "No live camera frame available"})
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=glados-frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        sequence = -1
        try:
            while True:
                frame = agent.live_preview(overlay=overlay)
                if frame is None:
                    break
                jpeg, revision = frame
                if revision != sequence:
                    self.wfile.write(
                        b"--glados-frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n" + jpeg + b"\r\n"
                    )
                    self.wfile.flush()
                    sequence = revision
                time.sleep(1 / 30)
            self.wfile.write(b"--glados-frame--\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_POST(self) -> None:
        if not self._host_allowed():
            self.close_connection = True
            return self._json(421, {"error": "Unexpected Host header"})
        if self._path() == "/api/command":
            return self._command()
        if self._path() in {
            "/api/control", "/api/instructions", "/api/input", "/api/slots", "/api/minds/control",
            "/api/tools/command", "/api/decisions", "/api/decisions/test",
            "/api/devices", "/api/vision/settings", "/api/search/settings", "/api/memory/edit", "/api/tasks/cancel",
        }:
            return self._operate()
        self._json(404, {"error": "not found"})

    def _operate(self) -> None:
        # Mutation routes are same-origin JSON, including when hosted beyond localhost.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._json(403, {"error": "Cross-origin changes are not allowed"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._json(415, {"error": "Expected application/json"})
        engine = self.server.engine
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = 2_000_000 if self._path() == "/api/decisions/test" else 65536
            if not 0 < length <= limit:
                self.close_connection = True
                return self._json(400, {"error": "Invalid request size"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            path = self._path()
            if path == "/api/search/settings":
                agent = getattr(engine, "search_agent", None)
                if agent is None:
                    return self._json(404, {"error": "Search Core is unavailable in this profile"})
                return self._json(200, agent.preferences.update(body))
            if path == "/api/memory/edit":
                core = getattr(engine, "compaction_agent", None)
                if core is None:
                    raise ValueError("Memory Core unavailable")
                if not isinstance(body.get("id"), str):
                    raise ValueError("Memory ID required")
                return self._json(200, core.mutate_memory(body["id"], body.get("action"),
                    body.get("revision"), body.get("content")))
            if path == "/api/tasks/cancel":
                manager = getattr(engine, "autonomy_tasks", None)
                if manager is None or not isinstance(body.get("slot_id"), str):
                    raise ValueError("Task manager or task ID unavailable")
                return self._json(200, {"cancellation_requested": manager.cancel(body["slot_id"])})
            if path == "/api/vision/settings":
                if set(body) != {"interval_min_s", "interval_max_s"}:
                    raise ValueError("Provide interval_min_s and interval_max_s only")
                agent = getattr(engine, "vision_agent", None)
                if agent is None:
                    return self._json(404, {"error": "Vision Core is unavailable in this profile"})
                agent.set_interval_range(body["interval_min_s"], body["interval_max_s"])
                return self._json(200, agent.snapshot())
            if path == "/api/decisions":
                return self._json(200, engine.decision_lists.mutate(body))
            if path == "/api/decisions/test":
                decision = engine.decision_lists.get(body.get("list_id"))
                if decision is None:
                    return self._json(404, {"error": "Decision list not found"})
                text = body.get("text", "")
                if not isinstance(text, str) or len(text) > 8000:
                    raise ValueError("Text must be at most 8000 characters")
                audio = None
                if body.get("audio"):
                    import base64
                    import io

                    import soundfile as sf

                    from ..core.native_audio import NativeAudioConfig, NativeAudioInput
                    raw = base64.b64decode(body["audio"], validate=True)
                    with sf.SoundFile(io.BytesIO(raw)) as wave:
                        if wave.samplerate != 16000 or wave.channels != 1 or wave.frames > 30 * 16000:
                            raise ValueError("Use mono 16 kHz WAV audio, at most 30 seconds")
                        samples = wave.read(dtype="float32")
                    message = NativeAudioInput(NativeAudioConfig()).message([samples])
                    if not message:
                        raise ValueError("Audio contains no signal")
                    audio = message["_native_audio"]
                if not text.strip() and not audio:
                    raise ValueError("Enter text or record speech to test")
                return self._json(
                    200, engine.router.score(decision, text, audio, dry_run=True, spoken=body.get("spoken") is True)
                )
            if path == "/api/tools/command":
                runner = getattr(engine, "command_runner", None)
                if runner is None:
                    return self._json(503, {"error": "Command slot unavailable"})
                return self._json(200, runner.run(body, source="console"))
            if path == "/api/devices":
                select_device(engine, body.get("kind"), body.get("device"))
                return self._json(200, device_snapshot(engine))
            if path == "/api/minds/control":
                if getattr(engine, "quiet_event", None) and engine.quiet_event.is_set():
                    return self._json(409, {"error": "Wake GLaDOS before running or changing background cores."})
                manager = _agent_manager(engine)
                agent_id, action = body.get("agent_id"), body.get("action")
                if not isinstance(agent_id, str):
                    raise ValueError("agent_id must be text")
                agent = manager.get(agent_id) if manager else None
                if agent is None:
                    return self._json(404, {"error": "Background core not found"})
                if action in {"pause", "resume"}:
                    manager.pause(agent_id, action == "pause")
                elif action == "run":
                    manager.trigger(agent_id)
                else:
                    raise ValueError("Unknown core action")
                engine.observability_bus.emit("subagent", "control", f"{agent_id}: {action}")
                return self._json(200, {"accepted": True, "paused": agent.paused})
            if path == "/api/control":
                action, enabled = body.get("action"), body.get("enabled")
                if not isinstance(enabled, bool):
                    raise ValueError("enabled must be a boolean")
                if action == "microphone":
                    engine.set_asr_muted(not enabled)
                elif action == "voice":
                    engine.set_tts_muted(not enabled)
                elif action == "quiet":
                    engine.set_quiet_mode(enabled)
                elif action == "autonomy":
                    engine.set_autonomy_enabled(enabled)
                elif action == "transcripts" and getattr(engine, "native_audio", None):
                    engine.native_audio.config.user_transcripts = enabled
                else:
                    raise ValueError("Control unavailable")
                return self._json(200, build_state(engine))
            if path == "/api/instructions":
                engine.operator_state.set_instructions(body.get("instructions"))
                engine.observability_bus.emit("operator", "instructions", "Session instructions updated")
                return self._json(200, engine.operator_state.snapshot())
            if path == "/api/input":
                message = body.get("text")
                if not isinstance(message, str) or not message.strip() or len(message) > 8000:
                    raise ValueError("Enter a message of 1-8000 characters")
                if engine.shutdown_event.is_set() or not engine.submit_text_input(message, source="webapp"):
                    return self._json(409, {"error": "Engine is not accepting input"})
                return self._json(202, {"accepted": True})
            store = _find_store(engine)
            if store is None:
                return self._json(409, {"error": "Task board unavailable"})
            slot = save_task(store, body)
            return self._json(200, serialize_slot_full(slot))
        except (ValueError, TypeError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception:
            logger.exception("Console operation failed")
            return self._json(500, {"error": "The engine could not complete the operation"})

    # -------------------------------------------------------------- routing
    def _route_api(self, path: str) -> None:
        engine = self.server.engine
        if path == "/api/devices":
            return self._json(200, device_snapshot(engine))
        if path == "/api/context":
            origin = self.headers.get("Origin")
            if (origin and urlparse(origin).netloc != self.headers.get("Host")) or self.headers.get(
                "Sec-Fetch-Site"
            ) == "cross-site":
                return self._json(403, {"error": "Cross-origin context access is not allowed"})
            query = self._query()
            try:
                return self._json(200, build_context(engine, query.get("mode", "user"), query.get("view", "live")))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
        if path == "/api/memory":
            origin = self.headers.get("Origin")
            if (origin and urlparse(origin).netloc != self.headers.get("Host")) or self.headers.get(
                "Sec-Fetch-Site"
            ) == "cross-site":
                return self._json(403, {"error": "Cross-origin memory access is not allowed"})
            core = getattr(engine, "compaction_agent", None)
            if core is None:
                return self._json(200, {"available": False, "reason": "Memory Core is disabled", "memories": []})
            query = self._query()
            try:
                if "id" in query:
                    return self._json(200, core.memory_entry(query["id"]))
                return self._json(200, core.memory_snapshot(query.get("query", "")[:280], query.get("kind", "all"),
                                                           int(query.get("offset", "0")), int(query.get("limit", "30"))))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
        if path == "/api/stream":
            return self._stream(engine)
        if path == "/api/decisions":
            return self._json(200, engine.decision_lists.snapshot())
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
                    return self._json(404, {"error": "No memory for that core"})
                return self._json(200, {"agent_id": mind_id, "memory": entries})
            if sub:
                return self._json(404, {"error": "Unknown sub-path"})
            mind = _find_mind(engine, mind_id)
            if mind is None:
                return self._json(404, {"error": "core not found"})
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

        bus = engine.observability_bus
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.end_headers()

        for event in bus.snapshot(limit=100)[-100:]:
            self._send_sse("obs", serialize_event(event))

        sub = bus.subscribe()
        next_state = time.monotonic()
        next_snapshot = next_state
        next_camera = next_state
        camera_revision = None
        audio_revision = None
        try:
            while not self.server.engine.shutdown_event.is_set():
                now = time.monotonic()
                if now >= next_state:
                    self._send_sse("state", build_state(engine))
                    next_state = now + 0.5
                if now >= next_snapshot:
                    self._send_sse("snapshot", build_snapshot(engine))
                    next_snapshot = now + 2.0
                if now >= next_camera:
                    audio = build_audio_state(engine)
                    revision = (audio["updated_at"], audio["vad_active"], audio["rms"])
                    if revision != audio_revision:
                        self._send_sse("audio", audio)
                        audio_revision = revision
                    agent = getattr(engine, "vision_agent", None)
                    tracking = agent.tracking_snapshot() if agent else None
                    camera = tracking["camera"] if tracking else {}
                    face = camera.get("face", {})
                    revision = (bool(tracking and tracking["paused"]), camera.get("enabled"),
                                camera.get("connected"), face.get("observed_at"),
                                tracking.get("inference_sequence") if tracking else None)
                    if revision != camera_revision:
                        self._send_sse("camera", tracking)
                        camera_revision = revision
                    next_camera = now + 1 / 30
                try:
                    event = sub.get(timeout=max(0, min(next_state, next_snapshot, next_camera) - time.monotonic()))
                except _queue.Empty:
                    continue
                self._send_sse("obs", serialize_event(event))
                if event.source == "avatar" and event.kind == "performance":
                    # Only live events drive animation; historical obs replay must not replay speech.
                    self._send_sse("performance", event.meta)
        except (BrokenPipeError, ConnectionResetError, OSError):  # pragma: no cover
            pass
        finally:
            bus.unsubscribe(sub)

    def _send_sse(self, event_type: str, payload: Any) -> None:
        data = dumps(payload)
        frame = (f"event: {event_type}\ndata: {data}\n\n").encode()
        self.wfile.write(frame)
        self.wfile.flush()

    # -------------------------------------------------------------- helpers
    def _slots(self, engine: Any) -> list[Any]:
        store = _find_store(engine)
        if store is None:
            return []
        try:
            return store.list_slots()
        except Exception:  # pragma: no cover
            return []

    def _agent_list(self, engine: Any) -> list[dict[str, Any]]:
        manager = _agent_manager(engine)
        if manager is None:
            return []
        try:
            return [_serialize_or(serialize_agent, a) for a in manager.list_agents()]
        except Exception:  # pragma: no cover
            return []

    def _memory(self, engine: Any, agent_id: str) -> list[dict[str, Any]] | None:
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

    def _command(self) -> None:
        engine = self.server.engine
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._json(403, {"error": "Cross-origin changes are not allowed"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._json(415, {"error": "Expected application/json"})
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if not 0 < length <= 65536:
            self.close_connection = True
            return self._json(400, {"error": "Invalid request size"})
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        if not isinstance(body, dict):
            return self._json(400, {"error": "Expected a JSON object"})
        command = str(body.get("command", "")).strip()
        if not command:
            return self._json(400, {"error": "missing 'command'"})
        try:
            result = engine.handle_command(command)
        except Exception as exc:  # pragma: no cover
            logger.exception("webapp command failed")
            result = f"Error: {exc}"
        return self._json(200, {"command": command, "result": result})


def _sub_path(path: str, prefix: str) -> str | None:
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix):]
    if not rest or rest.endswith("/"):
        return None
    return rest


class WebappServer:
    """Lifecycle wrapper: start/stop the console server on a background thread."""

    def __init__(
        self, engine: Any, host: str = "127.0.0.1", port: int = 8050, allowed_hosts: list[str] | None = None
    ) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.allowed_hosts = list(allowed_hosts or [])
        self._server: _EngineHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.bound_port: int | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            self._server = _EngineHTTPServer((self.host, self.port), self.engine, self.allowed_hosts)
            self.bound_port = self._server.server_address[1]
        except (OSError, ValueError) as exc:
            logger.error("webapp: failed to bind {}:{} - {}", self.host, self.port, exc)
            self._server = None
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="GladosWebappServer",
            daemon=True,
        )
        self._thread.start()
        logger.success("Webapp console live: http://{}:{}/", self.host, self.port)

    def shutdown(self) -> None:
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
        port = self.bound_port or self.port
        return f"http://{self.host}:{port}/"

    @property
    def is_running(self) -> bool:
        return self._server is not None


__all__ = ["WebappServer"]
