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
     POST /api/input             send text to GLaDOS without waiting ({"text": "..."})
     POST /api/chat              send text and wait for her reply ({"text": "...", "from": "Claude"})

Clients that are not on this machine and have no paired-browser cookie (scripts, other AIs)
authenticate with ``Authorization: Bearer <pairing token>`` (the token from ``glados pair``).
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Collection
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar, cast
from urllib.parse import parse_qs, unquote, urlparse

from loguru import logger
import numpy as np

from ..audio_io.remote_io import RemoteSwitchAudioIO
from ..tools.manage_slot import save_task
from . import auth
from .config import AUTO_HOST, console_hosts, local_addresses
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

if TYPE_CHECKING:
    from ..autonomy.mind_scheduler import MindScheduler
    from ..autonomy.slots import TaskSlot, TaskSlotStore
    from ..core.engine import Glados
    from ..observability.minds import MindStatus
    from ..vision.vision_mind import VisionMind

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Served instead of the console to a device that has not been paired yet.
_LOCKED_PAGE = (STATIC_DIR / "locked.html").read_text(encoding="utf-8")


def _serialize_or[T](serializer: Callable[[T], dict[str, Any]], item: T) -> dict[str, Any]:
    try:
        return serializer(item)
    except Exception:  # noqa: BLE001 - the console shows what it can
        logger.opt(exception=True).debug("webapp: {} failed", getattr(serializer, "__name__", serializer))
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
        ".webmanifest": "application/manifest+json",
    }.get(Path(name).suffix.lower(), "application/octet-stream")


def _find_mind(engine: Glados, mind_id: str) -> MindStatus | None:
    for mind in engine.mind_registry.snapshot():
        if mind.mind_id == mind_id:
            return mind
    return None


def _find_store(engine: Glados) -> TaskSlotStore | None:
    store: TaskSlotStore | None = getattr(engine, "autonomy_slots", None)
    return store


def _agent_manager(engine: Glados) -> MindScheduler | None:
    manager: MindScheduler | None = getattr(engine, "subagent_manager", None)
    return manager


class _EngineHTTPServer(ThreadingHTTPServer):
    """Threading server carrying the live engine reference to handlers."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        engine: Glados,
        allowed_hosts: list[str],
        tls: ssl.SSLContext | None = None,
        token: str | None = None,
        trust_loopback: bool = True,
        sessions: auth.SessionStore | None = None,
        authority: Path | None = None,
    ) -> None:
        self.engine = engine
        self.authority = authority
        self.token = token
        self.sessions = sessions or auth.SessionStore(path=None)
        self.trust_loopback = trust_loopback
        self.secure = tls is not None
        self._shared: dict[str, tuple[float, Any]] = {}
        self._shared_lock = threading.Lock()
        self.allowed_hosts = console_hosts(address[0], allowed_hosts)
        super().__init__(address, _Handler)
        if tls is not None:
            # The handshake runs in each handler thread, so one slow client cannot stall accept().
            self.socket = tls.wrap_socket(self.socket, server_side=True, do_handshake_on_connect=False)

    def shared[T](self, name: str, max_age_s: float, build: Callable[[Glados], T]) -> T:
        """Build a stream payload once for every connected tab instead of once per tab."""
        with self._shared_lock:
            built_at, payload = self._shared.get(name, (0.0, None))
            if time.monotonic() - built_at >= max_age_s:
                payload = build(self.engine)
                self._shared[name] = (time.monotonic(), payload)
            return cast("T", payload)  # each name is only ever built by the same function

    def handle_error(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: tuple[str, int],
    ) -> None:
        if isinstance(request, ssl.SSLSocket):
            logger.debug("webapp: TLS connection from {} failed", client_address)
            return
        if isinstance(sys.exc_info()[1], ConnectionError):
            logger.debug("webapp: {} dropped the connection", client_address)
            return  # A closed tab or a phone leaving Wi-Fi is not a server error.
        super().handle_error(request, client_address)


# Files served from the console root as well as under /static/, so index.html also works opened locally.
_STATIC_FILES = frozenset(
    {
        "/glados-rig.js",
        "/glados-avatar.js",
        "/glados-avatar.css",
        "/glados-vision.js",
        "/glados-routing.js",
        "/glados-context.js",
        "/glados-devices.js",
        "/glados-agent.js",
        "/glados-memory.js",
        "/glados-themes.js",
        "/glados-overview.js",
        "/glados-vision-themes.js",
        "/glados-favicon.js",
        "/glados-dom.js",
        "/glados-remote-audio.js",
        "/glados-i18n.js",
        "/manifest.webmanifest",
        "/glados-icon-192.png",
        "/glados-icon-512.png",
        "/glados-themes.css",
        "/glados-console.css",
        "/potato-skin.svg",
        "/potato-skin.png",
    }
)
# POST bodies are small JSON, except a decision test that may carry recorded speech.
_BODY_LIMITS = {"/api/decisions/test": 2_000_000}
# /api/chat waits this long for the spoken reply unless the request asks otherwise.
_CHAT_TIMEOUT_S, _CHAT_TIMEOUT_MAX_S = 120, 600


def _decode_test_audio(encoded: str) -> list[dict[str, Any]]:
    """A base64 mono 16 kHz WAV clip (at most 30 s) as the router's native-audio payload."""
    import io

    import soundfile as sf

    from ..core.native_audio import NativeAudioConfig, NativeAudioInput

    raw = base64.b64decode(encoded, validate=True)
    with sf.SoundFile(io.BytesIO(raw)) as wave:
        if wave.samplerate != 16000 or wave.channels != 1 or wave.frames > 30 * 16000:
            raise ValueError("Use mono 16 kHz WAV audio, at most 30 seconds")
        samples = wave.read(dtype="float32")
    message = NativeAudioInput(NativeAudioConfig()).message([samples])
    if not message:
        raise ValueError("Audio contains no signal")
    parts: list[dict[str, Any]] = message["_native_audio"]
    return parts


def _host_ok(host: str, allowed: Collection[str]) -> bool:
    """Whether a Host header names this console (DNS-rebinding guard, also used for redirects)."""
    if not host or any(char.isspace() for char in host):
        return False
    try:
        parsed = urlparse("//" + host)
        # Accessing port validates it, even though the allowlist uses hostnames.
        _ = parsed.port
        hostname = (parsed.hostname or "").lower()
        return bool(
            hostname
            and (hostname in allowed or (AUTO_HOST in allowed and hostname in local_addresses()))
            and parsed.username is None
            and parsed.password is None
            and not (parsed.path or parsed.query or parsed.fragment)
        )
    except ValueError:
        return False


class _Handler(BaseHTTPRequestHandler):
    """Stateless handler; reads the engine reference from ``self.server.engine``."""

    server: _EngineHTTPServer

    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        if isinstance(self.request, ssl.SSLSocket):
            self.request.settimeout(10)
            self._redirect_plain_http()
            self.request.do_handshake()
            self.request.settimeout(None)
        super().setup()

    def _redirect_plain_http(self) -> None:
        """Answer ``http://`` on the TLS port with a redirect to the same host over HTTPS.

        A TLS client always opens with a handshake record (0x16). Anything else is read raw,
        before OpenSSL would swallow it, so the redirect can reuse the Host the browser asked for.
        """
        raw = socket.socket  # Bypass SSLSocket's methods: no TLS session exists yet.
        if raw.recv(self.request, 1, socket.MSG_PEEK) in (b"", b"\x16"):
            return
        head = raw.recv(self.request, 8192)
        host = next(
            (
                line.split(":", 1)[1].strip()
                for line in head.decode("latin-1").split("\r\n")[1:]
                if line.lower().startswith("host:")
            ),
            "",
        )
        if _host_ok(host, self.server.allowed_hosts):
            target = f"https://{host}/".encode()
            raw.sendall(
                self.request,
                b"HTTP/1.1 301 Moved Permanently\r\nLocation: "
                + target
                + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
        else:
            raw.sendall(self.request, b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        raise ConnectionAbortedError("plain HTTP on the TLS port")

    # ------------------------------------------------------------ utilities
    def _path(self) -> str:
        return urlparse(self.path).path

    def _query(self) -> dict[str, str]:
        parsed = urlparse(self.path)
        return {k: v[0] for k, v in parse_qs(parsed.query).items()}

    def _host_allowed(self) -> bool:
        hosts = self.headers.get_all("Host", [])
        return len(hosts) == 1 and _host_ok(hosts[0], self.server.allowed_hosts)

    def _device(self) -> str | None:
        """This request's paired-device id, if it carries a live session."""
        return self.server.sessions.lookup(auth.cookie_token(self.headers.get("Cookie")), self.client_address[0])

    def _from_this_machine(self) -> bool:
        # Opening the PC's own LAN address from the PC arrives from that address, not from loopback.
        address = self.client_address[0].split("%", 1)[0].lower()
        return self.server.trust_loopback and (auth.is_loopback(address) or address in local_addresses())

    def _bearer(self) -> bool:
        """Scripts and other AIs authenticate with ``Authorization: Bearer <pairing token>``."""
        scheme, _, value = self.headers.get("Authorization", "").partition(" ")
        return self.server.token is not None and scheme.lower() == "bearer" and auth.matches(self.server.token, value)

    def _authorized(self) -> bool:
        if self.server.token is None or self._from_this_machine() or self._bearer():
            return True
        return self._device() is not None

    def _refuse_unpaired(self) -> None:
        self.close_connection = True
        if self._path().startswith("/api/"):
            return self._json(401, {"error": "Open the pairing link shown in GLaDOS's terminal first"})
        body = _LOCKED_PAGE.encode("utf-8")
        self.send_response(401)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _pair(self, token: str, candidate: str) -> None:
        """Exchange a pairing-link token for a session cookie, then drop it from the URL."""
        if not auth.matches(token, candidate):
            self.close_connection = True
            return self._text(403, "Invalid pairing link.")
        session = self.server.sessions.create(
            auth.device_label(self.headers.get("User-Agent", "")), self.client_address[0]
        )
        self.send_response(303)
        self.send_header("Set-Cookie", auth.session_cookie(session, self.server.secure))
        self.send_header("Location", self._path())
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authority(self) -> None:
        if self.server.authority is None or not self.server.authority.is_file():
            return self._text(404, "This console does not use the local certificate authority.")
        body = self.server.authority.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-x509-ca-cert")
        self.send_header("Content-Disposition", 'attachment; filename="glados-ca.crt"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: object) -> None:
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

    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug("[webapp] {} {}", self.address_string(), fmt % args)

    # --------------------------------------------------------------- do_GET
    def do_GET(self) -> None:
        if not self._host_allowed():
            self.close_connection = True
            return self._json(421, {"error": "Unexpected Host header"})
        path = self._path()
        if path == "/glados-ca.crt":
            # Public by nature, and needed before a phone can trust the console at all.
            return self._authority()
        if self.server.token is not None and "token" in self._query():
            return self._pair(self.server.token, self._query()["token"])
        if not self._authorized():
            return self._refuse_unpaired()
        if path in ("/", "/index.html"):
            return self._file("index.html")
        if path in _STATIC_FILES:
            return self._file(path[1:])
        if path.startswith("/static/"):
            return self._file(path[len("/static/") :])
        if path.startswith("/api/"):
            try:
                return self._route_api(path)
            except OSError:  # pragma: no cover - the client went away mid-response
                return
        return self._text(404, "Not found")

    def _cross_site_read(self) -> bool:
        """Whether another site is reading data that only this console's pages should see."""
        origin = self.headers.get("Origin")
        return (
            bool(origin and urlparse(origin).netloc != self.headers.get("Host"))
            or self.headers.get("Sec-Fetch-Site") == "cross-site"
        )

    def _camera_stream(self, agent: VisionMind | None) -> None:
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
            while not self.server.engine.shutdown_event.is_set():
                frame = agent.live_preview(overlay=overlay)
                if frame is None:
                    break
                jpeg, revision = frame
                if revision != sequence:
                    self.wfile.write(
                        b"--glados-frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
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
        if not self._authorized():
            return self._refuse_unpaired()
        if self._path() == "/api/command":
            return self._command()
        if self._path() in self._OPERATIONS:
            return self._operate()
        self._json(404, {"error": "not found"})

    def _operate(self) -> None:
        # Mutation routes are same-origin JSON, including when hosted beyond localhost.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._json(403, {"error": "Cross-origin changes are not allowed"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._json(415, {"error": "Expected application/json"})
        path = self._path()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= _BODY_LIMITS.get(path, 65536):
                self.close_connection = True
                return self._json(400, {"error": "Invalid request size"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise TypeError("Expected a JSON object")
            return self._OPERATIONS[path](self, self.server.engine, body)
        except (ValueError, TypeError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception:
            logger.exception("Console operation failed")
            return self._json(500, {"error": "The engine could not complete the operation"})

    # ----------------------------------------------------------- operations
    # POST handlers: each gets the live engine and the decoded JSON object, and a
    # ValueError or TypeError it raises becomes a 400 with its message.
    def _op_pairing(self, engine: Glados, body: dict[str, Any]) -> None:
        if body.get("action") == "revoke" and isinstance(body.get("id"), str):
            revoked = self.server.sessions.revoke(body["id"])
        elif body.get("action") == "revoke_all":
            revoked = self.server.sessions.revoke()
        else:
            raise ValueError("Use action revoke (with id) or revoke_all")
        self._json(200, {"revoked": revoked, "devices": self.server.sessions.devices()})

    def _op_search_settings(self, engine: Glados, body: dict[str, Any]) -> None:
        agent = getattr(engine, "search_agent", None)
        if agent is None:
            return self._json(404, {"error": "Search Core is unavailable in this profile"})
        self._json(200, agent.preferences.update(body))

    def _op_nexus_settings(self, engine: Glados, body: dict[str, Any]) -> None:
        agent = getattr(engine, "nexus_agent", None)
        if agent is None:
            return self._json(404, {"error": "Nexus Core is unavailable in this profile"})
        self._json(200, agent.update_settings(body))

    def _op_memory_edit(self, engine: Glados, body: dict[str, Any]) -> None:
        core = getattr(engine, "compaction_agent", None)
        if core is None:
            raise ValueError("Memory Core unavailable")
        if not isinstance(body.get("id"), str):
            raise TypeError("Memory ID required")
        self._json(200, core.mutate_memory(body["id"], body.get("action"), body.get("revision"), body.get("content")))

    def _op_cancel_task(self, engine: Glados, body: dict[str, Any]) -> None:
        manager = getattr(engine, "autonomy_tasks", None)
        if manager is None or not isinstance(body.get("slot_id"), str):
            raise ValueError("Task manager or task ID unavailable")
        self._json(200, {"cancellation_requested": manager.cancel(body["slot_id"])})

    def _op_vision_settings(self, engine: Glados, body: dict[str, Any]) -> None:
        if set(body) != {"interval_min_s", "interval_max_s"}:
            raise ValueError("Provide interval_min_s and interval_max_s only")
        agent = getattr(engine, "vision_agent", None)
        if agent is None:
            return self._json(404, {"error": "Vision Core is unavailable in this profile"})
        agent.set_interval_range(body["interval_min_s"], body["interval_max_s"])
        self._json(200, agent.snapshot())

    def _op_decisions(self, engine: Glados, body: dict[str, Any]) -> None:
        self._json(200, engine.decision_lists.mutate(body))

    def _op_test_decision(self, engine: Glados, body: dict[str, Any]) -> None:
        decision = engine.decision_lists.get(body.get("list_id"))
        if decision is None:
            return self._json(404, {"error": "Decision list not found"})
        text = body.get("text", "")
        if not isinstance(text, str) or len(text) > 8000:
            raise ValueError("Text must be at most 8000 characters")
        audio = _decode_test_audio(body["audio"]) if body.get("audio") else None
        if not text.strip() and not audio:
            raise ValueError("Enter text or record speech to test")
        self._json(200, engine.router.score(decision, text, audio, dry_run=True, spoken=body.get("spoken") is True))

    def _op_tool_command(self, engine: Glados, body: dict[str, Any]) -> None:
        runner = getattr(engine, "command_runner", None)
        if runner is None:
            return self._json(503, {"error": "Command slot unavailable"})
        self._json(200, runner.run(body, source="console"))

    def _op_devices(self, engine: Glados, body: dict[str, Any]) -> None:
        select_device(engine, str(body.get("kind") or ""), body.get("device"))
        self._json(200, device_snapshot(engine))

    def _op_mind_control(self, engine: Glados, body: dict[str, Any]) -> None:
        if getattr(engine, "quiet_event", None) and engine.quiet_event.is_set():
            return self._json(409, {"error": "Wake GLaDOS before running or changing background cores."})
        manager = _agent_manager(engine)
        agent_id, action = body.get("agent_id"), body.get("action")
        if not isinstance(agent_id, str):
            raise TypeError("agent_id must be text")
        agent = manager.get(agent_id) if manager else None
        if manager is None or agent is None:
            return self._json(404, {"error": "Background core not found"})
        if action in {"pause", "resume"}:
            manager.pause(agent_id, action == "pause")
        elif action == "run":
            manager.trigger(agent_id)
        else:
            raise ValueError("Unknown core action")
        engine.observability_bus.emit("subagent", "control", f"{agent_id}: {action}")
        self._json(200, {"accepted": True, "paused": agent.paused})

    def _op_control(self, engine: Glados, body: dict[str, Any]) -> None:
        action, enabled = body.get("action"), body.get("enabled")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        if action == "microphone":
            engine.set_asr_muted(not enabled)
        elif action == "voice":
            engine.set_tts_muted(not enabled)
        elif action == "quiet":
            engine.set_quiet_mode(enabled)
        elif action == "autonomy":
            engine.set_autonomy_enabled(enabled)
        elif action == "transcripts" and (native := getattr(engine, "native_audio", None)):
            native.config.user_transcripts = enabled
        else:
            raise ValueError("Control unavailable")
        self._json(200, build_state(engine))

    def _op_shutdown(self, engine: Glados, body: dict[str, Any]) -> None:
        """Shut down GLaDOS: stop service, release GPU VRAM and close console."""
        self._json(200, {"status": "shutting_down", "message": "Shutting down GLaDOS"})

        def _do_shutdown() -> None:
            time.sleep(0.5)
            launcher = Path.home() / ".local/bin/glados-app"
            if launcher.exists():
                try:
                    subprocess.run([str(launcher), "stop"], check=False)
                    return
                except OSError as exc:
                    logger.warning("glados-app stop failed: {}", exc)
            engine.shutdown_event.set()

        threading.Thread(target=_do_shutdown, daemon=True).start()

    def _op_instructions(self, engine: Glados, body: dict[str, Any]) -> None:
        instructions = body.get("instructions")
        if not isinstance(instructions, str):
            raise TypeError("Instructions must be text")
        engine.operator_state.set_instructions(instructions)
        engine.observability_bus.emit("operator", "instructions", "Session instructions updated")
        self._json(200, engine.operator_state.snapshot())

    def _op_input(self, engine: Glados, body: dict[str, Any]) -> None:
        message = body.get("text")
        if not isinstance(message, str) or not message.strip() or len(message) > 8000:
            raise ValueError("Enter a message of 1-8000 characters")
        if engine.shutdown_event.is_set() or not engine.submit_text_input(message, source="webapp"):
            return self._json(409, {"error": "Engine is not accepting input"})
        self._json(202, {"accepted": True})

    def _op_chat(self, engine: Glados, body: dict[str, Any]) -> None:
        """Send a message and wait for GLaDOS's delivered reply to that turn."""
        message, sender, timeout = body.get("text"), body.get("from"), body.get("timeout", _CHAT_TIMEOUT_S)
        if not isinstance(message, str) or not message.strip() or len(message) > 8000:
            raise ValueError("Enter a message of 1-8000 characters")
        if sender is not None and (not isinstance(sender, str) or not 0 < len(sender.strip()) <= 64):
            raise ValueError("from must be a name of 1-64 characters")
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not 1 <= timeout <= _CHAT_TIMEOUT_MAX_S:
            raise ValueError(f"timeout must be 1-{_CHAT_TIMEOUT_MAX_S} seconds")
        if sender:
            message = f"[Message from {' '.join(sender.split())}, another AI] {message}"
        bus = engine.observability_bus
        events = bus.subscribe()
        try:
            if engine.shutdown_event.is_set() or not engine.submit_text_input(message, source="api"):
                return self._json(409, {"error": "Engine is not accepting input"})
            deadline, generation = time.monotonic() + timeout, None
            # One entry per finished turn that spoke, in speaking order: True when it was the final answer
            # rather than a turn that handed off to tools (whose results start another turn).
            spoken: list[bool] = []
            steps: list[str] = []
            while (left := deadline - time.monotonic()) > 0:
                try:
                    event = events.get(timeout=left)
                except queue.Empty:
                    break
                mine = generation is not None and event.meta.get("generation") == generation
                if event.source == "api" and event.kind == "user_input" and generation is None:
                    generation = event.meta.get("generation")
                elif event.kind == "user_input":
                    if generation is not None:  # Someone else spoke; this turn was superseded.
                        return self._json(409, {"error": "Another input replaced this turn", "generation": generation})
                elif event.source == "llm" and event.kind == "turn_end" and mine:
                    final = not event.meta.get("tool_handoff")
                    if event.meta.get("has_text"):
                        spoken.append(final)
                    elif final and not spoken:
                        return self._json(200, {"reply": "", "steps": steps, "generation": generation})
                elif event.source == "tts" and event.kind == "reply" and mine and not event.meta.get("autonomy"):
                    text = event.meta.get("text", event.message)
                    if spoken.pop(0) if spoken else True:
                        return self._json(200, {"reply": text, "steps": steps, "generation": generation})
                    steps.append(text)
            self._json(504, {"error": f"No reply within {timeout:g} s", "accepted": True, "generation": generation})
        finally:
            bus.unsubscribe(events)

    def _op_agent_approve(self, engine: Glados, body: dict[str, Any]) -> None:
        state = getattr(engine, "agent_state", None)
        if state is None:
            return self._json(409, {"error": "Agent tools are unavailable"})
        if self.server.token is not None and not (self._from_this_machine() or self._device() is not None):
            # Bearer clients are scripts and other AIs: they must not approve their own requests.
            return self._json(403, {"error": "Approve from this PC or a paired browser"})
        approval_id, approve = body.get("id"), body.get("approve")
        if not isinstance(approval_id, str) or not isinstance(approve, bool):
            raise TypeError("Give id and approve (true or false)")
        if not state.decide(approval_id, approve):
            return self._json(409, {"error": "This request is no longer pending"})
        engine.observability_bus.emit("agent", "approval", ("Approved: " if approve else "Denied: ") + approval_id)
        self._json(200, state.snapshot())

    def _op_save_task(self, engine: Glados, body: dict[str, Any]) -> None:
        store = _find_store(engine)
        if store is None:
            return self._json(409, {"error": "Task board unavailable"})
        self._json(200, serialize_slot_full(save_task(store, body)))

    _OPERATIONS: ClassVar[dict[str, Callable[[_Handler, Any, dict[str, Any]], None]]] = {
        "/api/pairing": _op_pairing,
        "/api/search/settings": _op_search_settings,
        "/api/nexus/settings": _op_nexus_settings,
        "/api/memory/edit": _op_memory_edit,
        "/api/tasks/cancel": _op_cancel_task,
        "/api/vision/settings": _op_vision_settings,
        "/api/decisions": _op_decisions,
        "/api/decisions/test": _op_test_decision,
        "/api/tools/command": _op_tool_command,
        "/api/devices": _op_devices,
        "/api/minds/control": _op_mind_control,
        "/api/control": _op_control,
        "/api/shutdown": _op_shutdown,
        "/api/instructions": _op_instructions,
        "/api/input": _op_input,
        "/api/chat": _op_chat,
        "/api/agent/approve": _op_agent_approve,
        "/api/slots": _op_save_task,
    }

    # -------------------------------------------------------------- routing
    def _route_api(self, path: str) -> None:
        engine = self.server.engine
        query = self._QUERIES.get(path)
        if query is not None:
            return query(self, engine)
        mind_rest = _sub_path(path, "/api/minds/")
        if mind_rest is not None:
            return self._get_mind(engine, mind_rest)
        slot_id = _sub_path(path, "/api/slots/")
        if slot_id is not None:
            return self._get_slot(engine, unquote(slot_id))
        self._json(404, {"error": "not found"})

    # GET handlers: each gets the live engine and answers the request itself.
    def _get_pairing(self, engine: Glados) -> None:
        self._json(
            200,
            {
                "enabled": self.server.token is not None,
                "current": self._device(),
                "authority": self.server.authority is not None,
                "devices": self.server.sessions.devices(),
            },
        )

    def _get_vision(self, engine: Glados) -> None:
        if self._cross_site_read():
            return self._json(403, {"error": "Cross-origin camera access is not allowed"})
        agent = getattr(engine, "vision_agent", None)
        if self._path() == "/api/vision/live":
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

    def _get_context(self, engine: Glados) -> None:
        if self._cross_site_read():
            return self._json(403, {"error": "Cross-origin context access is not allowed"})
        query = self._query()
        try:
            self._json(200, build_context(engine, query.get("mode", "user"), query.get("view", "live")))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})

    def _get_memory(self, engine: Glados) -> None:
        if self._cross_site_read():
            return self._json(403, {"error": "Cross-origin memory access is not allowed"})
        core = getattr(engine, "compaction_agent", None)
        if core is None:
            return self._json(200, {"available": False, "reason": "Memory Core is disabled", "memories": []})
        query = self._query()
        try:
            if "id" in query:
                return self._json(200, core.memory_entry(query["id"]))
            self._json(
                200,
                core.memory_snapshot(
                    query.get("query", "")[:280],
                    query.get("kind", "all"),
                    int(query.get("offset", "0")),
                    int(query.get("limit", "30")),
                ),
            )
        except ValueError as exc:
            self._json(400, {"error": str(exc)})

    def _get_agent(self, engine: Glados) -> None:
        state = getattr(engine, "agent_state", None)
        if state is None:
            return self._json(200, {"available": False})
        from ..tools.agent import _Collect
        from ..tools.self_improve import SelfImprove

        improver = SelfImprove(_Collect(), {"agent_state": state})
        proposal_id = self._query().get("proposal")
        if proposal_id:
            try:
                meta = improver.load(proposal_id)
            except ValueError as exc:
                return self._json(404, {"error": str(exc)})
            patch = improver.proposals_dir() / proposal_id / "change.patch"
            # The owner reviews the whole change here, so it is not trimmed like tool output.
            diff = patch.read_text(errors="replace")[:400_000] if patch.exists() else ""
            return self._json(200, {**meta, "diff": diff})
        self._json(200, {**state.snapshot(), "proposals": improver.listing()})

    def _get_mind(self, engine: Glados, rest: str) -> None:
        mind_id, _, sub = rest.partition("/")
        mind_id = unquote(mind_id)
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
        self._json(200, serialize_mind(mind))

    def _get_slot(self, engine: Glados, slot_id: str) -> None:
        store = _find_store(engine)
        slot = store.get_slot(slot_id) if store is not None else None
        if slot is None:
            return self._json(404, {"error": "slot not found"})
        self._json(200, serialize_slot_full(slot))

    _QUERIES: ClassVar[dict[str, Callable[[_Handler, Any], None]]] = {
        "/api/audio/remote": lambda handler, engine: handler._remote_audio(),
        "/api/pairing": _get_pairing,
        "/api/vision/frame": _get_vision,
        "/api/vision/live": _get_vision,
        "/api/devices": lambda handler, engine: handler._json(200, device_snapshot(engine)),
        "/api/context": _get_context,
        "/api/memory": _get_memory,
        "/api/stream": lambda handler, engine: handler._stream(engine),
        "/api/decisions": lambda handler, engine: handler._json(200, engine.decision_lists.snapshot()),
        "/api/agent": lambda handler, engine: handler._get_agent(engine),
        "/api/snapshot": lambda handler, engine: handler._json(200, build_snapshot(engine)),
        "/api/state": lambda handler, engine: handler._json(200, build_state(engine)),
        "/api/minds": lambda handler, engine: handler._json(
            200, {"minds": [_serialize_or(serialize_mind, m) for m in engine.mind_registry.snapshot()]}
        ),
        "/api/agents": lambda handler, engine: handler._json(200, {"agents": handler._agent_list(engine)}),
        "/api/slots": lambda handler, engine: handler._json(
            200, {"slots": [_serialize_or(serialize_slot, s) for s in handler._slots(engine)]}
        ),
    }

    # --------------------------------------------------------------- SSE
    def _stream(self, engine: Glados) -> None:
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
                    self._send_sse("state", self.server.shared("state", 0.4, build_state))
                    next_state = now + 0.5
                if now >= next_snapshot:
                    self._send_sse("snapshot", self.server.shared("snapshot", 1.5, build_snapshot))
                    next_snapshot = now + 2.0
                if now >= next_camera:
                    audio = build_audio_state(engine)
                    audio_key = (audio["updated_at"], audio["vad_active"], audio["rms"])
                    if audio_key != audio_revision:
                        self._send_sse("audio", audio)
                        audio_revision = audio_key
                    agent = getattr(engine, "vision_agent", None)
                    tracking = agent.tracking_snapshot() if agent else None
                    camera = tracking["camera"] if tracking else {}
                    face = camera.get("face", {})
                    revision = (
                        bool(tracking and tracking["paused"]),
                        camera.get("enabled"),
                        camera.get("connected"),
                        face.get("observed_at"),
                        tracking.get("inference_sequence") if tracking else None,
                    )
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

    # ------------------------------------------------------- remote audio
    _WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    _WS_MAX_MESSAGE = 1 << 20

    def _remote_audio(self) -> None:
        """WebSocket: a browser's microphone and speaker replace the local ones while connected."""
        audio: RemoteSwitchAudioIO | None = getattr(self.server.engine, "audio_io", None)
        if not isinstance(audio, RemoteSwitchAudioIO):
            return self._json(404, {"error": "Remote audio is disabled"})
        origin = self.headers.get("Origin")
        # Browsers always send Origin on WebSockets; requiring it blocks cross-site hijacking.
        if not origin or urlparse(origin).netloc != self.headers.get("Host"):
            return self._json(403, {"error": "Cross-origin audio is not allowed"})
        key = self.headers.get("Sec-WebSocket-Key", "")
        try:
            valid_key = len(base64.b64decode(key, validate=True)) == 16
        except ValueError:
            valid_key = False
        if self.headers.get("Upgrade", "").lower() != "websocket" or not valid_key:
            return self._json(400, {"error": "WebSocket upgrade required"})
        accept = base64.b64encode(hashlib.sha1((key + self._WS_GUID).encode()).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True
        lock = threading.Lock()

        def send(opcode: int, payload: bytes) -> None:
            size = len(payload)
            head = bytes([0x80 | opcode]) + (
                bytes([size])
                if size < 126
                else struct.pack(">BH", 126, size)
                if size < 1 << 16
                else struct.pack(">BQ", 127, size)
            )
            with lock:
                self.wfile.write(head + payload)

        agent = self.headers.get("User-Agent", "")
        label = "Phone" if any(word in agent for word in ("Mobile", "Android", "iPhone")) else "Browser"
        label += " · " + self.client_address[0]
        # The page pings every 2 s; a vanished phone hands audio back to the PC.
        self.connection.settimeout(10)
        client = audio.attach(label, lambda text: send(0x1, text.encode()), lambda data: send(0x2, data))
        try:
            while True:
                message = self._ws_receive()
                if message is None:
                    break
                opcode, payload = message
                if opcode == 0x2:
                    audio.feed(client, payload)
                elif opcode == 0x1:
                    text = payload.decode("utf-8", "replace")
                    if text.startswith("played:") and text[7:].isdigit():
                        audio.played(client, int(text[7:]))
                elif opcode == 0x9:
                    send(0xA, payload)
                elif opcode == 0x8:
                    send(0x8, payload[:2])
                    break
        except OSError:  # Includes the keepalive timeout.
            pass
        finally:
            audio.detach(client)

    def _ws_receive(self) -> tuple[int, bytes] | None:
        """Read one (possibly fragmented) client message; None when the socket closes or misbehaves."""
        opcode, parts, size = None, [], 0
        while True:
            head = self.rfile.read(2)
            if len(head) < 2:
                return None
            fin, frame_op, masked, length = head[0] & 0x80, head[0] & 0x0F, head[1] & 0x80, head[1] & 0x7F
            if not masked:
                return None
            if length in (126, 127):
                extended = self.rfile.read(2 if length == 126 else 8)
                if len(extended) != (2 if length == 126 else 8):
                    return None
                length = struct.unpack(">H" if length == 126 else ">Q", extended)[0]
            size += length
            if size > self._WS_MAX_MESSAGE:
                return None
            mask = np.frombuffer(self.rfile.read(4), dtype=np.uint8)
            data = np.frombuffer(self.rfile.read(length), dtype=np.uint8)
            if len(mask) < 4 or len(data) < length:
                return None
            payload = (data ^ np.resize(mask, length)).tobytes()
            if frame_op >= 0x8:  # Control frames may interleave with fragments.
                if not fin:
                    return None
                return frame_op, payload
            if frame_op:
                opcode = frame_op
            parts.append(payload)
            if fin and opcode is not None:
                return opcode, b"".join(parts)

    def _send_sse(self, event_type: str, payload: object) -> None:
        data = dumps(payload)
        frame = (f"event: {event_type}\ndata: {data}\n\n").encode()
        self.wfile.write(frame)
        self.wfile.flush()

    # -------------------------------------------------------------- helpers
    def _slots(self, engine: Glados) -> list[TaskSlot]:
        store = _find_store(engine)
        if store is None:
            return []
        try:
            return store.list_slots()
        except Exception:  # noqa: BLE001 - the console shows what it can
            return []

    def _agent_list(self, engine: Glados) -> list[dict[str, Any]]:
        manager = _agent_manager(engine)
        if manager is None:
            return []
        try:
            return [_serialize_or(serialize_agent, a) for a in manager.list_agents()]
        except Exception:  # noqa: BLE001 - the console shows what it can
            return []

    def _memory(self, engine: Glados, agent_id: str) -> list[dict[str, Any]] | None:
        manager = _agent_manager(engine)
        if manager is None:
            return None
        try:
            subagent = manager.get(agent_id)
        except Exception:  # noqa: BLE001 - the console shows what it can
            return None
        if subagent is None:
            return None
        try:
            entries = subagent.memory.list_all()
        except Exception:  # noqa: BLE001 - the console shows what it can
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
    rest = path[len(prefix) :]
    if not rest or rest.endswith("/"):
        return None
    return rest


class WebappServer:
    """Lifecycle wrapper: start/stop the console server on a background thread."""

    def __init__(
        self,
        engine: Glados,
        host: str = "127.0.0.1",
        port: int = 8050,
        allowed_hosts: list[str] | None = None,
        tls: ssl.SSLContext | None = None,
        token: str | None = None,
        trust_loopback: bool = True,
        sessions: auth.SessionStore | None = None,
        authority: Path | None = None,
    ) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.allowed_hosts = list(allowed_hosts or [])
        self.tls = tls
        self.token = token
        self.trust_loopback = trust_loopback
        self.sessions = sessions
        self.authority = authority
        self._server: _EngineHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.bound_port: int | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            self._server = _EngineHTTPServer(
                (self.host, self.port),
                self.engine,
                self.allowed_hosts,
                self.tls,
                self.token,
                self.trust_loopback,
                self.sessions,
                self.authority,
            )
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
        logger.success("Webapp console live: {}", self.url)

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
        return f"{'https' if self.tls else 'http'}://{self.host}:{port}/"

    @property
    def is_running(self) -> bool:
        return self._server is not None


__all__ = ["WebappServer"]
