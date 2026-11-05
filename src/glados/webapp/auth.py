"""Shared-secret pairing for console clients that are not on this machine.

Anyone on the same network could otherwise watch the camera, read memory, take over the
microphone or stop GLaDOS. Opening the pairing link (which carries the token) gives that
browser its own session cookie, sent with every fetch, event stream, WebSocket and camera
request, so each device can be revoked on its own.
"""

from __future__ import annotations

import hashlib
import hmac
from http.cookies import CookieError, SimpleCookie
import ipaddress
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any

DEFAULT_PATH = Path("data/webapp_token")
SESSIONS_PATH = Path("data/webapp_sessions.json")
COOKIE = "glados_console"
MIN_LENGTH = 32


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    temporary.replace(path)


def ensure_token(path: Path = DEFAULT_PATH, reset: bool = False) -> str:
    """Return the saved console token, creating a private one on first use (or when ``reset``)."""
    if path.exists() and not reset:
        token = path.read_text().strip()
        if len(token) >= MIN_LENGTH:
            return token
    token = secrets.token_urlsafe(32)
    _write_private(path, token + "\n")
    return token


class SessionStore:
    """Paired browsers. Only a hash of each session cookie is stored, so the file cannot be replayed."""

    TOUCH_INTERVAL_S = 60.0

    def __init__(self, path: Path | None = SESSIONS_PATH) -> None:
        """``path=None`` keeps sessions in memory only."""
        self._path = path
        self._lock = threading.Lock()
        try:
            self._sessions: dict[str, dict[str, Any]] = json.loads(path.read_text()) if path else {}
        except (OSError, ValueError):
            self._sessions = {}

    @staticmethod
    def _key(session: str) -> str:
        return hashlib.sha256(session.encode()).hexdigest()

    def _save(self) -> None:
        if self._path is not None:
            _write_private(self._path, json.dumps(self._sessions, indent=1))

    def create(self, label: str, address: str) -> str:
        session = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._sessions[self._key(session)] = {
                "label": label,
                "address": address,
                "paired_at": now,
                "last_seen": now,
            }
            self._save()
        return session

    def lookup(self, session: str | None, address: str) -> str | None:
        """The device id for a valid session cookie, refreshing when it was last seen."""
        if not session:
            return None
        key = self._key(session)
        with self._lock:
            entry = self._sessions.get(key)
            if entry is None:
                return None
            now = time.time()
            if now - entry["last_seen"] >= self.TOUCH_INTERVAL_S or entry["address"] != address:
                entry.update(last_seen=now, address=address)
                self._save()
        return key[:16]

    def devices(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                ({"id": key[:16], **entry} for key, entry in self._sessions.items()),
                key=lambda device: -device["last_seen"],
            )

    def revoke(self, device_id: str | None = None) -> int:
        """Forget one device (by id) or, with no id, every device."""
        with self._lock:
            doomed = [key for key in self._sessions if device_id is None or key[:16] == device_id]
            for key in doomed:
                del self._sessions[key]
            if doomed:
                self._save()
        return len(doomed)


def is_loopback(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return (mapped or ip).is_loopback


def is_loopback_bind(host: str) -> bool:
    return host.strip("[]").lower() == "localhost" or is_loopback(host.strip("[]"))


def matches(expected: str, candidate: str | None) -> bool:
    return candidate is not None and hmac.compare_digest(expected.encode(), candidate.encode())


def cookie_token(header: str | None) -> str | None:
    try:
        morsel = SimpleCookie(header or "").get(COOKIE)
    except CookieError:
        return None
    return morsel.value if morsel else None


def session_cookie(session: str, secure: bool) -> str:
    return f"{COOKIE}={session}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict" + ("; Secure" if secure else "")


def device_label(user_agent: str) -> str:
    for needle, name in (
        ("iPhone", "iPhone"),
        ("iPad", "iPad"),
        ("Android", "Android"),
        ("Mac OS", "Mac"),
        ("Windows", "Windows"),
        ("Linux", "Linux"),
    ):
        if needle in user_agent:
            browser = next((b for b in ("Firefox", "Edg", "Chrome", "Safari") if b in user_agent), "Browser")
            return f"{name} · {'Edge' if browser == 'Edg' else browser}"
    return "Browser"
