"""Shared-secret pairing for console clients that are not on this machine.

Anyone on the same network could otherwise watch the camera, read memory, take over the
microphone or stop GLaDOS. A pairing link carrying the token sets an HttpOnly cookie, which
the browser then sends with every fetch, event stream, WebSocket and camera request.
"""

from __future__ import annotations

import hmac
from http.cookies import CookieError, SimpleCookie
import ipaddress
import os
from pathlib import Path
import secrets

DEFAULT_PATH = Path("data/webapp_token")
COOKIE = "glados_console"
MIN_LENGTH = 32


def ensure_token(path: Path = DEFAULT_PATH) -> str:
    """Return the saved console token, creating a private one on first use."""
    if path.exists():
        token = path.read_text().strip()
        if len(token) >= MIN_LENGTH:
            return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token + "\n")
    return token


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


def session_cookie(token: str, secure: bool) -> str:
    return f"{COOKIE}={token}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict" + ("; Secure" if secure else "")
