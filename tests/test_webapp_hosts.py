"""Reject DNS-rebinding requests before reads, writes, or streaming routes."""

from collections.abc import Iterator
import http.client
import json
from unittest.mock import Mock

import pytest

from glados.webapp.config import WebappConfig
from glados.webapp.server import WebappServer
from tests.test_webapp import _FakeEngine


@pytest.fixture
def server() -> Iterator[WebappServer]:
    engine = _FakeEngine()
    engine.submit_text_input = Mock(return_value=True)
    app = WebappServer(engine, port=0, allowed_hosts=["console.local"])
    app.start()
    assert app.is_running
    try:
        yield app
    finally:
        engine.shutdown_event.set()
        app.shutdown()


def request(server: WebappServer, method: str, path: str, host: str, origin: str | None = None) -> int:
    connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
    headers = {"Host": host}
    if origin:
        headers["Origin"] = origin
    body = None
    if method == "POST":
        headers["Content-Type"] = "application/json"
        body = json.dumps({"text": "hello"})
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


def test_rebinding_host_is_rejected_on_every_route(server: WebappServer) -> None:
    for path in (
        "/",
        "/api/context",
        "/api/memory",
        "/api/snapshot",
        "/api/stream",
        "/api/vision/frame",
        "/api/vision/live",
    ):
        assert request(server, "GET", path, "evil.example:8050", "http://evil.example:8050") == 421
    for path in ("/api/input", "/api/memory/edit", "/api/control", "/api/command"):
        assert request(server, "POST", path, "evil.example:8050", "http://evil.example:8050") == 421
    server.engine.submit_text_input.assert_not_called()


def test_loopback_and_explicit_hosts_work_but_origin_guard_remains(server: WebappServer) -> None:
    for host in (f"localhost:{server.bound_port}", "127.0.0.1", "[::1]:8050", "LOCALHOST", "console.local:8050"):
        assert request(server, "GET", "/api/snapshot", host) == 200
    assert request(server, "POST", "/api/input", "console.local:8050", "http://console.local:8050") == 202
    assert request(server, "POST", "/api/input", "localhost:8050", "http://evil.example:8050") == 403
    server.engine.submit_text_input.assert_called_once_with("hello", source="webapp")


def test_malformed_and_duplicate_hosts_fail_closed(server: WebappServer) -> None:
    for host in (
        "localhost:bad",
        "[broken",
        "user@localhost",
        "localhost/path",
        "localhost?x",
        "localhost#x",
        "localhost,evil.example",
        "localhost ",
    ):
        assert request(server, "GET", "/api/snapshot", host) == 421
    connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
    try:
        connection.putrequest("GET", "/api/snapshot", skip_host=True)
        connection.putheader("Host", "localhost")
        connection.putheader("Host", "evil.example")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 421
        response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", ""])
def test_wildcard_bind_requires_explicit_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="allowed_hosts"):
        WebappConfig(enabled=True, host=host)
    # Direct server construction must not bypass config validation.
    app = WebappServer(_FakeEngine(), host=host, port=0)
    app.start()
    assert not app.is_running


def test_wildcard_bind_with_explicit_hosts() -> None:
    config = WebappConfig(enabled=True, host="0.0.0.0", allowed_hosts=["console.local"])
    server = WebappServer(_FakeEngine(), host=config.host, port=0, allowed_hosts=config.allowed_hosts)
    server.start()
    try:
        assert server.is_running
        assert request(server, "GET", "/api/snapshot", "console.local") == 200
        assert request(server, "GET", "/api/snapshot", "evil.example") == 421
        assert request(server, "GET", "/api/snapshot", "0.0.0.0") == 421
    finally:
        server.shutdown()


def test_environment_cannot_enable_wildcard_without_host_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from glados.core.engine import GladosConfig

    profile = GladosConfig.model_construct(webapp=WebappConfig(enabled=False))
    monkeypatch.setenv("GLADOS_WEBAPP_ENABLED", "1")
    monkeypatch.setenv("GLADOS_WEBAPP_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="allowed_hosts"):
        profile._apply_webapp_env()
    profile.webapp.allowed_hosts = ["console.local"]
    profile._apply_webapp_env()
    assert profile.webapp.host == "0.0.0.0"
    assert profile.webapp.allowed_hosts == ["console.local"]
