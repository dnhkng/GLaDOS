"""Unit tests for MCPManager HTTP transport auth handling.

These guard the streamable-HTTP fix: the http branch must build an
httpx client carrying the auth headers (including a derived Bearer token)
and hand it to the canonical streamable_http_client via http_client=.

The original bug was calling streamable_http_client(url, headers=...) —
that function has no headers parameter, so it raised TypeError on connect.
The fix routes headers through a pre-built client instead.
"""

import asyncio
import inspect

import pytest

import glados.mcp.manager as manager_mod
from glados.mcp import MCPManager, MCPServerConfig

pytest.importorskip("mcp")


def _http_config(**overrides) -> MCPServerConfig:
    base = dict(name="auth_server", transport="http", url="https://example.com/mcp")
    base.update(overrides)
    return MCPServerConfig(**base)


class _FakeClient:
    """Stand-in for the httpx.AsyncClient built by create_mcp_http_client."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _enter_http_transport(config, monkeypatch):
    """Drive _open_transport's http branch with the SDK calls mocked out.

    Returns (yielded_streams, captured) where captured records the headers
    passed to the client factory and the args passed to streamable_http_client.
    """
    captured: dict = {}

    def fake_factory(headers=None, **kwargs):
        captured["headers"] = headers
        return _FakeClient()

    class _FakeTransport:
        def __init__(self, url, *, http_client=None, **kwargs):
            captured["url"] = url
            captured["http_client"] = http_client

        async def __aenter__(self):
            return ("read", "write", "get_session_id")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(manager_mod, "create_mcp_http_client", fake_factory)
    monkeypatch.setattr(manager_mod, "streamable_http_client", _FakeTransport)

    async def drive():
        async with MCPManager([config])._open_transport(config) as streams:
            return streams

    streams = asyncio.run(drive())
    return streams, captured


def test_http_transport_forwards_bearer_token(monkeypatch):
    config = _http_config(token="s3cr3t")
    streams, captured = _enter_http_transport(config, monkeypatch)

    assert streams == ("read", "write", "get_session_id")
    assert captured["url"] == str(config.url)
    assert captured["headers"]["Authorization"] == "Bearer s3cr3t"
    # The auth-bearing client is what gets handed to the SDK transport.
    assert isinstance(captured["http_client"], _FakeClient)


def test_http_transport_without_auth_is_clean(monkeypatch):
    # No token and no headers: the transport still opens, the factory just
    # receives an empty header set (i.e. an unauthenticated client).
    config = _http_config()
    streams, captured = _enter_http_transport(config, monkeypatch)

    assert streams == ("read", "write", "get_session_id")
    assert captured["url"] == str(config.url)
    assert captured["headers"] == {}


def test_http_transport_preserves_explicit_authorization(monkeypatch):
    # An explicit Authorization header must win over the token-derived one.
    config = _http_config(token="ignored", headers={"Authorization": "Bearer explicit"})
    _, captured = _enter_http_transport(config, monkeypatch)

    assert captured["headers"]["Authorization"] == "Bearer explicit"


def test_streamable_http_client_api_uses_http_client_not_headers():
    # Pin the SDK contract the fix depends on: the canonical client takes a
    # pre-built http_client and has no headers parameter (passing headers= is
    # exactly the original bug). create_mcp_http_client is where headers go.
    #
    # importorskip("mcp") only proves the top-level package imports, not that
    # mcp.client.streamable_http exports these symbols. On older SDKs the
    # try/except in manager.py nulls them, so guard before inspect.signature
    # (which would raise TypeError on None instead of skipping).
    if manager_mod.streamable_http_client is None or manager_mod.create_mcp_http_client is None:
        pytest.skip("installed mcp SDK does not expose streamable_http_client / create_mcp_http_client")

    params = inspect.signature(manager_mod.streamable_http_client).parameters
    assert "http_client" in params
    assert "headers" not in params

    factory_params = inspect.signature(manager_mod.create_mcp_http_client).parameters
    assert "headers" in factory_params


def test_create_mcp_http_client_applies_headers():
    # End-to-end on the real factory: headers land on the built httpx client.
    client = manager_mod.create_mcp_http_client(headers={"Authorization": "Bearer x"})
    try:
        assert client.headers["Authorization"] == "Bearer x"
    finally:
        asyncio.run(client.aclose())
