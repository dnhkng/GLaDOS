"""GLaDOS webapp observability console.

An in-process web server (stdlib only) that exposes the engine's live state to a
browser: a snapshot REST API plus a Server-Sent-Events stream built on the
thread-safe ``ObservabilityBus``. Enable it via ``webapp: {enabled: true}`` in
the config (or ``GLADOS_WEBAPP_ENABLED=1``) and open http://127.0.0.1:8050/.
"""

from __future__ import annotations

from .config import WebappConfig
from .server import WebappServer

__all__ = ["WebappConfig", "WebappServer"]
