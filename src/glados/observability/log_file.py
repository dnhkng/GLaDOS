"""Rotated log file, so a session can be investigated after its terminal is gone."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

DEFAULT_PATH = Path("data/logs/glados.log")
_handler_id: int | None = None


def enable_file_log() -> Path | None:
    """Log INFO and above to ``GLADOS_LOG_FILE`` (default data/logs/glados.log; empty disables it).

    Safe to call again after something ran ``logger.remove()``, as the TUI does.
    """
    global _handler_id
    target = os.environ.get("GLADOS_LOG_FILE", str(DEFAULT_PATH))
    if not target:
        return None
    if _handler_id is not None:
        try:
            logger.remove(_handler_id)
        except ValueError:
            pass
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    # diagnose=False keeps variable values (API keys, tokens) out of logged tracebacks.
    _handler_id = logger.add(
        path,
        level="INFO",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    return path
