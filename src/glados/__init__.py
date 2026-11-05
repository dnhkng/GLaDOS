"""GLaDOS - Voice Assistant using ONNX models for speech synthesis and recognition."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.config import GladosConfig
    from .core.engine import Glados

__version__ = "0.1.0"
__all__ = ["Glados", "GladosConfig"]


def __getattr__(name: str) -> type:
    if name in {"Glados", "GladosConfig"}:
        from .core.config import GladosConfig
        from .core.engine import Glados

        return Glados if name == "Glados" else GladosConfig
    raise AttributeError(f"module 'glados' has no attribute {name}")
