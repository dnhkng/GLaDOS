"""GLaDOS - Voice Assistant using ONNX models for speech synthesis and recognition."""

import os

if os.getenv("PYAPP") == "1":
    os.environ["PYAPP_RELATIVE_DIR"] = os.getcwd()

from .core.engine import Glados, GladosConfig

__version__ = "0.1.0"
__all__ = ["Glados", "GladosConfig"]
