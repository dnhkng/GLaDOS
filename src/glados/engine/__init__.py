"""GLaDOS Engine - Modular voice assistant engine with separated concerns.

This module provides the complete GLaDOS voice assistant engine, built from
modular components with clean separation of concerns. The main Glados class
combines all functionality through mixin inheritance.
"""

# Import logging configuration
import sys
from loguru import logger

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="SUCCESS")

# Import all components from correct modules
from .core import Glados, start
from .config import GladosConfig, PersonalityPrompt
from .audio_models import AudioMessage

# Export main classes and functions for public API
__all__ = [
    "Glados",
    "GladosConfig", 
    "PersonalityPrompt",
    "AudioMessage",
    "start",
]

# For direct script execution, start GLaDOS
if __name__ == "__main__":
    start()
