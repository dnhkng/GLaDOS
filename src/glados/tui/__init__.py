"""TUI Components for GLaDOS - Modular user interface components.

This package provides all the user interface components for the GLaDOS
voice assistant TUI, organized into logical modules for maintainability.

Modules:
- widgets: Custom Textual widgets for various UI elements
- screens: Screen classes for different application states

The package exports the most commonly used components for easy importing
by the main application.
"""

from .widgets import (
    Printer,
    ScrollingBlocks,
    Typewriter,
    DynamicContentPanel,
    TTSInterruptionWidget,
    ChainOfThoughtWidget,
    LLMTextInput,
)

from .screens import (
    SplashScreen,
    HelpScreen,
)

# Export all commonly used components
__all__ = [
    # Widgets
    "Printer",
    "ScrollingBlocks", 
    "Typewriter",
    "DynamicContentPanel",
    "TTSInterruptionWidget",
    "ChainOfThoughtWidget",
    "LLMTextInput",
    
    # Screens
    "SplashScreen",
    "HelpScreen",
]