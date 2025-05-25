"""TUI Screens for GLaDOS - Screen definitions for different app states.

This module contains all screen classes used in the GLaDOS TUI, including
the splash screen, help screen, and other modal dialogs.
"""

from pathlib import Path
from typing import ClassVar

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen, Screen
from textual.widgets import Label, Static

from .widgets import Typewriter
from glados.glados_ui.text_resources import aperture, help_text, login_text


class SplashScreen(Screen[None]):
    """Splash screen displayed on application startup.
    
    Shows the GLaDOS logo, aperture branding, and animated login text
    while the AI engine is being initialized in the background.
    
    Features:
    - ANSI art logo display
    - Aperture Science branding
    - Typewriter-animated login text
    - Key press handling for progression
    - Automatic scrolling for visibility
    """

    # Load splash screen ANSI art with error handling
    try:
        with open(Path("src/glados/glados_ui/images/splash.ansi"), encoding="utf-8") as f:
            SPLASH_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")
    except FileNotFoundError:
        logger.error("Splash screen ANSI art file not found. Using placeholder.")
        SPLASH_ANSI = Text.from_markup("[bold red]Splash ANSI Art Missing[/bold red]")

    def compose(self) -> ComposeResult:
        """Compose the splash screen layout.
        
        Creates a container with the GLaDOS logo, Aperture Science banner,
        and animated login text using the typewriter effect.
        
        Returns:
            ComposeResult: Generator yielding the splash screen components
        """
        # Main logo container
        with Container(id="splash_logo_container"):
            yield Static(self.SPLASH_ANSI, id="splash_logo")
            yield Label(aperture, id="banner")
        
        # Animated login text with slow reveal speed
        yield Typewriter(login_text, id="login_text", speed=0.0075)

    def on_mount(self) -> None:
        """Set up splash screen behavior when mounted.
        
        Establishes a periodic scroll timer to ensure the splash content
        remains visible and scrolls to show the latest elements.
        """
        # Auto-scroll to bottom every 0.5 seconds for visibility
        self.set_interval(0.5, self.scroll_end)

    def on_key(self, event: events.Key) -> None:
        """Handle key press events on the splash screen.
        
        Processes user input to either quit the application or proceed
        to the main interface with GLaDOS initialization.
        
        Args:
            event: The key event containing the pressed key information
        """
        if event.key == "q":
            # Quit application immediately
            self.app.action_quit()
        else:
            # Any other key proceeds to main app
            # Play announcement if engine is ready
            if hasattr(self.app, 'glados_engine_instance') and self.app.glados_engine_instance:
                self.app.glados_engine_instance.play_announcement()
            
            # Start the main GLaDOS application
            if hasattr(self.app, 'start_glados'):
                self.app.start_glados()
            
            # Dismiss the splash screen
            self.dismiss()


class HelpScreen(ModalScreen[None]):
    """Modal help screen displaying usage information.
    
    A modal dialog that shows help text with instructions on how to
    use the GLaDOS voice assistant interface.
    
    Features:
    - Modal overlay presentation
    - Typewriter-animated help text
    - Escape key to close
    - Bordered dialog appearance
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Help"

    def compose(self) -> ComposeResult:
        """Compose the help screen layout.
        
        Creates a container with animated help text displayed using
        the typewriter effect for engaging presentation.
        
        Returns:
            ComposeResult: Generator yielding the help dialog components
        """
        yield Container(
            Typewriter(help_text, id="help_text"), 
            id="help_dialog"
        )

    def on_mount(self) -> None:
        """Configure the help dialog when mounted.
        
        Sets up the dialog border with title and subtitle, including
        instructions for closing the help screen.
        """
        dialog = self.query_one("#help_dialog")
        dialog.border_title = self.TITLE
        dialog.border_subtitle = "[blink]Press Esc key to continue[/blink]"