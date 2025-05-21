from collections.abc import Iterator
from pathlib import Path
import random
import sys
from typing import ClassVar

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Digits, Footer, Header, Label, Log, RichLog, Static
from textual.worker import Worker, WorkerState

# These imports are from the original code. Ensure they are resolvable in your environment.
# If they are part of the larger project, they should be fine.
# For standalone testing, they would need to be stubbed or provided.
from glados.engine import Glados, GladosConfig
from glados.glados_ui.text_resources import aperture, help_text, login_text, recipe

# Custom Widgets


class Printer(RichLog):
    """A subclass of textual's RichLog which captures and displays all print calls."""

    def on_mount(self) -> None:
        self.wrap = True
        self.markup = True
        self.begin_capture_print()

    def on_print(self, event: events.Print) -> None:
        if (text := event.text) != "\n":
            self.write(text.rstrip().replace("DEBUG", "[red]DEBUG[/]"))


class ScrollingBlocks(Log):
    """A widget for displaying random scrolling blocks."""

    BLOCKS = "⚊⚌☰𝌆䷀"
    DEFAULT_CSS = """
    ScrollingBlocks {
        scrollbar_size: 0 0;
        overflow-x: hidden;
    }"""

    def _animate_blocks(self) -> None:
        # Create a string of blocks of the right length, allowing
        # for border and padding
        """
        Generates and writes a line of random block characters to the log.

        This method creates a string of random block characters with a length adjusted
        to fit the current widget width, accounting for border and padding. Each block
        is randomly selected from the predefined `BLOCKS` attribute.

        The generated line is written to the log using `write_line()`, creating a
        visually dynamic scrolling effect of random block characters.

        Parameters:
            None

        Returns:
            None
        """
        # Ensure width calculation doesn't go negative if self.size.width is small
        num_blocks_to_generate = max(0, self.size.width - 8)
        random_blocks = " ".join(random.choice(self.BLOCKS) for _ in range(num_blocks_to_generate))
        self.write_line(f"{random_blocks}")

    def on_show(self) -> None:
        """
        Set up an interval timer to periodically animate scrolling blocks.

        This method is called when the widget becomes visible, initiating a recurring animation
        that calls the `_animate_blocks` method at a fixed time interval of 0.18 seconds.

        The interval timer ensures continuous block animation while the widget is displayed.
        """
        self.set_interval(0.18, self._animate_blocks)


class Typewriter(Static):
    """A widget which displays text a character at a time."""

    def __init__(
        self,
        text: str = "_",
        id: str | None = None,  # Consistent with typical Textual widget `id` parameter
        speed: float = 0.01,  # time between each character
        repeat: bool = False,  # whether to start again at the end
        *args: str,  # Passed to super().__init__
        **kwargs: str,  # Passed to super().__init__
    ) -> None:
        super().__init__(*args)  # Pass id via kwargs if Typewriter(id=...) is used
        self._text = text
        self.__id_for_child = id  # Store id specifically for the child VerticalScroll
        self._speed = speed
        self._repeat = repeat

    def compose(self) -> ComposeResult:
        self._static = Static()
        self._vertical_scroll = VerticalScroll(self._static, id=self.__id_for_child)
        yield self._vertical_scroll

    def _get_iterator(self) -> Iterator[str]:
        # THE FIX: Changed "[blink]▃[/]" to "[blink]▃[/blink]"
        # This makes the closing tag explicit and less prone to parsing issues
        # across different Rich versions or platforms.
        return (self._text[:i] + "[blink]▃[/blink]" for i in range(len(self._text) + 1))

    def on_mount(self) -> None:
        self._iter_text = self._get_iterator()
        self.set_interval(self._speed, self._display_next_char)

    def _display_next_char(self) -> None:
        """Get and display the next character."""
        try:
            # Scroll down first, then update. This feels more natural for a typewriter.
            if not self._vertical_scroll.is_vertical_scroll_end:
                self._vertical_scroll.scroll_down()
            self._static.update(next(self._iter_text))
        except StopIteration:
            if self._repeat:
                self._iter_text = self._get_iterator()
            # else:
            # Optional: If not repeating, remove the cursor or show final text without cursor.
            # For example: self._static.update(self._text)


# Screens


class SplashScreen(Screen[None]):
    """Splash screen shown on startup."""

    # Ensure this path is correct relative to your project structure/runtime directory
    # Using a try-except block for robustness if the file is missing
    try:
        with open(Path("src/glados/glados_ui/images/splash.ansi"), encoding="utf-8") as f:
            SPLASH_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")
    except FileNotFoundError:
        logger.error("Splash screen ANSI art file not found. Using placeholder.")
        SPLASH_ANSI = Text.from_markup("[bold red]Splash ANSI Art Missing[/bold red]")

    def compose(self) -> ComposeResult:
        """
        Compose the layout for the splash screen.

        This method defines the visual composition of the SplashScreen, creating a container
        with a logo, a banner, and a typewriter-style login text.

        Returns:
            ComposeResult: A generator yielding the screen's UI components, including:
                - A container with a static ANSI logo
                - A label displaying the aperture text
                - A typewriter-animated login text with a slow character reveal speed
        """
        with Container(id="splash_logo_container"):
            yield Static(self.SPLASH_ANSI, id="splash_logo")
            yield Label(aperture, id="banner")
        yield Typewriter(login_text, id="login_text", speed=0.0075)

    def on_mount(self) -> None:
        """
        Automatically scroll the widget to its bottom at regular intervals.

        This method sets up a periodic timer to ensure the widget always displays
        the most recent content by scrolling to the end. The scrolling occurs
        every 0.5 seconds, providing a smooth and continuous view of the latest information.

        Args:
            None

        Returns:
            None
        """
        self.set_interval(0.5, self.scroll_end)

    # Removed duplicated on_key method. Python uses the last definition.
    def on_key(self, event: events.Key) -> None:
        """
        Handle key press events on the splash screen.

        This method is triggered when a key is pressed during the splash screen display.
        If the 'q' key is pressed, it triggers the application quit action.
        Regardless of the key pressed, it dismisses the current splash screen
        and starts the main GLADOS application.

        Args:
            event (events.Key): The key event that was triggered.
        """
        if event.key == "q":
            self.app.action_quit()
        else:
            if self.app.glados_engine_instance:
                self.app.glados_engine_instance.play_announcement()
                self.app.start_glados()
                self.dismiss()


class HelpScreen(ModalScreen[None]):
    """The help screen. Possibly not that helpful."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Help"

    def compose(self) -> ComposeResult:
        """
        Compose the help screen's layout by creating a container with a typewriter widget.

        This method generates the visual composition of the help screen, wrapping the help text
        in a Typewriter widget for an animated text display within a Container.

        Returns:
            ComposeResult: A generator yielding the composed help screen container with animated text.
        """
        yield Container(Typewriter(help_text, id="help_text"), id="help_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#help_dialog")
        dialog.border_title = self.TITLE
        # Consistent use of explicit closing tag for blink
        dialog.border_subtitle = "[blink]Press Esc key to continue[/blink]"


def instantiate_glados() -> Glados:
    """
    Instantiate the GLaDOS engine.

    This function creates an instance of the GLaDOS engine, which is responsible for
    managing the GLaDOS system's operations and interactions. The instance can be used
    to control various aspects of the GLaDOS engine, including starting and stopping
    its event loop.

    Returns:
        Glados: An instance of the GLaDOS engine.
    """

    config_path = Path("configs/glados_config.yaml")
    if not config_path.exists():
        logger.error(f"GLaDOS config file not found: {config_path}")

    glados_config = GladosConfig.from_yaml(str(config_path))
    glados_instance = Glados.from_config(glados_config)
    return glados_instance


# The App


class GladosUI(App[None]):
    """The main app class for the GlaDOS ui."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(
            key="question_mark",
            action="help",
            description="Help",
            key_display="?",
        ),
    ]
    CSS_PATH = "glados_ui/glados.tcss"

    ENABLE_COMMAND_PALETTE = False

    TITLE = "GlaDOS v 1.09"

    SUB_TITLE = "(c) 1982 Aperture Science, Inc."

    try:
        with open(Path("src/glados/glados_ui/images/logo.ansi"), encoding="utf-8") as f:
            LOGO_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")
    except FileNotFoundError:
        logger.error("Logo ANSI art file not found. Using placeholder.")
        LOGO_ANSI = Text.from_markup("[bold red]Logo ANSI Art Missing[/bold red]")

    glados_engine_instance: Glados | None = None
    glados_worker: object | None = None
    instantiation_worker: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        """
        Compose the user interface layout for the GladosUI application.

        This method generates the primary UI components, including a header, body with log and utility areas,
        a footer, and additional decorative blocks. The layout is structured to display:
        - A header with a clock
        - A body containing:
          - A log area (Printer widget)
          - A utility area with a typewriter displaying a recipe
        - A footer
        - Additional decorative elements like scrolling blocks, text digits, and a logo

        Returns:
            ComposeResult: A generator yielding Textual UI components for rendering
        """
        # It would be nice to have the date in the header, but see:
        # https://github.com/Textualize/textual/issues/4666
        yield Header(show_clock=True)

        with Container(id="body"):
            with Horizontal():
                yield (Printer(id="log_area"))
                with Container(id="utility_area"):
                    typewriter = Typewriter(recipe, id="recipe", speed=0.01, repeat=True)
                    yield typewriter

        yield Footer()

        with Container(id="block_container", classes="fadeable"):
            yield ScrollingBlocks(id="scrolling_block", classes="block")
            with Vertical(id="text_block", classes="block"):
                yield Digits("2.67")
                yield Digits("1002")
                yield Digits("45.6")
            yield Label(self.LOGO_ANSI, id="logo_block", classes="block")

    def on_load(self) -> None:
        """
        Configure logging settings when the application starts.

        This method is called during the application initialization, before the
        terminal enters app mode. It sets up a custom logging format and ensures
        that all log messages are printed.

        Key actions:
            - Removes any existing log handlers
            - Adds a new log handler that prints messages with a detailed, formatted output
            - Enables capturing of log text by the main log widget

        The log format includes:
            - Timestamp (YYYY-MM-DD HH:mm:ss.SSS)
            - Log level (padded to 8 characters)
            - Module name
            - Function name
            - Line number
            - Log message
        """
        # Cause logger to print all log text. Printed text can then be  captured
        # by the main_log widget

        logger.remove()
        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}"

        self.instantiation_worker = None  # Reset the instantiation worker reference
        self.start_instantiation()

        logger.add(print, format=fmt, level="SUCCESS")  # Changed to DEBUG for more verbose logging during dev

    def on_mount(self) -> None:
        """
        Mount the application and display the initial splash screen.

        This method is called when the application is first mounted, pushing the SplashScreen
        onto the screen stack to provide a welcome or loading experience for the user before
        transitioning to the main application interface.

        Returns:
            None: Does not return any value, simply initializes the splash screen.
        """
        # Display the splash screen for a few moments
        self.push_screen(SplashScreen())
        self.notify("Loading AI engine...", title="GLaDOS", timeout=6)

    def action_help(self) -> None:
        """Someone pressed the help key!."""
        self.push_screen(HelpScreen(id="help_screen"))

    # def on_key(self, event: events.Key) -> None:
    #     """Useful for debugging via key presses."""
    #     logger.success(f"Key pressed: {self.glados_worker}")

    async def action_quit(self) -> None:
        logger.info("Quit action initiated in TUI.")
        if hasattr(self, "glados_engine_instance") and self.glados_engine_instance is not None:
            logger.info("Signalling GLaDOS engine to stop...")
            self.glados_engine_instance.stop_listen_event_loop()

            if hasattr(self, "glados_worker") and self.glados_worker is not None:
                if isinstance(self.glados_worker, Worker) and self.glados_worker.is_running:
                    logger.warning("Waiting for GLaDOS worker to complete...")
                    try:
                        await self.glados_worker.wait()
                        if self.glados_worker.is_running:
                            logger.warning("GLaDOS worker is still running after wait.")
                        else:
                            logger.info("GLaDOS worker has completed.")
                    except TimeoutError:
                        logger.warning("Timeout waiting for GLaDOS worker to complete.")
                    except Exception as e:
                        logger.error(f"Error waiting for GLaDOS worker: {e}")
                else:
                    logger.info("GLaDOS worker was not running or already finished.")
            else:
                logger.warning("GLaDOS worker attribute not found.")

            # del self.glados_engine_instance
            self.glados_engine_instance = None
        else:
            logger.info("GLaDOS engine instance not found or already cleaned up.")

        logger.info("Exiting Textual application.")
        self.exit()

    def on_worker_state_changed(self, message: Worker.StateChanged) -> None:
        """Handle messages from workers."""

        if message.state == WorkerState.SUCCESS:
            self.notify("AI Engine operational", title="GLaDOS", timeout=2)
        elif message.state == WorkerState.ERROR:
            self.notify("Instantiation failed!", severity="error")

        self.instantiation_worker = None  # Clear the worker reference

    def start_glados(self) -> None:
        """
        Start the GLaDOS worker thread in the background.

        This method initializes a worker thread to run the GLaDOS module's start function.
        The worker is run exclusively and in a separate thread to prevent blocking the main application.

        Notes:
            - Uses `run_worker` to create a non-blocking background task
            - Sets the worker as an instance attribute for potential later reference
            - The `exclusive=True` parameter ensures only one instance of this worker runs at a time
        """
        try:
            # Run in a thread to avoid blocking the UI
            if self.glados_engine_instance is not None:
                self.glados_worker = self.run_worker(
                    self.glados_engine_instance.start_listen_event_loop, exclusive=True, thread=True
                )
                logger.info("GLaDOS worker started.")
            else:
                logger.error("Cannot start GLaDOS worker: glados_engine_instance is None.")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to start GLaDOS: {e}")

    def instantiate_glados(self) -> None:
        """
        Instantiate the GLaDOS engine.

        This function creates an instance of the GLaDOS engine, which is responsible for
        managing the GLaDOS system's operations and interactions. The instance can be used
        to control various aspects of the GLaDOS engine, including starting and stopping
        its event loop.

        Returns:
            Glados: An instance of the GLaDOS engine.
        """

        config_path = Path("configs/glados_config.yaml")
        if not config_path.exists():
            logger.error(f"GLaDOS config file not found: {config_path}")

        glados_config = GladosConfig.from_yaml(str(config_path))
        self.glados_engine_instance = Glados.from_config(glados_config)

    def start_instantiation(self) -> None:
        """Starts the worker to instantiate the slow class."""
        if self.instantiation_worker is not None:
            self.notify("Instantiation already in progress!", severity="warning")
            return

        self.instantiation_worker = self.run_worker(
            self.instantiate_glados,  # The callable function
            thread=True,  # Run in a thread (default)
        )

    @classmethod
    def run_app(cls, config_path: str | Path = "glados_config.yaml") -> None:
        app = None  # Initialize app to None
        try:
            app = cls()
            app.run()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user. Exiting.")
            if app is not None and hasattr(app, "action_quit") and callable(app.action_quit):
                app.action_quit()  # This will now call the improved action_quit
            # No explicit sys.exit(0) here; Textual's app.exit() will handle it.
        except Exception:
            logger.opt(exception=True).critical("Unhandled exception in app run:")
            if app is not None and hasattr(app, "action_quit") and callable(app.action_quit):
                # Attempt a graceful shutdown even on other exceptions
                logger.info("Attempting graceful shutdown due to unhandled exception...")
                app.action_quit()
            sys.exit(1)  # Exit with error for unhandled exceptions


if __name__ == "__main__":
    GladosUI.run_app()
