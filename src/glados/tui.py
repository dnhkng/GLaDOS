from collections.abc import Iterator
from pathlib import Path
import random
import sys
from datetime import datetime
from typing import ClassVar, Literal

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Digits, Footer, Header, Label, Log, RichLog, Static, Input

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
        random_blocks = " ".join(
            random.choice(self.BLOCKS) for _ in range(self.size.width - 8)
        )
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
        id: str | None = "",
        speed: float = 0.01,  # time between each character
        repeat: bool = False,  # whether to start again at the end
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._text = text
        self.__id = id
        self._speed = speed
        self._repeat = repeat

    def compose(self) -> ComposeResult:
        self._static = Static()
        self._vertical_scroll = VerticalScroll(self._static, id=self.__id)
        yield self._vertical_scroll

    def _get_iterator(self) -> Iterator[str]:
        # Use proper markup for blinking cursor
        return (self._text[:i] + "[blink]▃[/blink]" for i in range(len(self._text) + 1))

    def on_mount(self) -> None:
        self._iter_text = self._get_iterator()
        self.set_interval(self._speed, self._display_next_char)

    def _display_next_char(self) -> None:
        """Get and display the next character."""
        try:
            if not self._vertical_scroll.is_vertical_scroll_end:
                self._vertical_scroll.scroll_down()
            # Update the static widget with the next text
            self._static.update(next(self._iter_text))
        except StopIteration:
            if self._repeat:
                self._iter_text = self._get_iterator()


# Screens


class SplashScreen(Screen[None]):
    """Splash screen shown on startup."""

    def __init__(
        self,
        mode: Literal["audio", "text", "twitch"] = "audio",
        name=None,
        id=None,
        classes=None,
    ):
        super().__init__(name, id, classes)
        self.mode = mode
        self.glados_instance = None

    with open(Path("src/glados/glados_ui/images/splash.ansi"), encoding="utf-8") as f:
        SPLASH_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")

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

    def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.app.action_quit()  # Use self.app instead of global app
        self.dismiss()
        self.app.start_glados(mode=self.mode)


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
        dialog.border_subtitle = "[blink]Press Esc key to continue[/]"


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

    with open(Path("src/glados/glados_ui/images/logo.ansi"), encoding="utf-8") as f:
        LOGO_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")

    def __init__(
        self,
        mode: Literal["audio", "text", "twitch"] = "audio",
        driver_class=None,
        css_path=None,
        watch_css=False,
        ansi_color=False,
    ):
        super().__init__(driver_class, css_path, watch_css, ansi_color)
        self.mode = mode

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
                    typewriter = Typewriter(
                        recipe, id="recipe", speed=0.01, repeat=True
                    )
                    yield typewriter

        yield Footer()

        # Blocks are displayed in a different layer, and out of the normal flow
        with Container(id="block_container", classes="fadeable"):
            yield ScrollingBlocks(id="scrolling_block", classes="block")
            with Vertical(id="text_block", classes="block"):
                yield Digits("2.67")
                yield Digits("1002")
                yield Digits("45.6")
            yield Label(self.LOGO_ANSI, id="logo_block", classes="block")

        if self.mode == "text":
            yield Input(placeholder="Type your message here...", id="user_input")

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
        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} - {message}"
        logger.add(print, format=fmt, level="INFO")

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
        if self.mode != "twitch":
            self.push_screen(SplashScreen(mode=self.mode))
        else:
            self.start_glados(mode=self.mode)

    def action_help(self) -> None:
        """Someone pressed the help key!."""
        self.push_screen(HelpScreen(id="help_screen"))

    def on_key(self, event: events.Key) -> None:
        """ "A key is pressed."""
        logger.debug(f"Pressed {event.character}")

    def action_quit(self) -> None:  # type: ignore
        """
        Quit the application and exit with a status code of 0.

        This method terminates the current Textual application instance,
        effectively closing the terminal user interface.

        Note:
            - The commented-out `self.glados.cancel()` suggests a potential future implementation
                for cancelling background tasks before exiting.
            - Uses `exit(0)` to indicate a successful, intentional application termination.

        Raises:
            SystemExit: Exits the application with a zero status code.
        """
        # self.glados.cancel()
        self.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        if self.mode == "text" and event.value.strip():
            user_input = event.value
            self.send_message_to_llm(user_input=user_input)

    def send_message_to_llm(self, user_input: str, user_name: str = "You"):
        if user_input.strip():
            # Get the current time and format it as HH:mm
            current_time = datetime.now().strftime("%H:%M")

            # Update the line to include the timestamp
            self.query_one("#log_area").write(
                f"{current_time} | {user_name}: {user_input}"
            )
            if self.mode == "text":
                self.query_one("#user_input").value = ""
            self._send_message_to_llm(user_input=user_input)

    def _send_message_to_llm(self, user_input: str):
        if self.glados_instance:  # Ensure Glados instance exists
            # Send the message to the LLM queue
            self.glados_instance.llm_queue.put(user_input)

            # Wait for the assistant's response
            self.glados_instance.processing = True
            self.glados_instance.currently_speaking.set()

    def start_glados(self, mode: str = "audio") -> None:
        """
        Start the GLaDOS worker thread in the background.

        This method initializes a worker thread to run the GLaDOS module's start function.
        The worker is run exclusively and in a separate thread to prevent blocking the main application.

        Args:
            mode (str): Interaction mode, either "audio" or "text". Defaults to "audio".
        """
        config_path = "configs/glados_config.yaml"
        glados_config = GladosConfig.from_yaml(str(config_path))
        self.glados_instance = Glados.from_config(
            glados_config
        )  # Store the Glados instance
        if mode == "audio":
            self.run_worker(
                self.glados_instance.start_listen_event_loop,  # self.glados_instance.start_listen_event_loop,
                exclusive=False,
                thread=True,
            )
        elif mode == "twitch":
            self.run_worker(
                self.glados_instance.start_auto_talk_loop,
                exclusive=False,
                thread=True,
            )
        elif mode == "text":
            pass
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'audio', 'text' or 'twitch'."
            )

    @classmethod
    def run_app(
        cls, config_path: str | Path = "glados_config.yaml", mode: str = "audio"
    ) -> None:
        """Class method to create and run the app instance.

        Args:
            config_path (str | Path): Path to the configuration file. Defaults to "glados_config.yaml".
            mode (str): Interaction mode, either "audio" or "text". Defaults to "audio".
        """
        try:
            app = cls(mode=mode)
            app.run()
        except KeyboardInterrupt:
            sys.exit()


if __name__ == "__main__":
    GladosUI.run_app()
