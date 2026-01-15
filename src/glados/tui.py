from collections.abc import Iterator
from dataclasses import dataclass
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import ClassVar, cast

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Input, Label, RichLog, Static
from textual.worker import Worker, WorkerState

from glados.core.engine import Glados, GladosConfig
from glados.glados_ui.text_resources import aperture, help_text, login_text
from glados.observability import ObservabilityEvent

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


class Typewriter(Static):
    """A widget which displays text a character at a time."""

    def __init__(
        self,
        text: str = "_",
        id: str | None = None,  # Consistent with typical Textual widget `id` parameter
        speed: float = 0.01,  # time between each character
        repeat: bool = False,  # whether to start again at the end
        # Static widget parameters
        content: str = "",
        expand: bool = False,
        shrink: bool = False,
        markup: bool = True,
        name: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        # Initialize our custom attributes first
        self._text = text
        self.__id_for_child = id  # Store id specifically for the child VerticalScroll
        self._speed = speed
        self._repeat = repeat
        # Flag to determine if we should use Rich markup
        self._use_markup = True
        # Check if text contains special Rich markup characters
        if "[" in text or "]" in text:
            # If there are brackets in the text, disable markup to avoid conflicts
            self._use_markup = False

        # Call parent constructor with proper parameters
        super().__init__(
            content,
            expand=expand,
            shrink=shrink,
            markup=markup,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )

    def compose(self) -> ComposeResult:
        self._static = Static(markup=self._use_markup)
        self._vertical_scroll = VerticalScroll(self._static, id=self.__id_for_child)
        yield self._vertical_scroll

    def _get_iterator(self) -> Iterator[str]:
        """
        Create an iterator that returns progressively longer substrings of the text,
        with a cursor at the end.

        If markup is enabled, uses a blinking cursor with Rich markup.
        If markup is disabled (due to brackets in the text), uses a plain underscore.
        """
        if self._use_markup:
            # Use Rich markup for the blinking cursor if markup is enabled
            return (self._text[:i] + "[blink]_[/blink]" for i in range(len(self._text) + 1))
        else:
            # Use a simple underscore cursor if markup is disabled
            return (self._text[:i] + "_" for i in range(len(self._text) + 1))

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


@dataclass(frozen=True)
class DialogLine:
    role: str
    content: str


class DialogLog(RichLog):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._last_timestamp = 0.0
        self.wrap = True
        self.markup = True

    def refresh_from_bus(self, events: list[ObservabilityEvent]) -> None:
        new_events = [event for event in events if event.timestamp > self._last_timestamp]
        if not new_events:
            return
        new_events.sort(key=lambda event: event.timestamp)
        for event in new_events:
            dialog_line = self._event_to_dialog(event)
            if dialog_line:
                self._write_dialog(dialog_line)
        self._last_timestamp = new_events[-1].timestamp

    def _event_to_dialog(self, event: ObservabilityEvent) -> DialogLine | None:
        if event.kind == "user_input" and event.source in {"asr", "text"}:
            return DialogLine(role="You", content=event.message)
        if event.source == "tts" and event.kind == "play":
            return DialogLine(role="GLaDOS", content=event.message)
        return None

    def _write_dialog(self, line: DialogLine) -> None:
        color = "cyan" if line.role == "You" else "yellow"
        self.write(f"[bold {color}]{line.role}[/]: {line.content}")


class StatusPanel(Static):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.markup = True

    @staticmethod
    def _volume_bar(level: float, width: int = 18) -> str:
        level = max(0.0, min(1.0, level))
        filled = int(round(level * width))
        empty = max(0, width - filled)
        return f"[{'#' * filled}{'.' * empty}]"

    @staticmethod
    def _rms_to_db(rms: float) -> float:
        return 20.0 * math.log10(max(rms, 1e-6))

    def render_status(self, app: "GladosUI") -> None:
        engine = app.glados_engine_instance
        if not engine:
            self.update("Engine: starting...")
            return
        snapshot = engine.audio_state.snapshot()
        rms_db = self._rms_to_db(snapshot.rms)
        level = (rms_db + 60.0) / 60.0
        bar = self._volume_bar(level)
        vad_text = "[green]ON[/]" if snapshot.vad_active else "[red]OFF[/]"
        speaking = "[green]YES[/]" if engine.currently_speaking_event.is_set() else "NO"
        processing = "[green]YES[/]" if engine.processing_active_event.is_set() else "NO"
        autonomy = "ON" if engine.autonomy_config.enabled else "OFF"
        jobs = "ON" if engine.autonomy_config.jobs.enabled else "OFF"
        vision = "ON" if engine.vision_config is not None else "OFF"
        asr = "MUTED" if engine.asr_muted_event.is_set() else "ACTIVE"
        lines = [
            f"Input: {engine.input_mode}",
            f"ASR: {asr}",
            f"Speaking: {speaking}",
            f"Processing: {processing}",
            f"Autonomy: {autonomy}  Jobs: {jobs}",
            f"Vision: {vision}",
            "",
            f"VAD: {vad_text}",
            f"Volume: {bar} {rms_db:5.1f} dB",
        ]
        self.update("\n".join(lines))


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

    def on_key(self, event: events.Key) -> None:
        """
        Handle key press events on the splash screen.

        This method is triggered when a key is pressed during the splash screen display.
        All keybinds which are active in the main app are active here automatically
        so, for example, ctrl-q will terminate the app. They do not need to be handled.
        Any other key will start the GlaDOS engine and then dismiss the splash screen.

        Args:
            event (events.Key): The key event that was triggered.
        """
        app = cast(GladosUI, self.app)  # mypy gets confused about app's type
        if app.glados_engine_instance:
            app.glados_engine_instance.play_announcement()
            app.start_glados()
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


class ObservabilityScreen(ModalScreen[None]):
    """Live observability log for system events."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Observability"

    def compose(self) -> ComposeResult:
        with Container(id="observability_dialog"):
            yield Label(self.TITLE, id="observability_title")
            yield RichLog(id="observability_log")
            yield Static("", id="observability_status")

    def on_mount(self) -> None:
        dialog = self.query_one("#observability_dialog")
        dialog.border_title = self.TITLE
        dialog.border_title_align = "center"
        self._log = self.query_one("#observability_log", RichLog)
        self._log.markup = True
        self._status = self.query_one("#observability_status", Static)
        self._load_snapshot()
        self.set_interval(0.25, self._drain_events)

    def _load_snapshot(self) -> None:
        bus = self._get_bus()
        if not bus:
            self._log.write("[red]Observability bus unavailable.[/]")
            return
        for event in bus.snapshot(limit=200):
            self._write_event(event)
        self._update_status()

    def _drain_events(self) -> None:
        bus = self._get_bus()
        if not bus:
            return
        for event in bus.drain(max_items=100):
            self._write_event(event)
        self._update_status()

    def _get_bus(self):
        app = cast(GladosUI, self.app)
        if not app.glados_engine_instance:
            return None
        return app.glados_engine_instance.observability_bus

    def _update_status(self) -> None:
        app = cast(GladosUI, self.app)
        engine = app.glados_engine_instance
        if not engine:
            self._status.update("Engine not ready.")
            return
        slots = engine.autonomy_slots.list_slots() if engine.autonomy_slots else []
        minds = engine.mind_registry.snapshot() if engine.mind_registry else []
        self._status.update(f"slots: {len(slots)} | minds: {len(minds)}")

    def _write_event(self, event: ObservabilityEvent) -> None:
        timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        level = event.level.lower()
        color = {
            "debug": "grey50",
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
        }.get(level, "white")
        meta = self._format_meta(event.meta)
        meta_text = f" [{meta}]" if meta else ""
        message = event.message.replace("\n", " ")
        line = f"[{color}]{level.upper():<5}[/] {timestamp} {event.source}.{event.kind} {message}{meta_text}"
        self._log.write(line)

    @staticmethod
    def _format_meta(meta: dict[str, object]) -> str:
        parts = []
        for key, value in meta.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return " ".join(parts)


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
        Binding(key="slash", action="command", description="Command", key_display="/"),
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
    _dialog_log: DialogLog | None = None
    _status_panel: StatusPanel | None = None

    def compose(self) -> ComposeResult:
        """
        Compose the user interface layout for the GladosUI application.

        This method generates the primary UI components, including a header, a dialog area, status panels,
        a system log, and a command bar. The layout is structured to display:
        - A header with a clock
        - A body containing:
          - Dialog log (user/assistant messages)
          - System log
          - Status and hints panels
        - A command input bar
        - A footer

        Returns:
            ComposeResult: A generator yielding Textual UI components for rendering
        """
        # It would be nice to have the date in the header, but see:
        # https://github.com/Textualize/textual/issues/4666
        yield Header(show_clock=True)

        with Container(id="body"):
            with Horizontal():
                with Vertical(id="left_panel"):
                    yield Label("Dialog", id="dialog_title")
                    yield DialogLog(id="dialog_log")
                    yield Label("System Log", id="system_title")
                    yield Printer(id="log_area")
                with Vertical(id="right_panel"):
                    yield Label("Status", id="status_title")
                    yield StatusPanel(id="status_panel")
                    yield Label("Hints", id="tips_title")
                    yield Static("Type /help for commands\nUse /observe to open observability", id="tips_panel")

        yield Container(Input(placeholder="/help for commands", id="command_input"), id="command_bar")

        yield Footer()

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
        self._bind_panels()
        self.set_interval(0.3, self._refresh_panels)

    def on_unmount(self) -> None:
        """
        Called when the app is quitting.

        Makes sure that the GLaDOS engine is gracefully shut down.
        """
        logger.info("Quit action initiated in TUI.")
        if hasattr(self, "glados_engine_instance") and self.glados_engine_instance is not None:
            logger.info("Signalling GLaDOS engine to stop...")
            self.glados_engine_instance.shutdown_event.set()

    def action_help(self) -> None:
        """Someone pressed the help key!."""
        self.push_screen(HelpScreen(id="help_screen"))

    def action_command(self) -> None:
        """Focus the command input."""
        command_input = self.query_one("#command_input", Input)
        command_input.value = "/"
        command_input.focus()

    def action_observability(self) -> None:
        """Open the observability screen."""
        self.push_screen(ObservabilityScreen(id="observability_screen"))

    # def on_key(self, event: events.Key) -> None:
    #     """Useful for debugging via key presses."""
    #     logger.success(f"Key pressed: {self.glados_worker}")

    def on_worker_state_changed(self, message: Worker.StateChanged) -> None:
        """Handle messages from workers."""

        if message.state == WorkerState.SUCCESS:
            self.notify("AI Engine operational", title="GLaDOS", timeout=2)
        elif message.state == WorkerState.ERROR:
            self.notify("Instantiation failed!", severity="error")

        self.instantiation_worker = None  # Clear the worker reference

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "command_input":
            return
        command = event.value.strip()
        event.input.value = ""
        if not command:
            return
        if not command.startswith("/"):
            command = f"/{command}"
        if command in {"/observe", "/observability"}:
            self.action_observability()
            return
        if not self.glados_engine_instance:
            self.notify("Engine not ready.", severity="warning")
            return
        response = self.glados_engine_instance.handle_command(command)
        logger.info("TUI command: %s -> %s", command, response)
        self.notify(response, title="Command", timeout=4)

    def on_key(self, event: events.Key) -> None:
        if event.key != "/":
            return
        if isinstance(self.focused, Input) and self.focused.id == "command_input":
            return
        self.action_command()
        event.stop()

    def _refresh_panels(self) -> None:
        if not self._bind_panels():
            return
        engine = self.glados_engine_instance
        if not engine or self._dialog_log is None or self._status_panel is None:
            return
        events = engine.observability_bus.snapshot(limit=200)
        self._dialog_log.refresh_from_bus(events)
        self._status_panel.render_status(self)

    def _bind_panels(self) -> bool:
        if self._dialog_log is not None and self._status_panel is not None:
            return True
        try:
            self._dialog_log = self.query_one("#dialog_log", DialogLog)
            self._status_panel = self.query_one("#status_panel", StatusPanel)
            return True
        except NoMatches:
            return False

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
                self.glados_worker = self.run_worker(self.glados_engine_instance.run, exclusive=True, thread=True)
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
        app: GladosUI | None = None  # Initialize app to None
        try:
            app = cls()
            app.run()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user. Exiting.")
            if app is not None:
                app.exit()
            # No explicit sys.exit(0) here; Textual's app.exit() will handle it.
        except Exception:
            logger.opt(exception=True).critical("Unhandled exception in app run:")
            if app is not None:
                # Attempt a graceful shutdown even on other exceptions
                logger.info("Attempting graceful shutdown due to unhandled exception...")
                app.exit()
            sys.exit(1)  # Exit with error for unhandled exceptions


if __name__ == "__main__":
    GladosUI.run_app()
