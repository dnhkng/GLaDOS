from collections.abc import Callable, Iterator
from dataclasses import dataclass
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import ClassVar, cast, Iterable
from urllib.parse import urlparse

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.suggester import Suggester
from textual.widgets import Footer, Header, Input, Label, OptionList, RichLog, Static
from textual.worker import Worker, WorkerState

from glados.core.engine import Glados, GladosConfig
from glados.glados_ui.text_resources import shortcuts_text, welcome_tips
from glados.observability import ObservabilityEvent
from glados.utils.resources import resource_path

# Custom Widgets


class Printer(RichLog):
    """A subclass of textual's RichLog which captures and displays all print calls."""

    can_focus = False

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
    can_focus = False

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
    can_focus = False

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
        tts = "MUTED" if engine.tts_muted_event.is_set() else "ACTIVE"
        lines = [
            f"Input: {engine.input_mode}",
            f"ASR: {asr}",
            f"TTS: {tts}",
            f"Speaking: {speaking}",
            f"Processing: {processing}",
            f"Autonomy: {autonomy}  Jobs: {jobs}",
            f"Vision: {vision}",
            "",
            f"VAD: {vad_text}",
            f"Volume: {bar} {rms_db:5.1f} dB",
        ]
        self.update("\n".join(lines))


class CommandSuggester(Suggester):
    def __init__(self, get_commands: Callable[[], list[str]]) -> None:
        super().__init__(use_cache=False, case_sensitive=False)
        self._get_commands = get_commands

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        commands = self._get_commands()
        if not commands:
            return None
        prefix = value.casefold()
        for command in commands:
            if command.casefold().startswith(prefix):
                return command
        return None


# Screens
class SplashScreen(Screen[None]):
    """Splash screen shown on startup."""

    try:
        with open(Path("src/glados/glados_ui/images/logo.ansi"), encoding="utf-8") as f:
            WELCOME_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")
    except FileNotFoundError:
        logger.error("Logo ANSI art file not found. Using placeholder.")
        WELCOME_ANSI = Text.from_markup("[bold red]Logo ANSI Art Missing[/bold red]")

    def compose(self) -> ComposeResult:
        """
        Compose the layout for the splash screen with a welcome panel and tips.

        Returns:
            ComposeResult: A generator yielding the screen's UI components, including:
                - A welcome panel with logo, metadata, and tips
                - A prompt line and call-to-action
        """
        with Container(id="welcome_dialog"):
            with Horizontal(id="welcome_body"):
                with Vertical(id="welcome_left"):
                    yield Label("Welcome back!", id="welcome_title")
                    yield Static(self.WELCOME_ANSI, id="welcome_logo")
                    yield Static("Model: loading...\nEndpoint: loading...\nPath: loading...", id="welcome_meta")
                with Vertical(id="welcome_right"):
                    yield Label("Tips for getting started", id="welcome_tips_title")
                    yield Static(welcome_tips, id="welcome_tips")
                    yield Label("Recent activity", id="welcome_recent_title")
                    yield Static("No recent activity", id="welcome_recent")
        yield Static('Try "/help" or ask a question.', id="welcome_prompt")
        yield Static("Press any key to start.", id="welcome_cta")

    def on_mount(self) -> None:
        dialog = self.query_one("#welcome_dialog", Container)
        dialog.border_title = GladosUI.TITLE
        dialog.border_title_align = "center"
        self._load_welcome_meta()

    def _load_welcome_meta(self) -> None:
        app = cast(GladosUI, self.app)
        model = "unknown"
        endpoint = "unknown"
        try:
            config = GladosConfig.from_yaml(str(app._config_path))
            model = config.llm_model
            endpoint = self._format_endpoint(str(config.completion_url))
        except Exception as exc:
            logger.warning("Welcome screen failed to load config: {}", exc)
        meta = f"Model: {model}\nEndpoint: {endpoint}\nPath: {Path.cwd()}"
        self.query_one("#welcome_meta", Static).update(meta)

    @staticmethod
    def _format_endpoint(url: str) -> str:
        host = urlparse(url).hostname or url
        if host in {"localhost", "127.0.0.1"}:
            return f"{host} (local)"
        return host

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
        if event.key == "question_mark":
            app.action_help()
            return
        if app.glados_engine_instance:
            app.glados_engine_instance.play_announcement()
            app.start_glados()
            self.dismiss()
            app.focus_command_input()


class HelpScreen(ModalScreen[None]):
    """Shortcut and keybinding help screen."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Shortcuts"

    def compose(self) -> ComposeResult:
        """
        Compose the shortcuts screen layout using a static text block.

        This method generates the visual composition of the help screen, wrapping the shortcuts
        text in a Container for easy scanning.

        Returns:
            ComposeResult: A generator yielding the composed help screen container.
        """
        yield Container(Static(shortcuts_text, id="help_text"), id="help_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#help_dialog")
        dialog.border_title = self.TITLE
        # Consistent use of explicit closing tag for blink
        dialog.border_subtitle = "Press Esc to close"


class ThemePickerScreen(ModalScreen[None]):
    """Theme picker for the command palette and /theme command."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Themes"

    def compose(self) -> ComposeResult:
        with Container(id="theme_dialog"):
            yield Label(self.TITLE, id="theme_title")
            yield OptionList(id="theme_list")
            yield Static("Enter to select • Esc to cancel", id="theme_hint")

    def on_mount(self) -> None:
        dialog = self.query_one("#theme_dialog")
        dialog.border_title = self.TITLE
        dialog.border_title_align = "center"
        option_list = self.query_one("#theme_list", OptionList)
        app = cast(GladosUI, self.app)
        option_list.clear_options()
        option_list.add_options(list(app.THEMES))
        if app._active_theme in app.THEMES:
            option_list.highlighted = app.THEMES.index(app._active_theme)

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        app = cast(GladosUI, self.app)
        prompt = message.option.prompt
        selected = prompt.plain if hasattr(prompt, "plain") else str(prompt)
        app._apply_theme(selected)
        app.notify(f"Theme set to {selected}.", title="Theme", timeout=3)
        self.dismiss()


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

    DEFAULT_TIPS = "Enter to send\n/ for commands, ? for shortcuts"

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

    COMMAND_PALETTE_LIMIT: ClassVar[int] = 6
    THEMES: ClassVar[tuple[str, ...]] = ("aperture", "ice", "matrix", "mono", "ember")
    THEME_VARIABLES: ClassVar[dict[str, dict[str, str]]] = {
        "aperture": {
            "primary": "#ffb000",
            "foreground": "#ffb000",
            "background": "#282828",
            "surface": "#282828",
        },
        "ice": {
            "primary": "#7dd3fc",
            "foreground": "#e0f2fe",
            "background": "#0b1220",
            "surface": "#0f172a",
        },
        "matrix": {
            "primary": "#22c55e",
            "foreground": "#d1fae5",
            "background": "#0a0f0a",
            "surface": "#0f1a10",
        },
        "mono": {
            "primary": "#e5e7eb",
            "foreground": "#f9fafb",
            "background": "#111827",
            "surface": "#0b1020",
        },
        "ember": {
            "primary": "#f97316",
            "foreground": "#fdba74",
            "background": "#1f1308",
            "surface": "#2a1a0b",
        },
    }
    ENABLE_COMMAND_PALETTE = True

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
    _tips_panel: Static | None = None
    _command_palette: OptionList | None = None
    _command_matches: list[tuple[str, str]] = []
    _config_path: Path
    _input_mode_override: str | None
    _tts_enabled_override: bool | None
    _asr_muted_override: bool | None
    _theme_override: str | None
    _active_theme: str | None

    def __init__(
        self,
        config_path: str | Path | None = None,
        input_mode: str | None = None,
        tts_enabled: bool | None = None,
        asr_muted: bool | None = None,
        theme: str | None = None,
    ) -> None:
        super().__init__()
        default_config = resource_path("configs/glados_config.yaml")
        self._config_path = Path(config_path) if config_path else Path(default_config)
        self._input_mode_override = input_mode
        self._tts_enabled_override = tts_enabled
        self._asr_muted_override = asr_muted
        self._theme_override = theme
        self._active_theme = None

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
                    yield Static(self.DEFAULT_TIPS, id="tips_panel")

        with Container(id="command_bar"):
            yield OptionList(id="command_palette", classes="hidden")
            yield Input(
                placeholder="Type a message or /command",
                id="command_input",
                suggester=CommandSuggester(self._command_names),
            )

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
        self._apply_theme(self._resolve_theme())

        self.instantiation_worker = None  # Reset the instantiation worker reference
        self.start_instantiation()

        logger.add(print, format=fmt, level="SUCCESS")

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

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        theme_name = (
            getattr(self, "_active_theme", None)
            or getattr(self, "_theme_override", None)
            or "aperture"
        )
        theme_vars = self.THEME_VARIABLES.get(theme_name, self.THEME_VARIABLES["aperture"])
        variables.update(theme_vars)
        return variables

    def action_help(self) -> None:
        """Someone pressed the help key!."""
        self.push_screen(HelpScreen(id="help_screen"))

    def action_theme_picker(self) -> None:
        """Open the theme picker modal."""
        self.push_screen(ThemePickerScreen(id="theme_picker_screen"))

    def action_change_theme(self) -> None:
        """Override Textual's default theme picker with our custom themes."""
        self.action_theme_picker()

    def action_command(self) -> None:
        """Focus the command input."""
        command_input = self.query_one("#command_input", Input)
        if not command_input.value:
            command_input.value = "/"
            self._update_command_hints(command_input.value)
        command_input.focus()

    def action_observability(self) -> None:
        """Open the observability screen."""
        self.push_screen(ObservabilityScreen(id="observability_screen"))

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield SystemCommand("Choose color scheme", "Switch the TUI theme", self.action_theme_picker)
        yield SystemCommand("Observability", "Open the observability screen", self.action_observability)
        yield SystemCommand("Quit", "Quit the application", self.action_quit)

    # def on_key(self, event: events.Key) -> None:
    #     """Useful for debugging via key presses."""
    #     logger.success(f"Key pressed: {self.glados_worker}")

    def on_worker_state_changed(self, message: Worker.StateChanged) -> None:
        """Handle messages from workers."""

        if message.state == WorkerState.SUCCESS:
            self.notify("AI Engine operational", title="GLaDOS", timeout=2)
            try:
                command_input = self.query_one("#command_input", Input)
            except NoMatches:
                command_input = None
            if command_input is not None:
                self._update_command_hints(command_input.value)
        elif message.state == WorkerState.ERROR:
            self.notify("Instantiation failed!", severity="error")

        self.instantiation_worker = None  # Clear the worker reference

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "command_input":
            return
        command = event.value.strip()
        if not command:
            event.input.value = ""
            return
        if command.startswith("/") and self._palette_visible() and " " not in command:
            selected = self._selected_command()
            if selected and command != selected:
                event.input.value = f"{selected} "
                event.input.cursor_position = len(event.input.value)
                self._update_command_hints(event.input.value)
                return
        event.input.value = ""
        self._update_command_hints("")
        if command.startswith("/"):
            if command in {"/quit", "/exit"}:
                if self.glados_engine_instance:
                    self.glados_engine_instance.handle_command(command)
                self.exit()
                return
            if command in {"/observe", "/observability"}:
                self.action_observability()
                return
            if command.startswith("/theme"):
                parts = command.split()
                if len(parts) == 1:
                    self.action_theme_picker()
                    return
                theme = self._apply_theme(parts[1])
                self.notify(f"Theme set to {theme}.", title="Theme", timeout=4)
                return
            if not self.glados_engine_instance:
                self.notify("Engine not ready.", severity="warning")
                return
            response = self.glados_engine_instance.handle_command(command)
            logger.success("TUI command: {} -> {}", command, response)
            self.notify(response, title="Command", timeout=4)
            return
        if not self.glados_engine_instance:
            self.notify("Engine not ready.", severity="warning")
            return
        if not self.glados_engine_instance.submit_text_input(command):
            self.notify("No text submitted.", severity="warning")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "command_input":
            return
        self._update_command_hints(event.value)

    def on_key(self, event: events.Key) -> None:
        if self._palette_visible() and isinstance(self.focused, Input) and self.focused.id == "command_input":
            if event.key == "up":
                self._move_command_palette(-1)
                event.stop()
                return
            if event.key == "down":
                self._move_command_palette(1)
                event.stop()
                return
            if event.key == "tab":
                if self._complete_command_input(add_space=True):
                    event.stop()
                return
            if event.key == "escape":
                self._hide_command_palette()
                event.stop()
                return
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
        if (
            self._dialog_log is not None
            and self._status_panel is not None
            and self._tips_panel is not None
            and self._command_palette is not None
        ):
            return True
        try:
            self._dialog_log = self.query_one("#dialog_log", DialogLog)
            self._status_panel = self.query_one("#status_panel", StatusPanel)
            self._tips_panel = self.query_one("#tips_panel", Static)
            self._command_palette = self.query_one("#command_palette", OptionList)
            return True
        except NoMatches:
            return False

    def focus_command_input(self) -> None:
        try:
            command_input = self.query_one("#command_input", Input)
        except NoMatches:
            return
        command_input.focus()

    def _command_names(self) -> list[str]:
        engine = self.glados_engine_instance
        names: list[str] = []
        if engine:
            for spec in engine.command_specs():
                names.append(f"/{spec.name}")
                for alias in spec.aliases:
                    names.append(f"/{alias}")
        names.extend(["/observe", "/quit", "/theme"])
        return names

    def _command_entries(self) -> list[tuple[str, str]]:
        engine = self.glados_engine_instance
        entries: list[tuple[str, str]] = []
        if engine:
            for spec in engine.command_specs():
                label = f"/{spec.name}"
                entries.append((label, spec.description))
        entries.extend(
            [
                ("/observe", "Open observability screen"),
                ("/quit", "Quit GLaDOS"),
                ("/theme", "Switch TUI theme"),
            ]
        )
        return entries

    def _update_command_hints(self, value: str) -> None:
        if not self._bind_panels():
            return
        if self._tips_panel is None or self._command_palette is None:
            return
        text = value.strip()
        if not text.startswith("/"):
            self._tips_panel.update(self.DEFAULT_TIPS)
            self._hide_command_palette()
            return
        entries = self._command_entries()
        if not entries:
            self._tips_panel.update("Commands loading...")
            self._hide_command_palette()
            return
        prefix = text.split()[0].casefold()
        matches = [(label, desc) for label, desc in entries if label.casefold().startswith(prefix)]
        if not matches:
            self._tips_panel.update("No matching commands. Try /help.")
            self._hide_command_palette()
            return
        self._show_command_palette(matches)
        self._tips_panel.update("Up/Down to select, Tab to complete, Enter to run.")

    def _resolve_theme(self) -> str:
        if self._theme_override:
            return self._theme_override
        try:
            config = GladosConfig.from_yaml(str(self._config_path))
            if config.tui_theme:
                return config.tui_theme
        except Exception as exc:
            logger.warning("TUI theme load failed: {}", exc)
        return "aperture"

    def _apply_theme(self, theme: str | None) -> str:
        theme_name = (theme or "aperture").strip().casefold()
        if theme_name not in self.THEMES:
            logger.warning("Unknown theme '{}', defaulting to aperture.", theme_name)
            theme_name = "aperture"
        for name in self.THEMES:
            self.remove_class(f"theme-{name}")
        self.add_class(f"theme-{theme_name}")
        self._active_theme = theme_name
        self.refresh_css(animate=False)
        return theme_name

    def _theme_label(self) -> str:
        return self._active_theme or "aperture"

    def _show_command_palette(self, matches: list[tuple[str, str]]) -> None:
        if self._command_palette is None:
            return
        visible_matches = matches[: self.COMMAND_PALETTE_LIMIT]
        self._command_matches = visible_matches
        options: list[Text] = []
        for label, desc in visible_matches:
            prompt = Text.assemble((label, "bold"), ("  ", ""), (desc, "dim"))
            options.append(prompt)
        self._command_palette.clear_options()
        self._command_palette.add_options(options)
        self._command_palette.highlighted = 0 if options else None
        self._command_palette.remove_class("hidden")

    def _hide_command_palette(self) -> None:
        if self._command_palette is None:
            return
        self._command_palette.add_class("hidden")
        self._command_matches = []

    def _palette_visible(self) -> bool:
        return bool(self._command_palette and not self._command_palette.has_class("hidden"))

    def _selected_command(self) -> str | None:
        if not self._command_palette or self._command_palette.option_count == 0:
            return None
        index = self._command_palette.highlighted
        if index is None:
            index = 0
        if index < 0 or index >= len(self._command_matches):
            return None
        return self._command_matches[index][0]

    def _move_command_palette(self, delta: int) -> None:
        if not self._command_palette or self._command_palette.option_count == 0:
            return
        current = self._command_palette.highlighted
        if current is None:
            current = 0
        new_index = (current + delta) % self._command_palette.option_count
        self._command_palette.highlighted = new_index
        self._command_palette.scroll_to_highlight()

    def _complete_command_input(self, add_space: bool = True) -> bool:
        command = self._selected_command()
        if not command:
            return False
        try:
            command_input = self.query_one("#command_input", Input)
        except NoMatches:
            return False
        command_input.value = f"{command} " if add_space else command
        command_input.cursor_position = len(command_input.value)
        self._update_command_hints(command_input.value)
        return True

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

        config_path = self._config_path
        if not config_path.exists():
            logger.error(f"GLaDOS config file not found: {config_path}")

        glados_config = GladosConfig.from_yaml(str(config_path))
        updates: dict[str, object] = {}
        if self._input_mode_override:
            updates["input_mode"] = self._input_mode_override
        if self._tts_enabled_override is not None:
            updates["tts_enabled"] = self._tts_enabled_override
        if self._asr_muted_override is not None:
            updates["asr_muted"] = self._asr_muted_override
        if updates:
            glados_config = glados_config.model_copy(update=updates)
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
