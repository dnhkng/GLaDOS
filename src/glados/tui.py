from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
import math
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import ClassVar, cast, Iterable
from urllib.parse import urlparse

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.command import Provider, Hit, Hits, DiscoveryHit
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Input, Label, OptionList, RichLog, Static
from textual.worker import Worker, WorkerState

from glados.core.engine import Glados, GladosConfig
from glados.glados_ui.text_resources import shortcuts_text, welcome_tips
from glados.observability import ObservabilityEvent
from glados.utils.resources import resource_path

# Command Provider for Textual's built-in command palette


class GladosCommands(Provider):
    """Command provider for GLaDOS TUI command palette."""

    # Commands that duplicate TUI features or are irrelevant in TUI
    _HIDDEN_COMMANDS = {"help", "observe", "slots"}

    # Display names for commands (command_name -> display_name)
    _DISPLAY_NAMES: ClassVar[dict[str, str]] = {
        "tts": "Text-to-Speech",
        "asr": "Speech Recognition",
        "mcp": "MCP Servers",
    }

    @property
    def _app(self) -> "GladosUI":
        return cast("GladosUI", self.app)

    def _is_command_available(self, name: str) -> bool:
        """Check if a command should be shown based on engine state."""
        engine = self._app.glados_engine_instance
        if not engine:
            return False
        if name in self._HIDDEN_COMMANDS:
            return False
        # Hide commands for disabled features
        if name == "vision" and engine.vision_config is None:
            return False
        if name == "mcp" and engine.mcp_manager is None:
            return False
        if name == "emotion" and not getattr(engine, "_emotion_agent", None):
            return False
        if name == "agents" and not getattr(engine, "subagent_manager", None):
            return False
        return True

    def _get_display_name(self, name: str) -> str:
        """Get display name for a command."""
        return self._DISPLAY_NAMES.get(name, name.title())

    async def search(self, query: str) -> Hits:
        """Search for commands matching the query."""
        matcher = self.matcher(query)
        app = self._app

        # TUI commands - use partial for reliable binding
        tui_commands = [
            ("Theme", "Switch TUI theme", partial(app.action_theme_picker)),
            ("Context", "Show autonomy slot context", partial(app.action_context)),
            ("Messages", "Show dialog history", partial(app.action_messages)),
            ("Observability", "Open observability screen", partial(app.action_observability)),
            ("Help", "Show keyboard shortcuts", partial(app.action_help)),
        ]

        for name, desc, callback in tui_commands:
            if (score := matcher.match(name)) > 0:
                yield Hit(score, matcher.highlight(name), callback, help=desc)

        # Engine commands (dynamic)
        if app.glados_engine_instance:
            for spec in app.glados_engine_instance.command_specs():
                if not self._is_command_available(spec.name):
                    continue
                display_name = self._get_display_name(spec.name)
                if (score := matcher.match(spec.name)) > 0 or (score := matcher.match(display_name)) > 0:
                    # Handle special commands
                    if spec.name == "quit":
                        callback = partial(app.exit)
                    elif spec.usage and "on|off" in spec.usage:
                        callback = partial(app._open_toggle_picker, spec.name)
                    else:
                        callback = partial(app._run_engine_command, spec.name)
                    yield Hit(
                        score,
                        matcher.highlight(display_name),
                        callback,
                        help=spec.description,
                    )

    async def discover(self) -> Hits:
        """Show all commands when palette first opens."""
        app = self._app

        # TUI commands - use partial for reliable binding
        tui_commands = [
            ("Theme", "Switch TUI theme", partial(app.action_theme_picker)),
            ("Context", "Show autonomy slot context", partial(app.action_context)),
            ("Messages", "Show dialog history", partial(app.action_messages)),
            ("Observability", "Open observability screen", partial(app.action_observability)),
            ("Help", "Show keyboard shortcuts", partial(app.action_help)),
        ]

        for name, desc, callback in tui_commands:
            yield DiscoveryHit(name, callback, help=desc)

        # Engine commands (dynamic)
        if app.glados_engine_instance:
            for spec in app.glados_engine_instance.command_specs():
                if not self._is_command_available(spec.name):
                    continue
                display_name = self._get_display_name(spec.name)
                # Handle special commands
                if spec.name == "quit":
                    callback = partial(app.exit)
                elif spec.usage and "on|off" in spec.usage:
                    callback = partial(app._open_toggle_picker, spec.name)
                else:
                    callback = partial(app._run_engine_command, spec.name)
                yield DiscoveryHit(
                    display_name,
                    callback,
                    help=spec.description,
                )


# Custom Widgets


class Printer(RichLog):
    """A subclass of textual's RichLog which captures and displays all print calls."""

    can_focus = False
    _ignored_prefixes = (
        "Last login:",
        "Welcome to Ubuntu",
        " * Documentation:",
        " * Management:",
        " * Support:",
        "Expanded Security Maintenance",
        "To see these additional updates run:",
        "Learn more about enabling ESM Apps",
        "Your user’s .npmrc file",
        "Run `nvm use --delete-prefix",
        "(base)",
        "Assistant:",
        "Text input:",
    )

    def on_mount(self) -> None:
        self.wrap = True
        self.markup = True
        self.begin_capture_print()

    def on_print(self, event: events.Print) -> None:
        if (text := event.text) != "\n":
            stripped = text.strip()
            cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stripped)
            if any(cleaned.startswith(prefix) for prefix in self._ignored_prefixes):
                return
            if (
                "Last login:" in cleaned
                or "Welcome to Ubuntu" in cleaned
                or " | Assistant:" in cleaned
                or " | Text input:" in cleaned
            ):
                return
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
        vad_indicator = "[bold green]●[/]" if snapshot.vad_active else "[dim]○[/]"
        speaking_indicator = "[bold green]●[/]" if engine.currently_speaking_event.is_set() else "[dim]○[/]"
        autonomy = "ON" if engine.autonomy_config.enabled else "OFF"
        jobs = "ON" if engine.autonomy_config.jobs.enabled else "OFF"
        vision = "ON" if engine.vision_config is not None else "OFF"
        asr = "MUTED" if engine.asr_muted_event.is_set() else "ACTIVE"
        tts = "MUTED" if engine.tts_muted_event.is_set() else "ACTIVE"
        lines = [
            f"ASR: {asr}  TTS: {tts}",
            f"Autonomy: {autonomy}  Jobs: {jobs}",
            f"Vision: {vision}",
            f"Speaking: {speaking_indicator}",
            f"Microphone: {vad_indicator} {rms_db:5.1f} dB",
        ]
        self.update("\n".join(lines))


class QueuePanel(Static):
    can_focus = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.markup = True

    @staticmethod
    def _format_wait(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value * 1000:.0f}ms"

    def render_queue(self, app: "GladosUI") -> None:
        engine = app.glados_engine_instance
        if not engine:
            self.update("Queues: starting...")
            return
        priority_depth = engine.llm_queue_priority.qsize()
        autonomy_depth = engine.llm_queue_autonomy.qsize()
        priority_metrics = app.queue_metrics.get("priority", {})
        autonomy_metrics = app.queue_metrics.get("autonomy", {})
        priority_wait = self._format_wait(priority_metrics.get("wait_s"))
        autonomy_wait = self._format_wait(autonomy_metrics.get("wait_s"))
        lines = [
            f"Priority: {priority_depth} queued  wait {priority_wait}",
            f"Autonomy: {autonomy_depth} queued  wait {autonomy_wait}",
        ]
        self.update("\n".join(lines))


class AutonomyPanel(Static):
    can_focus = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.markup = True

    def render_autonomy(self, app: "GladosUI") -> None:
        engine = app.glados_engine_instance
        if not engine or not engine.autonomy_config:
            self.update("Autonomy: unavailable")
            return
        enabled = "[green]ON[/]" if engine.autonomy_config.enabled else "[red]OFF[/]"
        coalesce = "ON" if engine.autonomy_config.coalesce_ticks else "OFF"
        workers = engine.autonomy_config.autonomy_parallel_calls if engine.autonomy_config.enabled else 0
        inflight = engine.autonomy_inflight()
        queue_depth = engine.llm_queue_autonomy.qsize()
        jobs = "ON" if engine.autonomy_config.jobs.enabled else "OFF"
        lines = [
            f"Enabled: {enabled}",
            f"Workers: {workers}  In-flight: {inflight}",
            f"Queue: {queue_depth}  Jobs: {jobs}",
            f"Coalesce ticks: {coalesce}",
        ]
        self.update("\n".join(lines))


class MCPPanel(Static):
    can_focus = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.markup = True

    def render_mcp(self, app: "GladosUI") -> None:
        engine = app.glados_engine_instance
        if not engine or not engine.mcp_manager:
            self.update("MCP: disabled")
            return
        entries = engine.mcp_manager.status_snapshot()
        if not entries:
            self.update("MCP: no servers")
            return
        lines = []
        for entry in entries[:6]:
            status = "[green]online[/]" if entry["connected"] else "[red]offline[/]"
            lines.append(f"{entry['name']}: {status}  tools={entry['tools']}")
        self.update("\n".join(lines))


class CameraPanel(Static):
    """Displays ASCII camera feed from the vision system."""

    can_focus = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.markup = False  # Raw text, no markup processing

    def render_camera(self, app: "GladosUI") -> None:
        engine = app.glados_engine_instance
        if not engine or not engine.ascii_camera:
            self.update("Camera: unavailable")
            return
        if not engine.ascii_camera.enabled:
            self.update("Camera: disabled (Ctrl+V to enable)")
            return
        frame = engine.ascii_camera.get_frame()
        if frame:
            self.update(frame)
        else:
            self.update("Camera: no signal")


# Screens
class SplashScreen(Screen[None]):
    """Splash screen shown on startup."""

    try:
        with open(Path("src/glados/glados_ui/images/splash.ansi"), encoding="utf-8") as f:
            WELCOME_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")
    except FileNotFoundError:
        logger.error("Logo ANSI art file not found. Using placeholder.")
        WELCOME_ANSI = Text.from_markup("[bold red]Logo ANSI Art Missing[/bold red]")

    def __init__(self) -> None:
        super().__init__()
        self._ready = False

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
        yield Static("Press Ctrl+P for commands, or just ask a question.", id="welcome_prompt")
        yield Static("Initializing systems...", id="welcome_cta")

    def on_mount(self) -> None:
        dialog = self.query_one("#welcome_dialog", Container)
        dialog.border_title = GladosUI.TITLE
        dialog.border_title_align = "center"
        self._load_welcome_meta()
        app = cast(GladosUI, self.app)
        if app.glados_engine_instance is not None:
            self.set_ready()

    def set_ready(self) -> None:
        self._ready = True
        try:
            cta = self.query_one("#welcome_cta", Static)
        except NoMatches:
            return
        cta.update("Press any key to start.")

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
        if not self._ready:
            return
        # Just dismiss - start_glados() is called by on_worker_state_changed
        self.dismiss()
        app.focus_command_input()


class HelpScreen(ModalScreen[None]):
    """Shortcut and keybinding help screen."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Shortcuts"

    def compose(self) -> ComposeResult:
        yield Container(VerticalScroll(Static("", id="help_text")), id="help_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#help_dialog")
        dialog.border_title = self.TITLE
        # Consistent use of explicit closing tag for blink
        dialog.border_subtitle = "Press Esc to close"
        self._render_help()

    def _render_help(self) -> None:
        lines = [
            "[bold]Keyboard Shortcuts[/]",
            "",
            "F1      Help (this screen)",
            "^p      Command palette",
            "",
            "^d      Toggle dialog panel",
            "^l      Toggle system logs",
            "^s      Toggle status panel",
            "^a      Toggle autonomy panel",
            "^u      Toggle queue panel",
            "^m      Toggle MCP panel",
            "^i      Toggle info panels (right side)",
            "^r      Restore all panels",
            "",
            "Esc     Close dialogs",
            "",
            "[bold]Commands[/]",
            "Use ^p to access all commands including:",
            "  Theme, Context, Messages, Observability, Help",
            "  Mute/Unmute ASR/TTS, Reset, Autonomy, etc.",
        ]
        self.query_one("#help_text", Static).update("\n".join(lines))


class ContextScreen(ModalScreen[None]):
    """Display autonomy slot context."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Context Slots"

    def compose(self) -> ComposeResult:
        yield Container(VerticalScroll(Static("", id="context_text")), id="context_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#context_dialog")
        dialog.border_title = self.TITLE
        dialog.border_title_align = "center"
        dialog.border_subtitle = "Press Esc to close"
        self._render_context()

    def _render_context(self) -> None:
        app = cast(GladosUI, self.app)
        engine = app.glados_engine_instance
        if not engine or not engine.autonomy_slots:
            content = "No slots available."
        else:
            slots = engine.autonomy_slots.list_slots()
            if not slots:
                content = "No slots available."
            else:
                lines = []
                for slot in slots:
                    summary = slot.summary.strip()
                    summary_text = f" - {summary}" if summary else ""
                    lines.append(f"- {slot.title}: {slot.status}{summary_text}")
                content = "\n".join(lines)
        self.query_one("#context_text", Static).update(content)


class MessagesScreen(ModalScreen[None]):
    """Display dialog history as a scrollable list."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Dialog Messages"

    def compose(self) -> ComposeResult:
        yield Container(VerticalScroll(Static("", id="messages_text")), id="messages_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#messages_dialog")
        dialog.border_title = self.TITLE
        dialog.border_title_align = "center"
        dialog.border_subtitle = "Press Esc to close"
        self._render_messages()

    def _render_messages(self) -> None:
        app = cast(GladosUI, self.app)
        engine = app.glados_engine_instance
        if not engine:
            content = "Dialog unavailable."
        else:
            events = engine.observability_bus.snapshot(limit=500)
            lines: list[str] = []
            for event in events:
                if event.kind == "user_input" and event.source in {"asr", "text"}:
                    timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                    lines.append(f"[{timestamp}] You: {event.message}")
                elif event.source == "tts" and event.kind == "play":
                    timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                    lines.append(f"[{timestamp}] GLaDOS: {event.message}")
            content = "\n".join(lines) if lines else "No dialog yet."
        self.query_one("#messages_text", Static).update(content)


class InfoScreen(ModalScreen[None]):
    """Display command output in a scrollable dialog."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    def __init__(self, title: str, content: str) -> None:
        super().__init__()
        self._title = title
        self._content = content

    def compose(self) -> ComposeResult:
        yield Container(VerticalScroll(Static(self._content, id="info_text")), id="info_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#info_dialog")
        dialog.border_title = self._title
        dialog.border_title_align = "center"
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


class OnOffPickerScreen(ModalScreen[None]):
    """Picker for on/off command options."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    # Display names for toggle commands
    _DISPLAY_NAMES: ClassVar[dict[str, str]] = {
        "tts": "Text-to-Speech",
        "asr": "Speech Recognition",
    }

    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command

    def compose(self) -> ComposeResult:
        with Container(id="toggle_dialog"):
            yield Label("Select mode", id="toggle_title")
            yield OptionList(id="toggle_list")
            yield Static("Enter to select • Esc to cancel", id="toggle_hint")

    def on_mount(self) -> None:
        dialog = self.query_one("#toggle_dialog")
        dialog.border_title = self._DISPLAY_NAMES.get(self._command, self._command.title())
        dialog.border_title_align = "center"
        option_list = self.query_one("#toggle_list", OptionList)
        option_list.clear_options()
        option_list.add_options(["on", "off"])
        # Highlight current state
        current_on = self._get_current_state()
        option_list.highlighted = 0 if current_on else 1

    def _get_current_state(self) -> bool:
        """Get current state for toggle command (True = on/active)."""
        app = cast(GladosUI, self.app)
        engine = app.glados_engine_instance
        if not engine:
            return False
        if self._command == "asr":
            return not engine.asr_muted_event.is_set()
        if self._command == "tts":
            return not engine.tts_muted_event.is_set()
        if self._command == "autonomy":
            return engine.autonomy_config.enabled
        return False

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        app = cast(GladosUI, self.app)
        prompt = message.option.prompt
        choice = prompt.plain if hasattr(prompt, "plain") else str(prompt)
        if not app.glados_engine_instance:
            app.notify("Engine not ready.", severity="warning")
            self.dismiss()
            return
        command = f"/{self._command} {choice}"
        response = app.glados_engine_instance.handle_command(command)
        logger.success("TUI command: {} -> {}", command, response)
        app.notify(response, title="Command", timeout=4)
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
        priority_q = engine.llm_queue_priority.qsize()
        autonomy_q = engine.llm_queue_autonomy.qsize()
        inflight = engine.autonomy_inflight()
        coalesce = "ON" if engine.autonomy_config.coalesce_ticks else "OFF"
        if engine.mcp_manager:
            snapshot = engine.mcp_manager.status_snapshot()
            connected = sum(1 for entry in snapshot if entry["connected"])
            total = len(snapshot)
        else:
            connected = 0
            total = 0
        self._status.update(
            " | ".join(
                [
                    f"slots: {len(slots)}",
                    f"minds: {len(minds)}",
                    f"queue p:{priority_q} a:{autonomy_q}",
                    f"inflight: {inflight}",
                    f"mcp: {connected}/{total}",
                    f"coalesce: {coalesce}",
                ]
            )
        )

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
        Binding("tab", "toggle_info_panels", "Toggle info panels", priority=True),
    ]
    CSS_PATH = "glados_ui/glados.tcss"
    COMMANDS = {GladosCommands}

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
    _queue_panel: QueuePanel | None = None
    _autonomy_panel: AutonomyPanel | None = None
    _mcp_panel: MCPPanel | None = None
    _camera_panel: CameraPanel | None = None
    _queue_metrics: dict[str, dict[str, float | int | None]]
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
        self._queue_metrics = {}

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
        yield Header(show_clock=True, classes="-tall")

        with Container(id="body"):
            with Horizontal():
                with Vertical(id="left_panel"):
                    yield Label("[u]D[/u]ialog", id="dialog_title")
                    yield DialogLog(id="dialog_log")
                    yield Label("System [u]L[/u]og", id="system_title")
                    yield Printer(id="log_area")
                with Vertical(id="right_panel"):
                    yield Label("[u]V[/u]ision", id="camera_title")
                    yield CameraPanel(id="camera_panel")
                    yield Label("[u]S[/u]tatus", id="status_title")
                    yield StatusPanel(id="status_panel")
                    yield Label("[u]A[/u]utonomy", id="autonomy_title")
                    yield AutonomyPanel(id="autonomy_panel")
                    yield Label("Q[u]u[/u]eues", id="queue_title")
                    yield QueuePanel(id="queue_panel")
                    yield Label("[u]M[/u]CP", id="mcp_title")
                    yield MCPPanel(id="mcp_panel")

        with Container(id="command_bar"):
            yield Input(
                placeholder="Type a message...",
                id="command_input",
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
        logger.remove()
        self._apply_theme(self._resolve_theme())

        self.instantiation_worker = None  # Reset the instantiation worker reference
        self.start_instantiation()

        # Log to TUI via print (captured by Printer widget)
        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}"
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
        self._bind_panels()
        self.set_interval(0.3, self._refresh_panels)
        self.focus_command_input()

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

    def display_input_mode(self) -> str | None:
        if not self._input_mode_override:
            return None
        if self._input_mode_override == "both":
            return "text+audio (TUI)"
        if self._input_mode_override == "text":
            return "text (TUI)"
        return self._input_mode_override

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        """Filter out duplicate Help command from system commands."""
        for cmd in super().get_system_commands(screen):
            # Skip the default help command since we have our own
            if cmd.title.lower() == "help":
                continue
            yield cmd

    def action_help(self) -> None:
        """Someone pressed the help key!."""
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen())

    def action_context(self) -> None:
        """Open the context slots screen."""
        if not isinstance(self.screen, ContextScreen):
            self.push_screen(ContextScreen())

    def action_messages(self) -> None:
        """Open the dialog messages screen."""
        if not isinstance(self.screen, MessagesScreen):
            self.push_screen(MessagesScreen())

    def action_theme_picker(self) -> None:
        """Open the theme picker modal."""
        if not isinstance(self.screen, ThemePickerScreen):
            self.push_screen(ThemePickerScreen())

    def action_change_theme(self) -> None:
        """Override Textual's default theme picker with our custom themes."""
        self.action_theme_picker()

    def action_observability(self) -> None:
        """Open the observability screen."""
        if not isinstance(self.screen, ObservabilityScreen):
            self.push_screen(ObservabilityScreen())

    def action_toggle_info_panels(self) -> None:
        """Toggle the right info panel (Ctrl+I / Tab)."""
        self._toggle_right_panel()

    def _run_engine_command(self, name: str) -> None:
        """Execute an engine command and display the result."""
        if not self.glados_engine_instance:
            self.notify("Engine not ready.", severity="warning")
            return
        response = self.glados_engine_instance.handle_command(f"/{name}")
        if response:
            self.push_screen(InfoScreen(name.title(), response))

    def _open_toggle_picker(self, command_name: str) -> None:
        """Open the on/off picker for a toggle command."""
        self.push_screen(OnOffPickerScreen(command_name))

    # def on_key(self, event: events.Key) -> None:
    #     """Useful for debugging via key presses."""
    #     logger.success(f"Key pressed: {self.glados_worker}")

    def on_worker_state_changed(self, message: Worker.StateChanged) -> None:
        """Handle messages from workers."""

        if message.state == WorkerState.SUCCESS:
            self.notify("AI Engine operational", title="GLaDOS", timeout=2)
            if isinstance(self.screen, SplashScreen):
                self.screen.dismiss()
            # Start the engine's run() method to begin audio listening
            if self.glados_engine_instance is not None:
                self.glados_engine_instance.play_announcement()
                self.start_glados()
            self.focus_command_input()
        elif message.state == WorkerState.ERROR:
            worker = message.worker
            error_msg = str(worker.error) if worker.error else "Unknown error"
            logger.error(f"Worker failed: {error_msg}")
            self.notify(f"Engine error: {error_msg}", severity="error", timeout=10)

        self.instantiation_worker = None  # Clear the worker reference

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "command_input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if not self.glados_engine_instance:
            self.notify("Engine not ready.", severity="warning")
            return
        if not self.glados_engine_instance.submit_text_input(text):
            self.notify("No text submitted.", severity="warning")

    def on_key(self, event: events.Key) -> None:
        # Control-key shortcuts (work globally, even when typing)
        key = event.key
        if key == "f1":
            self.action_help()
            event.stop()
            return
        if key == "ctrl+d":
            self._toggle_panel("dialog_log", "dialog_title")
            event.stop()
            return
        if key == "ctrl+l":
            self._toggle_panel("log_area", "system_title")
            event.stop()
            return
        if key == "ctrl+s":
            self._toggle_panel("status_panel", "status_title")
            event.stop()
            return
        if key == "ctrl+a":
            self._toggle_panel("autonomy_panel", "autonomy_title")
            event.stop()
            return
        if key == "ctrl+m":
            self._toggle_panel("mcp_panel", "mcp_title")
            event.stop()
            return
        if key == "ctrl+u":
            self._toggle_panel("queue_panel", "queue_title")
            event.stop()
            return
        if key == "ctrl+v":
            self._toggle_camera_panel()
            event.stop()
            return
        if key == "ctrl+r":
            self._restore_all_panels()
            event.stop()
            return
        # Note: ctrl+i/tab handled via BINDINGS with priority=True

    def _refresh_panels(self) -> None:
        if not self._bind_panels():
            return
        engine = self.glados_engine_instance
        if not engine or self._dialog_log is None or self._status_panel is None:
            return
        events = engine.observability_bus.snapshot(limit=200)
        self._dialog_log.refresh_from_bus(events)
        self._update_queue_metrics(events)
        self._status_panel.render_status(self)
        if self._queue_panel:
            self._queue_panel.render_queue(self)
        if self._autonomy_panel:
            self._autonomy_panel.render_autonomy(self)
        if self._mcp_panel:
            self._mcp_panel.render_mcp(self)
        if self._camera_panel:
            self._camera_panel.render_camera(self)

    @property
    def queue_metrics(self) -> dict[str, dict[str, float | int | None]]:
        return self._queue_metrics

    def _update_queue_metrics(self, events: list[ObservabilityEvent]) -> None:
        for event in events:
            if event.source != "llm" or event.kind != "queue":
                continue
            lane = str(event.meta.get("lane", ""))
            if not lane:
                continue
            self._queue_metrics[lane] = {
                "wait_s": event.meta.get("wait_s"),
                "queue_depth": event.meta.get("queue_depth"),
            }

    def _bind_panels(self) -> bool:
        if (
            self._dialog_log is not None
            and self._status_panel is not None
            and self._queue_panel is not None
            and self._autonomy_panel is not None
            and self._mcp_panel is not None
            and self._camera_panel is not None
        ):
            return True
        try:
            self._dialog_log = self.query_one("#dialog_log", DialogLog)
            self._status_panel = self.query_one("#status_panel", StatusPanel)
            self._queue_panel = self.query_one("#queue_panel", QueuePanel)
            self._autonomy_panel = self.query_one("#autonomy_panel", AutonomyPanel)
            self._mcp_panel = self.query_one("#mcp_panel", MCPPanel)
            self._camera_panel = self.query_one("#camera_panel", CameraPanel)
            return True
        except NoMatches:
            return False

    def focus_command_input(self) -> None:
        try:
            command_input = self.query_one("#command_input", Input)
        except NoMatches:
            return
        command_input.focus()

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

    def _toggle_panel(self, panel_id: str, title_id: str | None = None) -> None:
        try:
            panel = self.query_one(f"#{panel_id}")
            # Toggle using display property directly (more reliable than CSS classes)
            new_display = "block" if panel.styles.display == "none" else "none"
            panel.styles.display = new_display
            if title_id:
                self.query_one(f"#{title_id}").styles.display = new_display
            # Force refresh of sibling widget to trigger proper redraw
            if panel_id == "log_area" and self._dialog_log:
                self._dialog_log.refresh()
            elif panel_id == "dialog_log":
                self.query_one("#log_area").refresh()
        except NoMatches:
            pass

    def _toggle_camera_panel(self) -> None:
        """Toggle camera panel and enable/disable the camera feed."""
        self._toggle_panel("camera_panel", "camera_title")
        # Enable/disable camera feed to save CPU when hidden
        engine = self.glados_engine_instance
        if engine and engine.ascii_camera:
            try:
                panel = self.query_one("#camera_panel")
                if panel.styles.display == "none":
                    engine.ascii_camera.disable()
                else:
                    engine.ascii_camera.enable()
            except NoMatches:
                pass

    def _restore_all_panels(self) -> None:
        """Restore all hidden panels to visible."""
        panel_ids = [
            ("dialog_log", "dialog_title"),
            ("log_area", "system_title"),
            ("camera_panel", "camera_title"),
            ("status_panel", "status_title"),
            ("autonomy_panel", "autonomy_title"),
            ("queue_panel", "queue_title"),
            ("mcp_panel", "mcp_title"),
        ]
        for panel_id, title_id in panel_ids:
            try:
                self.query_one(f"#{panel_id}").styles.display = "block"
                self.query_one(f"#{title_id}").styles.display = "block"
            except NoMatches:
                pass
        # Also restore right panel container and left panel width
        try:
            self.query_one("#right_panel").styles.display = "block"
            self.query_one("#left_panel").styles.width = "70%"
        except NoMatches:
            pass
        # Trigger redraw
        if self._dialog_log:
            self._dialog_log.refresh()

    def _toggle_right_panel(self) -> None:
        """Toggle the entire right panel (Status, Autonomy, Queues, MCP)."""
        try:
            right_panel = self.query_one("#right_panel")
            left_panel = self.query_one("#left_panel")
            # Check if currently hidden (display is "none")
            is_hidden = str(right_panel.styles.display) == "none"
            if is_hidden:
                # Show right panel, restore left panel width
                right_panel.styles.display = "block"
                left_panel.styles.width = "70%"
            else:
                # Hide right panel, expand left panel
                right_panel.styles.display = "none"
                left_panel.styles.width = "100%"
            # Trigger redraw
            if self._dialog_log:
                self._dialog_log.refresh()
        except NoMatches:
            pass

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
            if self._input_mode_override in {"text", "both"}:
                # Avoid stdin contention between Textual and TextListener.
                updates["input_mode"] = "audio"
            else:
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
