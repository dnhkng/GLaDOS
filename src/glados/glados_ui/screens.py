"""TUI screens: splash, help, context, messages, pickers and observability."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast
from urllib.parse import urlparse

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.widgets import Label, OptionList, RichLog, Static
import yaml

from glados.core.config import GladosConfig
from glados.glados_ui.text_resources import welcome_tips
from glados.observability import ObservabilityBus, ObservabilityEvent

if TYPE_CHECKING:
    from glados.tui import GladosUI


class SplashScreen(Screen[None]):
    """Splash screen shown on startup."""

    try:
        with open(Path(__file__).resolve().parent / "images/splash.ansi", encoding="utf-8") as f:
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
        with Container(id="welcome_dialog"), Horizontal(id="welcome_body"):
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
        dialog.border_title = self.app.TITLE
        dialog.styles.border_title_align = "center"
        self._load_welcome_meta()
        app = cast("GladosUI", self.app)
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
        app = cast("GladosUI", self.app)
        model = "unknown"
        endpoint = "unknown"
        try:
            config = GladosConfig.from_yaml(app._config_paths)
            model = config.llm_model
            endpoint = self._format_endpoint(str(config.completion_url))
        except (OSError, ValueError, yaml.YAMLError) as exc:
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
        app = cast("GladosUI", self.app)  # mypy gets confused about app's type
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
        dialog.styles.border_title_align = "center"
        dialog.border_subtitle = "Press Esc to close"
        self._render_context()

    def _render_context(self) -> None:
        app = cast("GladosUI", self.app)
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
        dialog.styles.border_title_align = "center"
        dialog.border_subtitle = "Press Esc to close"
        self._render_messages()

    def _render_messages(self) -> None:
        app = cast("GladosUI", self.app)
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
        dialog.styles.border_title_align = "center"
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
        dialog.styles.border_title_align = "center"
        option_list = self.query_one("#theme_list", OptionList)
        app = cast("GladosUI", self.app)
        option_list.clear_options()
        option_list.add_options(list(app.THEMES))
        if app._active_theme in app.THEMES:
            option_list.highlighted = app.THEMES.index(app._active_theme)

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        app = cast("GladosUI", self.app)
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
        dialog.styles.border_title_align = "center"
        option_list = self.query_one("#toggle_list", OptionList)
        option_list.clear_options()
        option_list.add_options(["on", "off"])
        # Highlight current state
        current_on = self._get_current_state()
        option_list.highlighted = 0 if current_on else 1

    def _get_current_state(self) -> bool:
        """Get current state for toggle command (True = on/active)."""
        app = cast("GladosUI", self.app)
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
        app = cast("GladosUI", self.app)
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
            yield RichLog(id="observability_log", max_lines=500)
            yield Static("", id="observability_status")

    def on_mount(self) -> None:
        dialog = self.query_one("#observability_dialog")
        dialog.border_title = self.TITLE
        dialog.styles.border_title_align = "center"
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

    def _get_bus(self) -> ObservabilityBus | None:
        app = cast("GladosUI", self.app)
        if not app.glados_engine_instance:
            return None
        return app.glados_engine_instance.observability_bus

    def _update_status(self) -> None:
        app = cast("GladosUI", self.app)
        engine = app.glados_engine_instance
        if not engine:
            self._status.update("Engine not ready.")
            return
        slots = engine.autonomy_slots.list_slots() if engine.autonomy_slots else []
        minds = engine.mind_registry.snapshot() if engine.mind_registry else []
        priority_q = engine.llm_queue_priority.qsize()
        autonomy_q = engine.llm_queue_autonomy.qsize()
        inflight = engine.autonomy_inflight()
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
