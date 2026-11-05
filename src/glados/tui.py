from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
import sys
from typing import ClassVar, cast

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label
from textual.worker import Worker, WorkerState
import yaml

from glados.core.config import GladosConfig
from glados.core.engine import Glados
from glados.glados_ui.screens import (
    ContextScreen,
    HelpScreen,
    InfoScreen,
    MessagesScreen,
    ObservabilityScreen,
    OnOffPickerScreen,
    SplashScreen,
    ThemePickerScreen,
)
from glados.glados_ui.widgets import AutonomyPanel, DialogLog, MCPPanel, Printer, QueuePanel, StatusPanel
from glados.observability import ObservabilityEvent
from glados.utils.resources import resource_path

# Command Provider for Textual's built-in command palette


class GladosCommands(Provider):
    """Command provider for GLaDOS TUI command palette."""

    # Commands that duplicate TUI features or are irrelevant in TUI
    _HIDDEN_COMMANDS: ClassVar[frozenset[str]] = frozenset({"help", "observe", "slots"})

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
        return not (name == "agents" and not getattr(engine, "subagent_manager", None))

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


# The App
class GladosUI(App[None]):
    """The main app class for the GlaDOS ui."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("tab", "toggle_info_panels", "Toggle info panels", priority=True),
    ]
    CSS_PATH = "glados_ui/glados.tcss"
    COMMANDS: ClassVar[set[type[Provider] | Callable[[], type[Provider]]]] = {GladosCommands}

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
        with open(Path(__file__).resolve().parent / "glados_ui/images/logo.ansi", encoding="utf-8") as f:
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
    _queue_metrics: dict[str, dict[str, float | int | None]]
    _config_paths: list[Path]
    _input_mode_override: str | None
    _tts_enabled_override: bool | None
    _asr_muted_override: bool | None
    _theme_override: str | None
    _active_theme: str | None

    def __init__(
        self,
        config_paths: str | Path | list[str] | list[Path] | None = None,
        input_mode: str | None = None,
        tts_enabled: bool | None = None,
        asr_muted: bool | None = None,
        theme: str | None = None,
    ) -> None:
        super().__init__()
        default_config = resource_path("configs/glados_config.yaml")
        if isinstance(config_paths, list):
            # is already list - convert each element to Path
            self._config_paths = [Path(p) for p in config_paths]
        elif config_paths is not None:
            # single element - convert to list[Path]
            self._config_paths = [Path(config_paths)]
        else:
            # None
            self._config_paths = [Path(default_config)]
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

        with Container(id="body"), Horizontal():
            with Vertical(id="left_panel"):
                yield Label("[u]D[/u]ialog", id="dialog_title")
                yield DialogLog(id="dialog_log")
                yield Label("System [u]L[/u]og", id="system_title")
                yield Printer(id="log_area")
            with Vertical(id="right_panel"):
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
        # logger.remove() above also dropped the session log file.
        from .observability.log_file import enable_file_log

        enable_file_log()

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
        theme_name = getattr(self, "_active_theme", None) or getattr(self, "_theme_override", None) or "aperture"
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

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
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
        ):
            return True
        try:
            self._dialog_log = self.query_one("#dialog_log", DialogLog)
            self._status_panel = self.query_one("#status_panel", StatusPanel)
            self._queue_panel = self.query_one("#queue_panel", QueuePanel)
            self._autonomy_panel = self.query_one("#autonomy_panel", AutonomyPanel)
            self._mcp_panel = self.query_one("#mcp_panel", MCPPanel)
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
            config = GladosConfig.from_yaml(self._config_paths)
            if config.tui_theme:
                return config.tui_theme
        except (OSError, ValueError, yaml.YAMLError) as exc:
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

    def _restore_all_panels(self) -> None:
        """Restore all hidden panels to visible."""
        panel_ids = [
            ("dialog_log", "dialog_title"),
            ("log_area", "system_title"),
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
        except Exception as e:  # noqa: BLE001 - reported in the TUI log
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

        config_paths = self._config_paths
        if not all(p.exists() for p in config_paths):
            missing = [p for p in config_paths if not p.exists()]
            msg = f"GLaDOS config file(s) not found: {missing}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        glados_config = GladosConfig.from_yaml(config_paths)
        updates: dict[str, object] = {}
        if self._input_mode_override:
            if self._input_mode_override == "text":
                # In TUI text mode, use "none" so neither audio stream nor TextListener
                # is started. The TUI Input widget handles all text input, avoiding
                # both high CPU from audio and stdin contention with TextListener.
                updates["input_mode"] = "none"
            elif self._input_mode_override == "both":
                # In TUI+both mode, keep audio ASR active; the TUI Input widget handles
                # text input, so TextListener is not needed (avoids stdin contention).
                updates["input_mode"] = "audio"
            else:
                updates["input_mode"] = self._input_mode_override
        elif glados_config.input_mode == "text":
            # Config file sets input_mode: "text" but TUI is running — remap to "none"
            # to prevent TextListener from competing with Textual for stdin, which causes
            # keyboard input to require multiple keystrokes. The TUI Input widget submits
            # text via submit_text_input() instead.
            updates["input_mode"] = "none"
        elif glados_config.input_mode == "both":
            # Config file sets input_mode: "both" but TUI is running — remap to "audio"
            # so audio ASR stays active while TextListener is suppressed to avoid stdin
            # contention with Textual. Text input is handled by the TUI Input widget.
            updates["input_mode"] = "audio"
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
        except Exception:  # noqa: BLE001 - last-resort handler that still attempts a clean shutdown
            logger.opt(exception=True).critical("Unhandled exception in app run:")
            if app is not None:
                # Attempt a graceful shutdown even on other exceptions
                logger.info("Attempting graceful shutdown due to unhandled exception...")
                app.exit()
            sys.exit(1)  # Exit with error for unhandled exceptions


if __name__ == "__main__":
    GladosUI.run_app()
