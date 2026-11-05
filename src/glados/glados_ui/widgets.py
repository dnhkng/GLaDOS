"""TUI widgets: the log panes, the typewriter and the status panels."""

from collections.abc import Iterator
from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING

from textual import events
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import RichLog, Static

from glados.observability import ObservabilityEvent

if TYPE_CHECKING:
    from glados.tui import GladosUI


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
        "Your user\u2019s .npmrc file",  # Typographic apostrophe, as nvm prints it.
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

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, wrap=True, markup=True)
        self._last_timestamp = 0.0

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

    @staticmethod
    def _volume_bar(level: float, width: int = 18) -> str:
        level = max(0.0, min(1.0, level))
        filled = round(level * width)
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

    def render_autonomy(self, app: "GladosUI") -> None:
        engine = app.glados_engine_instance
        if not engine or not engine.autonomy_config:
            self.update("Autonomy: unavailable")
            return
        enabled = "[green]ON[/]" if engine.autonomy_config.enabled else "[red]OFF[/]"
        workers = engine.autonomy_config.autonomy_parallel_calls if engine.autonomy_config.enabled else 0
        inflight = engine.autonomy_inflight()
        queue_depth = engine.llm_queue_autonomy.qsize()
        jobs = "ON" if engine.autonomy_config.jobs.enabled else "OFF"
        lines = [
            f"Enabled: {enabled}",
            f"Workers: {workers}  In-flight: {inflight}",
            f"Queue: {queue_depth}  Jobs: {jobs}",
        ]
        self.update("\n".join(lines))


class MCPPanel(Static):
    can_focus = False

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
