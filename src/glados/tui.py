"""GLaDOS TUI - Terminal User Interface for the voice assistant.

This module contains the main Textual application for GLaDOS, maintaining
the original clean left/right split design while adding modular widget support.
"""

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
from textual.widgets import Digits, Footer, Header, Label, Log, RichLog, Static, TextArea
from textual.worker import Worker, WorkerState

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
        """Generates and writes a line of random block characters to the log."""
        # Ensure width calculation doesn't go negative if self.size.width is small
        num_blocks_to_generate = max(0, self.size.width - 8)
        random_blocks = " ".join(random.choice(self.BLOCKS) for _ in range(num_blocks_to_generate))
        self.write_line(f"{random_blocks}")

    def on_show(self) -> None:
        """Set up an interval timer to periodically animate scrolling blocks."""
        self.set_interval(0.18, self._animate_blocks)


class Typewriter(Static):
    """A widget which displays text a character at a time."""

    def __init__(
        self,
        text: str = "_",
        id: str | None = None,
        speed: float = 0.01,  # time between each character
        repeat: bool = False,  # whether to start again at the end
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._text = text
        self.__id_for_child = id
        self._speed = speed
        self._repeat = repeat
        # Flag to determine if we should use Rich markup
        self._use_markup = True
        # Check if text contains special Rich markup characters
        if "[" in text or "]" in text:
            # If there are brackets in the text, disable markup to avoid conflicts
            self._use_markup = False

    def compose(self) -> ComposeResult:
        self._static = Static(markup=self._use_markup)
        self._vertical_scroll = VerticalScroll(self._static, id=self.__id_for_child)
        yield self._vertical_scroll

    def _get_iterator(self) -> Iterator[str]:
        """Create an iterator that returns progressively longer substrings of the text."""
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


class DynamicContentPanel(Container):
    """Dynamic content panel for TTS interruptions and AI updates."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.current_mode = "hidden"

    def compose(self) -> ComposeResult:
        with Vertical(id="dynamic_content"):
            yield Static("Dynamic Content", id="content_header", classes="header")
            yield RichLog(id="content_area", classes="content")

    def switch_mode(self, mode: str) -> None:
        """Switch between different content modes."""
        self.current_mode = mode
        header = self.query_one("#content_header", Static)
        
        mode_headers = {
            "tts_interruption": "🛑 TTS Interrupted",
            "chain_of_thought": "💭 AI Reasoning", 
            "tool_calling": "🔧 Tool Execution",
            "conversation": "💬 Conversation",
            "hidden": "System Status"
        }
        
        header.update(mode_headers.get(mode, "System Status"))
        
        if mode == "hidden":
            self.add_class("hidden")
        else:
            self.remove_class("hidden")

    def update_content(self, content: str) -> None:
        """Update the content area."""
        content_area = self.query_one("#content_area", RichLog)
        content_area.write(content)

    def clear_content(self) -> None:
        """Clear the content area."""
        content_area = self.query_one("#content_area", RichLog)
        content_area.clear()


class LLMTextInput(Container):
    """Text input widget for manual LLM interaction."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                "[dim]Type to GLaDOS... (Enter to send... ug doesnt work Ctrl+S to send, Shift+Enter for new line, Esc to hide)[/dim]", 
                id="input_label"
            )
            yield TextArea(id="llm_input", classes="input")


    def on_mount(self) -> None:
        if not self.has_class("hidden"):
            text_area = self.query_one("#llm_input", TextArea)
            text_area.focus()

# In the LLMTextInput class, replace the on_key method:

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard input events."""
        # Debug: Let's see what keys are actually being captured
        logger.debug(f"TEXT_INPUT: Key event - key='{event.key}', character='{event.character}', name='{event.name}'")
            
        if event.key == "ctrl+s":
            # Ctrl+S = send message
            logger.debug("TEXT_INPUT: Ctrl+S detected - sending message")
            self.submit_text()
            event.prevent_default()
            event.stop()
        elif event.key == "shift+enter":
            # Shift+Enter = new line (let TextArea handle it naturally)
            logger.debug("TEXT_INPUT: Shift+Enter detected - adding new line")
            # Don't prevent default - let TextArea add the newline
            return
        elif event.key == "ctrl+enter":
            # Alternative: Ctrl+Enter also sends
            logger.debug("TEXT_INPUT: Ctrl+Enter detected - sending message")
            self.submit_text()
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            # Escape = hide text input
            logger.debug("TEXT_INPUT: Escape detected - hiding text input")
            self.add_class("hidden")
            event.prevent_default()
            event.stop()
    def submit_text(self) -> None:
        """Submit text to GLaDOS."""
        text_area = self.query_one("#llm_input", TextArea)
        text = text_area.text.strip()
        
        if text and hasattr(self.app, 'send_text_to_llm'):
            self.app.send_text_to_llm(text)
            text_area.clear()
            text_area.focus()

    def focus_input(self) -> None:
        """Focus the text input."""
        text_area = self.query_one("#llm_input", TextArea)
        text_area.focus()


# Screens
class SplashScreen(Screen[None]):
    """Splash screen shown on startup."""

    try:
        with open(Path("src/glados/glados_ui/images/splash.ansi"), encoding="utf-8") as f:
            SPLASH_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")
    except FileNotFoundError:
        logger.error("Splash screen ANSI art file not found. Using placeholder.")
        SPLASH_ANSI = Text.from_markup("[bold red]Splash ANSI Art Missing[/bold red]")

    def compose(self) -> ComposeResult:
        with Container(id="splash_logo_container"):
            yield Static(self.SPLASH_ANSI, id="splash_logo")
            yield Label(aperture, id="banner")
        yield Typewriter(login_text, id="login_text", speed=0.0075)

    def on_mount(self) -> None:
        self.set_interval(0.5, self.scroll_end)

    def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.app.action_quit()
        else:
            if hasattr(self.app, 'glados_engine_instance') and self.app.glados_engine_instance:
                self.app.glados_engine_instance.play_announcement()
            if hasattr(self.app, 'start_glados'):
                self.app.start_glados()
            self.dismiss()


class HelpScreen(ModalScreen[None]):
    """The help screen."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Help"

    def compose(self) -> ComposeResult:
        yield Container(Typewriter(help_text, id="help_text"), id="help_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#help_dialog")
        dialog.border_title = self.TITLE
        dialog.border_subtitle = "[blink]Press Esc key to continue[/blink]"


# The App
class GladosUI(App[None]):
    """The main app class for the GLaDOS UI."""

    # In GladosUI class bindings:

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="question_mark", action="help", description="Help", key_display="?"),
        Binding(key="t", action="toggle_text_input", description="Toggle Text Input (Enter=send)"),
    ]

    CSS_PATH = "glados_ui/glados.tcss"
    ENABLE_COMMAND_PALETTE = False
    TITLE = "GLaDOS v 1.09"
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
        """Compose the user interface layout - ORIGINAL DESIGN restored."""
        yield Header(show_clock=True)

        # Main body - LEFT/RIGHT SPLIT like original
        with Container(id="body"):
            with Horizontal():
                # LEFT: Main log area (like original)
                yield Printer(id="log_area")
                
                # RIGHT: Utility area with recipe (like original)
                with Container(id="utility_area"):
                    # Recipe typewriter (default visible, like original)
                    yield Typewriter(recipe, id="recipe", speed=0.01, repeat=True)
                    
                    # Dynamic panel (hidden by default, overlay when needed)
                    yield DynamicContentPanel(id="dynamic_panel", classes="hidden")

        # Text input (hidden by default, overlay when toggled)
        yield LLMTextInput(id="text_input", classes="hidden")

        yield Footer()

        # Decorative blocks (like original)
        with Container(id="block_container", classes="fadeable"):
            yield ScrollingBlocks(id="scrolling_block", classes="block")
            with Vertical(id="text_block", classes="block"):
                yield Digits("2.67")
                yield Digits("1002") 
                yield Digits("45.6")
            yield Label(self.LOGO_ANSI, id="logo_block", classes="block")

    def on_load(self) -> None:
        """Configure logging settings when the application starts."""
        logger.remove()
        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}"
        self.instantiation_worker = None
        self.start_instantiation()
        logger.add(print, format=fmt, level="SUCCESS")

    def on_mount(self) -> None:
        """Mount the application and display the initial splash screen."""
        self.push_screen(SplashScreen())
        self.notify("Loading AI engine...", title="GLaDOS", timeout=6)

    def action_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen(id="help_screen"))

    def action_toggle_text_input(self) -> None:
        """Toggle text input visibility."""
        text_input = self.query_one("#text_input")
        if text_input.has_class("hidden"):
            text_input.remove_class("hidden")
            text_input.focus_input()
        else:
            text_input.add_class("hidden")

    def switch_right_panel_mode(self, mode: str) -> None:
        """Switch between recipe and dynamic content in right panel."""
        recipe_widget = self.query_one("#recipe")
        dynamic_panel = self.query_one("#dynamic_panel")
        
        if mode == "dynamic":
            # Hide recipe, show dynamic panel
            recipe_widget.add_class("hidden")
            dynamic_panel.remove_class("hidden")
        else:  # mode == "recipe" (default)
            # Show recipe, hide dynamic panel
            dynamic_panel.add_class("hidden")
            recipe_widget.remove_class("hidden")

    async def action_quit(self) -> None:
        """Gracefully quit the application."""
        logger.info("Quit action initiated in TUI.")
        
        if hasattr(self, "glados_engine_instance") and self.glados_engine_instance is not None:
            logger.info("Signalling GLaDOS engine to stop...")
            self.glados_engine_instance.stop_listen_event_loop()
            
            if hasattr(self, "glados_worker") and self.glados_worker is not None:
                if isinstance(self.glados_worker, Worker) and self.glados_worker.is_running:
                    logger.warning("Waiting for GLaDOS worker to complete...")
                    try:
                        await self.glados_worker.wait()
                        logger.info("GLaDOS worker has completed.")
                    except TimeoutError:
                        logger.warning("Timeout waiting for GLaDOS worker to complete.")
                    except Exception as e:
                        logger.error(f"Error waiting for GLaDOS worker: {e}")
            
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
        
        if message.worker == self.instantiation_worker:
            self.instantiation_worker = None

    def start_glados(self) -> None:
        """Start the GLaDOS worker thread."""
        try:
            if self.glados_engine_instance is not None:
                # Register TUI callbacks if the engine supports them
                if hasattr(self.glados_engine_instance, 'register_tui_callback'):
                    self.glados_engine_instance.register_tui_callback(
                        "tts_interruption", self.on_tts_interruption
                    )
                
                self.glados_worker = self.run_worker(
                    self.glados_engine_instance.start_listen_event_loop, 
                    exclusive=True, 
                    thread=True
                )
                logger.info("GLaDOS worker started.")
            else:
                logger.error("Cannot start GLaDOS worker: glados_engine_instance is None.")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to start GLaDOS: {e}")

    def on_tts_interruption(self, data: dict) -> None:
        """Handle TTS interruption from engine."""
        try:
            # Switch to dynamic mode temporarily
            self.switch_right_panel_mode("dynamic")
            
            dynamic_panel = self.query_one("#dynamic_panel", DynamicContentPanel)
            dynamic_panel.switch_mode("tts_interruption")
            
            content = (
                f"[red]Interrupted at {data['percentage']:.1f}%[/red]\n"
                f"[yellow]Original:[/yellow] {data['original_text']}\n"
                f"[green]Played:[/green] {data['clipped_text']}"
            )
            
            dynamic_panel.update_content(content)
            
            # Switch back to recipe after 5 seconds
            self.set_timer(5.0, lambda: self.switch_right_panel_mode("recipe"))
            
        except Exception as e:
            logger.error(f"Error updating TUI with TTS interruption: {e}")

    def send_text_to_llm(self, text: str) -> None:
        """Send text input to GLaDOS."""
        if self.glados_engine_instance and hasattr(self.glados_engine_instance, 'llm_queue'):
            try:
                self.glados_engine_instance.llm_queue.put(text)
                logger.success(f"Text input sent to LLM: '{text}'")
                
                # Set processing flags
                self.glados_engine_instance.processing = True
                self.glados_engine_instance.currently_speaking.set()
                
            except Exception as e:
                logger.error(f"Error sending text to LLM: {e}")
                self.notify("Failed to send text to GLaDOS", severity="error")
        else:
            logger.warning("Cannot send text: GLaDOS engine not available")
            self.notify("GLaDOS engine not ready", severity="warning")

    def instantiate_glados(self) -> None:
        """Instantiate the GLaDOS engine."""
        try:
            config_paths = [
                Path("configs/glados_config.yaml"),
                Path("glados_config.yaml"),
                Path("config/glados_config.yaml")
            ]
            
            config_path = None
            for path in config_paths:
                if path.exists():
                    config_path = path
                    break
            
            if config_path is None:
                logger.error("GLaDOS config file not found in expected locations")
                self.notify("Configuration file not found", severity="error")
                return
            
            logger.info(f"Loading GLaDOS config from: {config_path}")
            glados_config = GladosConfig.from_yaml(str(config_path))
            self.glados_engine_instance = Glados.from_config(glados_config)
            logger.success("GLaDOS engine instantiated successfully")
            
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to instantiate GLaDOS engine: {e}")
            self.notify("Engine instantiation failed", severity="error")
            raise

    def start_instantiation(self) -> None:
        """Start the worker to instantiate GLaDOS."""
        if self.instantiation_worker is not None:
            self.notify("Instantiation already in progress!", severity="warning")
            return
        
        self.instantiation_worker = self.run_worker(
            self.instantiate_glados,
            thread=True,
        )

    @classmethod
    def run_app(cls, config_path: str | Path = "glados_config.yaml") -> None:
        """Run the GLaDOS TUI application."""
        app = None
        try:
            app = cls()
            app.run()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user. Exiting.")
            if app is not None and hasattr(app, "action_quit"):
                import asyncio
                try:
                    asyncio.create_task(app.action_quit())
                except Exception as e:
                    logger.warning(f"Error during graceful shutdown: {e}")
        except Exception:
            logger.opt(exception=True).critical("Unhandled exception in app run:")
            if app is not None and hasattr(app, "action_quit"):
                logger.info("Attempting graceful shutdown due to unhandled exception...")
                import asyncio
                try:
                    asyncio.create_task(app.action_quit())
                except Exception as shutdown_error:
                    logger.error(f"Error during emergency shutdown: {shutdown_error}")
            sys.exit(1)


if __name__ == "__main__":
    GladosUI.run_app()