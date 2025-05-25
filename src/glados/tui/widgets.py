"""TUI Widgets for GLaDOS - Custom widgets for the voice assistant interface.

This module contains all custom Textual widgets used throughout the GLaDOS TUI,
including display widgets, input widgets, and animated components.
"""

from collections.abc import Iterator
import random
from typing import ClassVar

from rich.text import Text
from rich.panel import Panel
from textual import events
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Static, TextArea, Log, RichLog, Input  # Add Input import
from textual.app import ComposeResult

class Printer(RichLog):
    """A subclass of Textual's RichLog which captures and displays all print calls.
    
    This widget automatically captures print statements from the application and
    displays them in a rich log format with markup support. It's the primary
    logging display widget for the GLaDOS TUI.
    
    Features:
    - Automatic print capture and display
    - Rich markup support for colored text
    - Text wrapping for long lines
    - Special formatting for DEBUG messages
    """
    
    def on_mount(self) -> None:
        """Initialize the print capture widget.
        
        Sets up text wrapping, markup support, and begins capturing print
        statements from the application for display in the log.
        """
        self.wrap = True
        self.markup = True
        self.begin_capture_print()

    def on_print(self, event: events.Print) -> None:
        """Handle captured print events.
        
        Processes print statements captured from the application, filters out
        empty lines, and applies special formatting for certain log levels.
        
        Args:
            event: The print event containing the text to display
        """
        if (text := event.text) != "\n":
            # Apply special formatting for DEBUG messages
            formatted_text = text.rstrip().replace("DEBUG", "[red]DEBUG[/]")
            self.write(formatted_text)


class ScrollingBlocks(Log):
    """A widget for displaying animated random scrolling blocks.
    
    Creates a continuous scrolling display of random block characters,
    providing visual decoration and system activity indication.
    
    Features:
    - Random block character generation
    - Continuous scrolling animation
    - Width-adaptive block generation
    - Hidden scrollbars for clean appearance
    """
    
    BLOCKS = "⚊⚌☰𝌆䷀"
    DEFAULT_CSS = """
    ScrollingBlocks {
        scrollbar_size: 0 0;
        overflow-x: hidden;
    }"""

    def _animate_blocks(self) -> None:
        """Generate and display a line of random block characters.
        
        Creates a string of random block characters with length adjusted
        to fit the current widget width, accounting for border and padding.
        Each block is randomly selected from the predefined BLOCKS set.
        """
        # Ensure width calculation doesn't go negative for small widgets
        num_blocks_to_generate = max(0, self.size.width - 8)
        random_blocks = " ".join(
            random.choice(self.BLOCKS) for _ in range(num_blocks_to_generate)
        )
        self.write_line(random_blocks)

    def on_show(self) -> None:
        """Set up animation when widget becomes visible.
        
        Initiates a recurring animation timer that calls _animate_blocks
        at regular intervals to create the scrolling effect.
        """
        self.set_interval(0.18, self._animate_blocks)


class Typewriter(Static):
    """A widget that displays text one character at a time with typewriter effect.
    
    Provides animated text display simulating typewriter output, with support
    for Rich markup, custom speed, and repeat functionality.
    
    Features:
    - Character-by-character text reveal
    - Blinking cursor effect (when markup enabled)
    - Configurable typing speed
    - Optional repeat functionality
    - Automatic markup detection and handling
    """

    def __init__(
        self,
        text: str = "_",
        id: str | None = None,
        speed: float = 0.01,  # Time between each character
        repeat: bool = False,  # Whether to restart at the end
        *args,
        **kwargs,
    ) -> None:
        """Initialize the typewriter widget.
        
        Args:
            text: The text to display with typewriter effect
            id: Widget identifier for CSS and queries
            speed: Time delay between character reveals (seconds)
            repeat: Whether to restart animation when complete
            *args, **kwargs: Additional arguments passed to parent Static widget
        """
        super().__init__(*args, **kwargs)
        self._text = text
        self.__id_for_child = id
        self._speed = speed
        self._repeat = repeat
        
        # Determine if Rich markup should be used
        # Disable markup if text contains brackets to avoid conflicts
        self._use_markup = "[" not in text and "]" not in text

    def compose(self) -> ComposeResult:
        """Compose the typewriter widget layout.
        
        Creates a static widget for text display within a vertical scroll
        container to handle text overflow.
        """
        self._static = Static(markup=self._use_markup)
        self._vertical_scroll = VerticalScroll(self._static, id=self.__id_for_child)
        yield self._vertical_scroll

    def _get_iterator(self) -> Iterator[str]:
        """Create an iterator for progressive text reveal.
        
        Returns an iterator that yields progressively longer substrings
        of the text with a cursor at the end.
        
        Returns:
            Iterator yielding text substrings with cursor
        """
        if self._use_markup:
            # Use Rich markup for blinking cursor
            return (
                self._text[:i] + "[blink]_[/blink]" 
                for i in range(len(self._text) + 1)
            )
        else:
            # Use simple underscore cursor
            return (
                self._text[:i] + "_" 
                for i in range(len(self._text) + 1)
            )

    def on_mount(self) -> None:
        """Initialize typewriter animation when widget mounts."""
        self._iter_text = self._get_iterator()
        self.set_interval(self._speed, self._display_next_char)

    def _display_next_char(self) -> None:
        """Display the next character in the typewriter sequence.
        
        Updates the display with the next character from the iterator,
        handles scrolling, and manages repeat functionality.
        """
        try:
            # Scroll down for natural typewriter feel
            if not self._vertical_scroll.is_vertical_scroll_end:
                self._vertical_scroll.scroll_down()
            
            # Display next character
            self._static.update(next(self._iter_text))
        except StopIteration:
            if self._repeat:
                # Restart animation if repeat is enabled
                self._iter_text = self._get_iterator()


class DynamicContentPanel(Container):
    """Dynamic content panel that switches between different content types.
    
    A flexible panel that can display different types of real-time information
    from the GLaDOS engine, including TTS interruptions, chain of thought,
    tool calling status, and system information.
    
    Features:
    - Multiple content modes (status, TTS interruption, chain of thought, etc.)
    - Dynamic header updates based on content type
    - Rich text display with markup support
    - Easy mode switching for different content types
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the dynamic content panel.
        
        Args:
            **kwargs: Additional arguments passed to parent Container
        """
        super().__init__(**kwargs)
        self.current_mode = "status"

    def compose(self) -> ComposeResult:
        """Compose the dynamic content panel layout.
        
        Creates a vertical layout with a header for the content type
        and a scrolling log area for the actual content.
        """
        with Vertical(id="dynamic_content"):
            yield Static("System Status", id="content_header", classes="header")
            yield RichLog(id="content_area", classes="content")

    def switch_mode(self, mode: str) -> None:
        """Switch the panel to a different content mode.
        
        Updates the header text and prepares the panel for displaying
        the specified type of content.
        
        Args:
            mode: The content mode to switch to
                 ('status', 'tts_interruption', 'chain_of_thought', 'tool_calling', 'conversation')
        """
        self.current_mode = mode
        header = self.query_one("#content_header", Static)
        
        # Update header based on mode
        mode_headers = {
            "tts_interruption": "TTS Interruption",
            "chain_of_thought": "Chain of Thought", 
            "tool_calling": "Tool Execution",
            "conversation": "Conversation Context",
            "status": "System Status"
        }
        
        header.update(mode_headers.get(mode, "System Status"))

    def update_content(self, content: str) -> None:
        """Update the content area with new information.
        
        Args:
            content: Rich markup text to display in the content area
        """
        content_area = self.query_one("#content_area", RichLog)
        content_area.write(content)

    def clear_content(self) -> None:
        """Clear all content from the display area."""
        content_area = self.query_one("#content_area", RichLog)
        content_area.clear()


class TTSInterruptionWidget(Static):
    """Specialized widget for displaying TTS interruption information.
    
    Shows detailed information when text-to-speech is interrupted,
    including the percentage played, interrupted text, and reason.
    """

    def update_interruption(
        self, 
        interrupted_text: str, 
        percentage: float, 
        reason: str
    ) -> None:
        """Update the widget with interruption details.
        
        Args:
            interrupted_text: The text that was being spoken when interrupted
            percentage: Percentage of speech completed before interruption
            reason: Reason for the interruption (e.g., 'voice_activation')
        """
        content = Panel(
            f"[red]Interrupted at {percentage:.1f}%[/red]\n"
            f"Text: {interrupted_text}\n"
            f"Reason: {reason}",
            title="TTS Interruption",
            border_style="red"
        )
        self.update(content)


class ChainOfThoughtWidget(RichLog):
    """Widget for displaying chain of thought reasoning steps.
    
    Shows the step-by-step reasoning process of the AI, useful for
    debugging and understanding AI decision-making.
    """

    def add_thought_step(self, step: str, reasoning: str) -> None:
        """Add a new reasoning step to the display.
        
        Args:
            step: Name or identifier for the reasoning step
            reasoning: Detailed explanation of the reasoning
        """
        self.write(f"[bold cyan]{step}[/bold cyan]: {reasoning}")


class LLMTextInput(Container):
    """Text input widget for manual LLM interaction.
    
    Provides a text input area for users to type messages directly to
    the GLaDOS AI without using voice input. Uses a multi-line TextArea
    for longer messages.
    
    Features:
    - Multi-line text input with TextArea
    - Keyboard shortcuts for submission (Ctrl+Enter)
    - Integration with GLaDOS LLM queue
    - Visual feedback for input state
    - Proper placeholder-like functionality
    """

    def compose(self) -> ComposeResult:
        """Compose the text input layout.
        
        Creates a vertical container with a label and text area for user input.
        Since TextArea doesn't support placeholder directly, we use a label.
        """
        with Vertical():
            yield Static(
                "[dim]Type your message to GLaDOS... (Ctrl+Enter to send)[/dim]",
                id="input_label",
                classes="input_label"
            )
            yield TextArea(
                id="llm_input",
                classes="input"
            )

    def on_mount(self) -> None:
        """Set up the text input when mounted."""
        # Focus the text area when the widget is mounted and visible
        if not self.has_class("hidden"):
            text_area = self.query_one("#llm_input", TextArea)
            text_area.focus()

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard input events.
        
        Processes keyboard shortcuts, particularly Ctrl+Enter for
        submitting text to the LLM.
        
        Args:
            event: The keyboard event
        """
        if event.key == "ctrl+enter":
            self.submit_text()
            event.prevent_default()
            event.stop()

    def submit_text(self) -> None:
        """Submit the current text to the GLaDOS LLM queue.
        
        Retrieves text from the input area, sends it to the GLaDOS
        engine for processing, and clears the input field.
        """
        text_area = self.query_one("#llm_input", TextArea)
        text = text_area.text.strip()
        
        if text and hasattr(self.app, 'send_text_to_llm'):
            # Send text to GLaDOS
            self.app.send_text_to_llm(text)
            
            # Clear the input area
            text_area.clear()
            
            # Keep focus on the text area for continued input
            text_area.focus()

    def clear_input(self) -> None:
        """Clear the text input area."""
        text_area = self.query_one("#llm_input", TextArea)
        text_area.clear()

    def focus_input(self) -> None:
        """Focus the text input area."""
        text_area = self.query_one("#llm_input", TextArea)
        text_area.focus()