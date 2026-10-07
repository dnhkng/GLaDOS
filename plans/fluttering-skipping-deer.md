# Fix Non-Toggle Engine Commands Not Displaying Results

## Problem
Commands like MCP, Minds, Preferences, Context, Config, Knowledge, and Memory close the palette without showing any output. The `_run_engine_command` method executes the command but ignores the return value.

## Root Cause
```python
def _run_engine_command(self, name: str) -> None:
    if self.glados_engine_instance:
        self.glados_engine_instance.handle_command(f"/{name}")  # Returns string, but ignored!
```

## Solution
Create an `InfoScreen` modal to display command output, similar to how `ContextScreen` works.

### File: `src/glados/tui.py`

**1. Add new InfoScreen class** (after MessagesScreen):
```python
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
```

**2. Add CSS for info_dialog** (reuse context_dialog styles):
```css
#info_dialog {
    border-title-align: center;
    border: round $primary;
    color: $primary;
    background: $background;
    padding: 1 2;
    height: 70%;
    width: 80%;
}
```

**3. Update _run_engine_command to show results:**
```python
def _run_engine_command(self, name: str) -> None:
    """Execute an engine command and display the result."""
    if not self.glados_engine_instance:
        self.notify("Engine not ready.", severity="warning")
        return
    response = self.glados_engine_instance.handle_command(f"/{name}")
    if response:
        # Show in InfoScreen modal for multi-line output
        self.push_screen(InfoScreen(name.title(), response))
```

## Verification
1. Run `uv run glados tui`
2. Press ctrl+p
3. Select "Config" → should show config info in a modal
4. Select "Memory" → should show memory stats in a modal
5. Select "Knowledge" → should show knowledge entries in a modal
6. Press Esc to close each modal
