# TUI command output — implemented

Reviewed **2026-10-11**. This file retains its original name for existing links.
It is a completed fix, not a pending implementation plan.

Non-toggle palette commands formerly ignored `handle_command` return values.
`src/glados/tui.py` now provides `InfoScreen`, and
`GladosUI._run_engine_command` displays nonempty command output in that modal.
It warns when the engine is unavailable. The modal has a scrollable text area
and Escape-to-close binding; command execution remains in the engine.

Manual verification when the application is next running:

1. Run `uv run --no-sync glados tui` and open Ctrl+P.
2. Select Config, Memory or Knowledge and inspect its returned text.
3. Press Escape to close the modal.
4. Confirm toggle commands still use their existing control flow.

This documentation review verified the source; it did not launch GPU-backed TUI
or perform a new manual run. The remaining architecture work is tracked in
[architecture-refactor-progress.md](architecture-refactor-progress.md).
