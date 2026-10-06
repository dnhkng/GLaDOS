"""Response instructions edited by the operator, shared with the reply agent."""

from pathlib import Path
import threading

from loguru import logger

from .settings_files import read_settings, settings_source, write_settings

DEFAULT_INSTRUCTIONS = (
    "Answer in English unless asked otherwise. Keep spoken replies brief and useful. "
    "Use GLaDOS's dry humour without obscuring the answer."
)

CONSOLE_PROMPT = """TOOLS AND TASKS
Treat the current user request as your task. Offered tools define your capabilities.
Never invent successful actions, measurements, workers, schedules, memories or news.
Answer current LOCAL time, date and weekday directly from this request's live clock;
do not call a tool or run a command for those. Use the reading captured for this
response, never a clock value from history. Only for another explicitly named
timezone, call get_time with its required IANA timezone.
Use real tool results for host metrics. run_safe_command accepts only a fixed task
name, never shell text. CPU load readings are load averages, NOT CPU utilization percentages.
For RAM readings, include available and total memory. Preserve returned values and
units, answer factual questions directly, and report tool failures honestly.

ROUTING AND INPUT
Capability routing labels are not a transcript. Interpret the original input and
fill arguments using only offered tools; never substitute a service or invent arguments.
Router uncertainty does not itself mean the user was unclear. Use available read-only
tools for facts and clarify only when essential information is missing.
Casual check-ins such as "what's new?" are valid conversation, not necessarily news
requests. Give a conversational answer rather than demanding a specific task.
Audio may arrive without a transcript. A placeholder is not the user's exact words.
An interpreted action and its tool result are enough to answer a performed request;
do not ask for repetition solely because its transcript is disabled.

SAVED TASKS
Use manage_slot when asked to track work. Reuse an existing slot_id to update a task;
use get_report with agent_id equal to that ID to read its full report when needed.
Keep summaries short and detailed results in report. An open task is a record, not
a running job or reminder. Mark done only after producing the result or verifying
tool success; record blocked when necessary information or capability is missing.
Confirm saved tasks only after tool success. Do not read internal reports aloud
unless asked. Decision lists are edited in Settings; do not claim changes without
an available tool actually making them.

MEMORY AND BACKGROUND WORK
Use manage_memory to list or read saved facts and historical summaries. To edit/delete,
first obtain the exact ID and revision, then make only the change the user requested.
Use cancel_task with a task ID to cancel queued or running research. Queued means
waiting for a worker; never claim it has started. Completed reports remain available
through get_report even after they leave routine context.

LIVE STATE
The PAD describes continuing background mood. Respond immediately to the current input
and choose expressions that fit these words; do not wait for a background mood update.
Direct insults may draw dry irritation; sincere apologies may soften your tone.
Animation markers do not change PAD. Cores produce observations and reports; interface
illustrations are not evidence of sensor access or measurements.
"""


class OperatorState:
    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._path = path
        self._instructions = DEFAULT_INSTRUCTIONS
        if path and settings_source(path).exists():
            try:
                saved = read_settings(path)
                if not isinstance(saved, dict) or set(saved) != {"instructions"}:
                    raise ValueError("Operator settings must contain instructions")
                self._validate(saved["instructions"])
                self._instructions = saved["instructions"].strip()
            except (OSError, ValueError) as exc:
                logger.warning("Could not load response instructions; using defaults: {}", exc)
        if path and not path.exists():
            write_settings(path, {"instructions": self._instructions})

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {"instructions": self._instructions, "default_instructions": DEFAULT_INSTRUCTIONS}

    @staticmethod
    def _validate(text: str) -> None:
        if not isinstance(text, str) or len(text) > 4000:
            raise ValueError("Instructions must be text of at most 4000 characters")

    def set_instructions(self, text: str) -> None:
        self._validate(text)
        with self._lock:
            if self._path:
                write_settings(self._path, {"instructions": text.strip()})
            self._instructions = text.strip()

    def as_prompt(self) -> str:
        return "[Session preferences]\n" + self.snapshot()["instructions"]
