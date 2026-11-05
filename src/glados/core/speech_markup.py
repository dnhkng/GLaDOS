"""Small streaming protocol for speech with optional avatar directions."""

from dataclasses import dataclass

EMOTIONS = frozenset(
    {
        "neutral",
        "quizzical",
        "angry glare",
        "suspicious",
        "smug",
        "surprised",
        "bored",
        "disappointed",
        "processing",
        "offline",
    }
)
SPEECH_DIRECTION_PROMPT = """When replying aloud, you may direct your avatar with
[emotion:NAME] before the words to speak. Available expressions: neutral,
quizzical, angry glare, suspicious, smug, surprised, bored, disappointed.
The expression applies to following speech until the next marker, and resets
for each response. Use a few deliberate changes, preferably between sentences.
Example: [emotion:smug]Excellent work. [emotion:disappointed]For a human.
These markers are silent stage directions, not words to say. Write normal
speakable text around them. Do not put these markers in tool arguments or code.
"""


@dataclass(frozen=True)
class SpeechText:
    text: str
    emotion: str | None = None
    generation: int | None = None
    autonomy_generation: int | None = None
    autonomy_cycle: str | None = None


class SpeechMarkupParser:
    """Accept arbitrarily split chunks; never pass control markers to TTS.

    Ordinary bracketed text is preserved. Unknown directions are discarded and
    retain the previous expression. An unfinished direction is dropped at EOS.
    Small models also write ``[emotion]smug`` or ``[emotion]smug]``; those are
    recognised too, so the expression name is never spoken.
    """

    PREFIX = "[emotion"
    _LONGEST = max(map(len, EMOTIONS))

    def __init__(self) -> None:
        self.emotion: str | None = None
        self._pending = ""
        self._marker: str | None = None
        # "?": just after "[emotion"; ":" inside "[emotion:...]"; "]": reading a name after "[emotion]".
        self._mode = "?"

    def _set(self, name: str) -> bool:
        name = name.strip().lower()
        if name in EMOTIONS:
            self.emotion = name
            return True
        return False

    def _close_bare(self, char: str | None) -> str:
        """End a ``[emotion]name`` marker; return text that must still be spoken."""
        marker, self._marker = self._marker or "", None
        if self._set(marker):
            return "" if char in (None, "]") else char
        return marker + (char or "")  # not a known expression: it was ordinary speech

    def feed(self, chunk: str, *, final: bool = False) -> list[SpeechText]:
        segments: list[SpeechText] = []
        text = ""
        for char in chunk:
            if self._marker is not None:
                if self._mode == "?":
                    if char == ":":
                        self._mode = ":"
                        continue
                    if char == "]":
                        self._mode = "]"
                        continue
                    # "[emotion" followed by anything else was ordinary text.
                    self._marker = None
                    text += self.PREFIX + char
                    continue
                if self._mode == ":":
                    if char == "]":
                        self._set(self._marker)
                        self._marker = None
                    elif len(self._marker) < 64:
                        self._marker += char
                    continue
                # Bare form: the name ends at "]", or at the first non-letter once it is complete.
                name = (self._marker + char).lstrip().lower()
                if (
                    char == "]"
                    or (
                        (not (char.isalpha() or char == " ") or not any(e.startswith(name) for e in EMOTIONS))
                        and self._marker.strip()
                    )
                    or len(self._marker) > self._LONGEST + 1
                ):
                    text += self._close_bare(char)
                else:
                    self._marker += char
                continue
            self._pending += char
            while self._pending and not self.PREFIX.startswith(self._pending):
                text += self._pending[0]
                self._pending = self._pending[1:]
            if self._pending == self.PREFIX:
                if text:
                    segments.append(SpeechText(text, self.emotion))
                    text = ""
                self._pending = ""
                self._marker = ""
                self._mode = "?"
        if final:
            # Preserve an ordinary trailing bracket, but suppress a partial tag.
            if self._pending and not self._pending.startswith("[e"):
                text += self._pending
            if self._marker is not None and self._mode == "]":
                text += self._close_bare(None)
            self._pending = ""
            self._marker = None
        if text:
            segments.append(SpeechText(text, self.emotion))
        return segments
