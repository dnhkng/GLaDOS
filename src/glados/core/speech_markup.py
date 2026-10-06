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
    """

    PREFIX = "[emotion:"

    def __init__(self) -> None:
        self.emotion: str | None = None
        self._pending = ""
        self._marker: str | None = None

    def feed(self, chunk: str, *, final: bool = False) -> list[SpeechText]:
        segments: list[SpeechText] = []
        text = ""
        for char in chunk:
            if self._marker is not None:
                if char == "]":
                    name = self._marker.strip().lower()
                    if name in EMOTIONS:
                        self.emotion = name
                    self._marker = None
                elif len(self._marker) < 64:
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
        if final:
            # Preserve an ordinary trailing bracket, but suppress a partial tag.
            if self._pending and not self._pending.startswith("[e"):
                text += self._pending
            self._pending = ""
            self._marker = None
        if text:
            segments.append(SpeechText(text, self.emotion))
        return segments
