from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import threading
import time
import unicodedata

# Without echo cancellation the microphone hears the speakers: a transcript made mostly of words she
# played in the last few seconds is her own voice, not the user's.
ECHO_TAIL_S = 8.0
ECHO_OVERLAP = 0.7


def _words(text: str) -> list[str]:
    folded = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", folded)


@dataclass(frozen=True)
class AudioSnapshot:
    rms: float
    vad_active: bool
    updated_at: float


class AudioState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rms = 0.0
        self._vad_active = False
        self._updated_at = 0.0
        self._spoken: deque[tuple[float, list[str]]] = deque(maxlen=40)

    def spoke(self, text: str, duration_s: float) -> None:
        """Remember a line she is playing until shortly after it ends."""
        with self._lock:
            self._spoken.append((time.monotonic() + duration_s + ECHO_TAIL_S, _words(text)))

    def is_echo(self, heard: str) -> bool:
        """Whether a transcript is mostly her own recent speech picked up by the microphone."""
        words = _words(heard)
        if not words:
            return False
        now = time.monotonic()
        with self._lock:
            recent = {word for until, line in self._spoken if until >= now for word in line}
        return sum(word in recent for word in words) / len(words) >= ECHO_OVERLAP

    def update(self, rms: float, vad_active: bool) -> None:
        now = time.time()
        with self._lock:
            self._rms = float(rms)
            self._vad_active = bool(vad_active)
            self._updated_at = now

    def reset(self) -> None:
        with self._lock:
            self._rms = 0.0
            self._vad_active = False
            self._updated_at = time.time()

    def snapshot(self) -> AudioSnapshot:
        with self._lock:
            return AudioSnapshot(
                rms=self._rms,
                vad_active=self._vad_active,
                updated_at=self._updated_at,
            )
