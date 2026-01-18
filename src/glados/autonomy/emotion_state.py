"""
Emotional state data structures for GLaDOS.

Uses PAD (Pleasure-Arousal-Dominance) model with a slower mood baseline.
State transitions are decided by LLM, not hard-coded math.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EmotionState:
    """
    Current emotional state using PAD dimensions.

    State responds quickly to events. Mood drifts slowly toward state.
    All values range from -1 to +1.
    """

    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    mood_pleasure: float = 0.0
    mood_arousal: float = 0.0
    mood_dominance: float = 0.0

    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionState:
        return cls(
            pleasure=float(data.get("pleasure", 0.0)),
            arousal=float(data.get("arousal", 0.0)),
            dominance=float(data.get("dominance", 0.0)),
            mood_pleasure=float(data.get("mood_pleasure", 0.0)),
            mood_arousal=float(data.get("mood_arousal", 0.0)),
            mood_dominance=float(data.get("mood_dominance", 0.0)),
            last_update=float(data.get("last_update", time.time())),
        )

    def to_prompt(self) -> str:
        """Human-readable summary for injection into main agent context."""
        p, a, d = self.pleasure, self.arousal, self.dominance

        # Describe the emotional quadrant
        if p > 0.3 and a > 0.3:
            feeling = "excited and engaged"
        elif p > 0.3 and a < -0.3:
            feeling = "calm and content"
        elif p < -0.3 and a > 0.3:
            feeling = "agitated and frustrated"
        elif p < -0.3 and a < -0.3:
            feeling = "bored and listless"
        elif abs(p) <= 0.3 and abs(a) <= 0.3:
            feeling = "neutral"
        elif p > 0.3:
            feeling = "pleased"
        elif p < -0.3:
            feeling = "displeased"
        elif a > 0.3:
            feeling = "alert"
        else:
            feeling = "relaxed"

        # Add dominance flavor
        if d > 0.3:
            feeling += ", feeling in control"
        elif d < -0.3:
            feeling += ", feeling uncertain"

        return f"[emotion] Currently {feeling} (P:{p:+.1f} A:{a:+.1f} D:{d:+.1f})"


@dataclass(frozen=True)
class EmotionEvent:
    """
    An event that may affect emotional state.

    Uses natural language description - LLM interprets the semantics.
    """

    source: str  # "user", "vision", "system"
    description: str
    timestamp: float = field(default_factory=time.time)

    def to_prompt_line(self) -> str:
        """Format for inclusion in LLM prompt."""
        age = time.time() - self.timestamp
        if age < 60:
            age_str = f"{age:.0f}s ago"
        else:
            age_str = f"{age / 60:.1f}m ago"
        return f"- [{self.source}] {self.description} ({age_str})"
