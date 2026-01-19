from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObservabilityEvent:
    timestamp: float
    source: str
    kind: str
    message: str
    level: str = "info"
    meta: dict[str, Any] = field(default_factory=dict)


def trim_message(text: str, limit: int = 500) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)].rstrip()}..."
