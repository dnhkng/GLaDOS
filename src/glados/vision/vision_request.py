from __future__ import annotations

from dataclasses import dataclass
import queue


@dataclass(slots=True)
class VisionRequest:
    """Request for an on-demand vision description."""

    prompt: str
    max_tokens: int
    response_queue: queue.Queue[str]
