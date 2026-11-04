"""Per-turn voice latency, derived from the events the pipeline already publishes.

A user turn starts at ``user_input`` and ends when its first audio starts playing; the
breakdown (speech recognition, model first token, first synthesized clip, playback) is
published as a ``latency/turn`` event and kept for the console.
"""

from __future__ import annotations

from collections import deque
import threading
from typing import Any

from .bus import ObservabilityBus
from .events import ObservabilityEvent

USER_SOURCES = {"asr", "text", "audio"}


class LatencyTracker:
    def __init__(self, bus: ObservabilityBus, keep: int = 50) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._turn: dict[str, Any] | None = None
        self._recent: deque[dict[str, Any]] = deque(maxlen=keep)
        bus.add_listener(self.observe)

    def observe(self, event: ObservabilityEvent) -> None:
        if event.meta.get("autonomy"):
            return  # Background speech is not a reply anyone is waiting for.
        finished = None
        with self._lock:
            if event.kind == "user_input" and event.source in USER_SOURCES:
                asr_ms = event.meta.get("asr_ms")
                start = event.timestamp - (asr_ms or 0) / 1000
                self._turn = {"source": event.source, "start": start, "input": event.timestamp, "asr_ms": asr_ms}
            elif self._turn is None:
                return
            elif (event.source == "llm" and event.kind == "first_token"
                  and event.meta.get("lane", "priority") == "priority"):
                self._turn.setdefault("first_token", event.timestamp)
            elif event.source == "tts" and event.kind == "ready":
                self._turn.setdefault("ready", event.timestamp)
            elif event.source == "tts" and event.kind == "play":
                finished, self._turn = self._finish(self._turn, event.timestamp), None
            if finished:
                self._recent.appendleft(finished)
        if finished:
            self._bus.emit("latency", "turn", f"Reply started {finished['total_ms']:.0f} ms after the user finished",
                           level="debug", meta=finished)

    @staticmethod
    def _finish(turn: dict[str, Any], played: float) -> dict[str, Any]:
        def span(start: float | None, end: float | None) -> float | None:
            return round((end - start) * 1000, 1) if start is not None and end is not None else None

        first_token, ready = turn.get("first_token"), turn.get("ready")
        return {
            "source": turn["source"],
            "at": played,
            "asr_ms": turn["asr_ms"],
            "llm_ms": span(turn["input"], first_token),
            "tts_ms": span(first_token, ready),
            "playback_ms": span(ready, played),
            "total_ms": span(turn["start"], played),
        }

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._recent)
