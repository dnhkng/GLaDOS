"""Playback-owned avatar state, independent of the browser and audio format."""

import threading
import time

from ..observability import ObservabilityBus
from .speech_markup import EMOTIONS


class SpeechAnimationState:
    def __init__(self, bus: ObservabilityBus) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._revision = 0
        self._state: dict[str, object] = {"revision": 0, "active": False, "emotion": None}

    def set(self, active: bool, emotion: str | None = None) -> None:
        with self._lock:
            if not active and not self._state["active"]:
                return
            self._revision += 1
            self._state = {
                "revision": self._revision,
                "active": active,
                "emotion": emotion if active and emotion in EMOTIONS else None,
                "updated_at": time.time(),
            }
            self._bus.emit(
                "avatar", "performance", "speech" if active else "idle", level="debug", meta=dict(self._state)
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._state)
