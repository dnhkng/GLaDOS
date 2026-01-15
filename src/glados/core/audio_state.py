from __future__ import annotations

from dataclasses import dataclass
import threading
import time


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
