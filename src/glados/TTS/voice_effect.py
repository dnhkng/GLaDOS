"""Post-processing that pushes a neutral voice towards GLaDOS: a little higher, and metallic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..audio_io.resample import resample


@dataclass(frozen=True)
class GladosEffect:
    # The synthesizer speaks this much slower, then playback is resampled back to the original
    # duration: the pitch rises by this factor while the speaking rate stays the same.
    pitch: float = 1.06
    # A short feedforward comb gives the hollow, synthetic resonance; a tremolo too slow to hear
    # as vibrato adds a faint machine flutter.
    comb_ms: float = 2.6
    comb_gain: float = 0.45
    flutter_hz: float = 28.0
    flutter_depth: float = 0.08

    def apply(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        shape = audio.shape
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not x.size:
            return audio
        if self.pitch != 1.0:
            x = resample(x, sample_rate * self.pitch, sample_rate)
        delay = max(1, round(sample_rate * self.comb_ms / 1000))
        y = x.copy()
        y[delay:] += self.comb_gain * x[:-delay]
        t = np.arange(y.size, dtype=np.float32) / sample_rate
        y *= 1.0 - self.flutter_depth * (0.5 + 0.5 * np.sin(2 * np.pi * self.flutter_hz * t))
        peak = float(np.max(np.abs(y)))
        if peak > 0.95:  # the comb can push loud vowels past full scale
            y *= 0.95 / peak
        return y.astype(np.float32).reshape(-1, *shape[1:]) if len(shape) > 1 else y.astype(np.float32)
