#!/usr/bin/env python3
"""Dependency-free high-quality audio resampling.

The TTS model produces audio at a fixed sample rate (``glados.onnx`` outputs at
22050 Hz), while most hardware outputs at its own native rate. If playback runs
at the TTS rate through the PortAudio sample-rate conversion, the built-in SRC
is low quality and produces audible crackling.

This module provides a windowed-sinc (polyphase) resampler implemented purely
with the existing ``numpy`` dependency, so audio can be converted to the output
device's native rate before playback -- without introducing ``soxr`` (the engine
used by the original upstream PR) as a new dependency.
"""

import math

import numpy as np
from numpy.typing import NDArray


def resample(
    audio: NDArray[np.float32],
    src_rate: float,
    dst_rate: float,
    *,
    zero_crossings: int = 12,
    chunk_size: int = 262144,
) -> NDArray[np.float32]:
    """Resample mono audio from ``src_rate`` Hz to ``dst_rate`` Hz.

    Uses a windowed-sinc low-pass kernel convolved at fractionally interpolated
    output positions. This approximates the fidelity of ``soxr``'s HQ mode while
    depending only on numpy.

    Args:
        audio: Float32 mono audio samples.
        src_rate: Source sample rate in Hz.
        dst_rate: Destination sample rate in Hz.
        zero_crossings: Sinc zero-crossings retained on each side of each output
            sample (controls fidelity vs. cost). 12 gives good quality.
        chunk_size: Maximum number of output samples processed per buffer, to
            bound peak memory. Defaults to 262144 (~50 MB with 25 taps).

    Returns:
        A new ``float32`` array containing audio at ``dst_rate``.

    Raises:
        ValueError: If either sample rate is not positive.
    """
    src_rate = float(src_rate)
    dst_rate = float(dst_rate)
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Sample rates must be positive.")

    audio = np.asarray(audio, dtype=np.float32)

    if src_rate == dst_rate:
        return audio.copy()

    # Work on a single channel (take the first if multi-channel was provided).
    if audio.ndim > 1:
        audio = audio[:, 0]

    if audio.size == 0:
        return audio.copy()

    ratio = src_rate / dst_rate  # input samples per output sample

    # Anti-aliasing cutoff relative to input Nyquist. When downsampling
    # (ratio > 1) this keeps the pass band under the output Nyquist.
    cutoff = min(1.0, 1.0 / ratio)

    n_in = audio.size
    n_out = max(1, int(math.ceil(n_in / ratio)))

    offsets_i = np.arange(-zero_crossings, zero_crossings + 1, dtype=np.int64)
    offsets = offsets_i.astype(np.float64)

    # Hann window across the retained taps, centered on each output position.
    window = 0.5 + 0.5 * np.cos(math.pi * (offsets / float(zero_crossings)))

    result = np.empty(n_out, dtype=np.float32)
    out_positions = np.arange(n_out, dtype=np.float64) * ratio

    for start in range(0, n_out, chunk_size):
        end = min(start + chunk_size, n_out)
        pos = out_positions[start:end]
        base = np.floor(pos).astype(np.int64)
        idx = base[:, None] + offsets_i[None, :]  # (chunk, taps) integer
        distance = pos[:, None] - idx.astype(np.float64)
        # Cutoff-windowed sinc.
        kernel = np.sinc(cutoff * distance) * cutoff * window[None, :]
        valid = (idx >= 0) & (idx < n_in)
        safe_idx = np.clip(idx, 0, n_in - 1)
        contrib = np.where(valid, audio[safe_idx], 0.0)
        result[start:end] = np.sum(kernel * contrib, axis=1)

    return result.astype(np.float32)
