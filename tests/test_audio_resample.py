import numpy as np
import pytest

from glados.audio_io.resample import resample


def _sine(rate: float, duration: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * duration))
    return (np.sin(2 * np.pi * freq * t / rate)).astype(np.float32)


def test_same_rate_returns_identical_copy() -> None:
    x = _sine(22050.0, 0.2)
    y = resample(x, 22050, 22050)
    assert y is not x
    np.testing.assert_array_equal(y, x)


def test_upsample_ratio_and_signal_preserved() -> None:
    x = _sine(22050.0, 0.5)
    y = resample(x, 22050, 44100)
    assert len(y) == 2 * len(x)
    # In the interior, every-other sample of the 2x upsample approximates the
    # original samples (windowed-sinc reconstruction), away from edge transients.
    start = 2 * 2000
    stop = len(y) - 2 * 2000
    err = np.abs(y[start:stop:2] - x[2000 : len(x) - 2000]).max()
    assert err < 0.01


def test_downsample_preserves_amplitude_and_tone() -> None:
    x = _sine(44100.0, 0.5)
    y = resample(x, 44100, 22050)
    assert len(y) == len(x) // 2
    assert abs(np.sqrt(np.mean(y**2)) - 1.0 / np.sqrt(2.0)) < 0.05
    assert abs(float(np.abs(y).max()) - 1.0) < 0.05


def test_multichannel_input_flattens_to_mono() -> None:
    x = np.tile(_sine(22050.0, 0.1)[:, None], (1, 2))
    y = resample(x, 22050, 44100)
    assert y.ndim == 1
    assert len(y) == 2 * x.shape[0]


def test_empty_and_invalid_rates() -> None:
    assert resample(np.array([], dtype=np.float32), 22050, 44100).size == 0
    with pytest.raises(ValueError):
        resample(_sine(22050.0, 0.1), 0, 44100)
    with pytest.raises(ValueError):
        resample(_sine(22050.0, 0.1), 22050, -44100)
