"""Mel spectrogram calculation module

This module provides a class to compute mel spectrograms from audio signals.
The implementation is based on NVIDIA's implementation in the NeMo library.
The mel filterbank is created to match librosa's implementation.

https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py

Example:
Create a MelSpectrogramCalculator object and compute a mel spectrogram:

    >>> import numpy as np
    >>> from glados.core.mel_spectrogram import MelSpectrogramCalculator
    >>> mel_spectrogram_calculator = MelSpectrogramCalculator()
    >>> audio = np.random.randn(16000)
    >>> mel_spectrogram = mel_spectrogram_calculator.compute(audio)

Attributes:
    sr: Sample rate
    n_mels: Number of mel bins
    n_fft: FFT size
    hop_length: Hop length
    win_length: Window length
    fmin: Minimum frequency
    fmax: Maximum frequency
    preemph: Preemphasis coefficient
    mag_power: Power to raise magnitude to
    normalize: Normalization type
    dither: Dithering strength
    log_zero_guard_value: Zero guard value for log scaling

Methods:
    _create_mel_filterbank: Create mel filterbank
    _apply_preemphasis: Apply preemphasis filter
    _normalize_spectrogram: Apply per-feature normalization
    compute: Compute mel spectrogram matching NVIDIA's implementation

Functions:
    _extract_windows: Extract and window frames from audio signal
"""

from numba import jit
import numpy as np


@jit(nopython=True)
def _extract_windows(
    audio_padded: np.ndarray,
    window: np.ndarray,
    n_fft: int,
    hop_length: int,
    n_frames: int,
) -> np.ndarray:
    """
    Extract and window frames from a padded audio signal.
    
    Args:
        audio_padded (np.ndarray): Padded audio signal with sufficient length to extract frames
        window (np.ndarray): Pre-computed window function to apply to each frame
        n_fft (int): Size of the Fast Fourier Transform (FFT) window
        hop_length (int): Number of samples between successive frames
        n_frames (int): Total number of frames to extract from the audio signal
    
    Returns:
        np.ndarray: 2D array of extracted and windowed frames, with shape (n_frames, n_fft)
    """
    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    for t in range(n_frames):
        start = t * hop_length
        frames[t] = audio_padded[start : start + n_fft] * window
    return frames


class MelSpectrogramCalculator:
    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        fmin: float = 0.0,
        fmax: float | None = None,
        preemph: float = 0.97,
        mag_power: float = 2.0,
        normalize: str = "per_feature",
        dither: float = 1e-5,
        log_zero_guard_value: float = 2**-24,
    ) -> None:
        """
        Initialize a MelSpectrogramCalculator with configurable audio processing parameters.
        
        Args:
            sr (int, optional): Audio sample rate in Hz. Defaults to 16000.
            n_mels (int, optional): Number of mel frequency bins. Defaults to 80.
            n_fft (int, optional): Size of the Fast Fourier Transform window. Defaults to 400.
            hop_length (int, optional): Number of samples between successive frames. Defaults to 160.
            win_length (int, optional): Length of each window frame. Defaults to 400.
            fmin (float, optional): Minimum frequency for mel filterbank. Defaults to 0.0.
            fmax (float | None, optional): Maximum frequency for mel filterbank. Defaults to half the sample rate.
            preemph (float, optional): Preemphasis filter coefficient. Defaults to 0.97.
            mag_power (float, optional): Power to apply to magnitude spectrogram. Defaults to 2.0.
            normalize (str, optional): Normalization method for mel spectrogram. Defaults to "per_feature".
            dither (float, optional): Strength of noise added to prevent quantization artifacts. Defaults to 1e-5.
            log_zero_guard_value (float, optional): Small value to prevent log(0) errors. Defaults to 2**-24.
        
        Attributes:
            mel_filterbank (np.ndarray): Pre-computed mel filterbank matrix.
            window (np.ndarray): Pre-computed Hanning window function.
        """
        if fmax is None:
            fmax = float(sr) / 2

        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.preemph = preemph
        self.mag_power = mag_power
        self.normalize = normalize
        self.dither = dither
        self.log_zero_guard_value = log_zero_guard_value

        # Pre-compute constants
        self.mel_filterbank = self._create_mel_filterbank(fmin, fmax)
        self.window = np.hanning(n_fft).astype(np.float32)

    def _create_mel_filterbank(self, fmin: float, fmax: float) -> np.ndarray:
        """
        Create a mel filterbank matrix matching librosa's implementation.
        
        This method generates a mel filterbank matrix that transforms linear frequency bins to mel-scale frequency bins. It follows the triangular filter design used in librosa, creating a set of overlapping triangular filters across the frequency spectrum.
        
        Args:
            fmin (float): Minimum frequency for the mel filterbank
            fmax (float): Maximum frequency for the mel filterbank
        
        Returns:
            np.ndarray: A mel filterbank matrix of shape (n_mels, n_freqs) with triangular filter weights
        
        Notes:
            - The method uses the standard mel frequency conversion formula
            - Filters are triangular and normalized to preserve total energy
            - Supports custom minimum and maximum frequency ranges
        """
        # Center frequencies of each FFT bin
        n_freqs = int(1 + self.n_fft // 2)
        fftfreqs = np.linspace(0, self.sr / 2, n_freqs)

        # Mel points
        mel_f = 2595.0 * np.log10(1.0 + np.array([fmin, fmax]) / 700.0)
        mel_points = np.linspace(mel_f[0], mel_f[1], self.n_mels + 2)
        freq_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # Initialize weights matrix
        weights = np.zeros((self.n_mels, n_freqs), dtype=np.float32)

        # Vectorized filter creation
        for i in range(self.n_mels):
            f_left = freq_points[i]
            f_center = freq_points[i + 1]
            f_right = freq_points[i + 2]

            # Left side of triangle
            indices = (fftfreqs >= f_left) & (fftfreqs <= f_center)
            weights[i, indices] = (fftfreqs[indices] - f_left) / (f_center - f_left)

            # Right side of triangle
            indices = (fftfreqs >= f_center) & (fftfreqs <= f_right)
            weights[i, indices] = (f_right - fftfreqs[indices]) / (f_right - f_center)

        # Normalize
        enorm = 2.0 / (freq_points[2:] - freq_points[:-2])
        weights *= enorm[:, np.newaxis]

        return weights.astype(np.float32)

    def _apply_preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply a preemphasis filter to the audio signal to enhance higher frequencies.
        
        This method applies a first-order high-pass filter to the audio signal, which amplifies higher frequency components. The preemphasis helps to counteract the natural spectral slope of speech and audio signals.
        
        Args:
            audio (np.ndarray): Input audio signal as a NumPy array.
        
        Returns:
            np.ndarray: Preemphasized audio signal with enhanced high-frequency content.
        
        Notes:
            - The preemphasis is performed using the formula: y[n] = x[n] - preemph * x[n-1]
            - The first sample is preserved as-is to maintain the original signal's length
            - The preemphasis coefficient (self.preemph) controls the strength of high-frequency enhancement
        """
        return np.concatenate([audio[:1], audio[1:] - self.preemph * audio[:-1]])

    def _normalize_spectrogram(self, mel_spec: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Apply per-feature normalization to the mel spectrogram.
        
        Normalizes the mel spectrogram using per-feature mean and variance standardization when 
        the normalization type is set to "per_feature". This method helps to center and scale 
        the spectrogram features, which can improve subsequent machine learning model performance.
        
        Args:
            mel_spec (np.ndarray): Input mel spectrogram to be normalized
            seq_len (int): Number of frames in the spectrogram
        
        Returns:
            np.ndarray: Normalized mel spectrogram with zero mean and unit variance per feature
        
        Notes:
            - Uses sample variance calculation with Bessel's correction (dividing by seq_len - 1)
            - Adds a small epsilon (1e-5) to variance to prevent division by zero
            - Only applies normalization if self.normalize is set to "per_feature"
            - Preserves the original shape of the input mel spectrogram
        """
        if self.normalize == "per_feature":
            mean = np.sum(mel_spec, axis=1, keepdims=True) / seq_len
            var = np.sum((mel_spec - mean) ** 2, axis=1, keepdims=True) / (seq_len - 1)
            mel_spec = (mel_spec - mean) / (np.sqrt(var) + 1e-5)
        return mel_spec

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from an audio signal, matching NVIDIA's implementation.
        
        Args:
            audio (np.ndarray): Input audio signal as a numpy array.
        
        Returns:
            np.ndarray: Mel spectrogram computed from the input audio.
        
        Details:
            This method performs several preprocessing and transformation steps:
            1. Converts input to float32
            2. Applies optional dithering to reduce quantization noise
            3. Applies optional preemphasis filter
            4. Pads audio signal with reflection
            5. Extracts and windows audio frames
            6. Computes Short-Time Fourier Transform (STFT)
            7. Calculates power spectrum with configurable power
            8. Applies mel filterbank
            9. Applies logarithmic scaling with zero guard
            10. Optionally normalizes the spectrogram
        
        Notes:
            - Dithering strength controlled by `self.dither`
            - Preemphasis applied if `self.preemph` is not None
            - Logarithmic scaling uses `self.log_zero_guard_value`
            - Normalization controlled by `self.normalize`
        """
        # Convert to float32
        audio = np.asarray(audio, dtype=np.float32)

        # Apply dithering
        if self.dither > 0:
            audio = audio + self.dither * np.random.randn(*audio.shape)

        # Apply preemphasis
        if self.preemph is not None:
            audio = self._apply_preemphasis(audio)

        # Pad audio
        padding = int(self.n_fft // 2)
        audio_padded = np.pad(audio, (padding, padding), mode="reflect")

        # Calculate frames
        n_frames = 1 + (len(audio_padded) - self.n_fft) // self.hop_length

        # Extract and window frames
        frames = _extract_windows(audio_padded, self.window, self.n_fft, self.hop_length, n_frames)

        # Compute STFT
        stft = np.fft.rfft(frames, axis=1).T

        # Power spectrum with configurable power
        power_spec = (np.abs(stft) ** self.mag_power).astype(np.float32)

        # Apply mel filterbank
        mel_spec = self.mel_filterbank @ power_spec

        # Log scaling with zero guard
        mel_spec = np.log(mel_spec + self.log_zero_guard_value)

        # Normalize if requested
        if self.normalize:
            mel_spec = self._normalize_spectrogram(mel_spec, n_frames)

        return mel_spec
