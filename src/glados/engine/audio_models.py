"""GLaDOS Engine Audio Models - Data structures for audio processing.

This module contains dataclasses and models used throughout the GLaDOS audio
processing pipeline. These models provide standardized containers for audio
data and associated metadata.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioMessage:
    """Container for audio data with associated text and processing metadata.
    
    This dataclass carries audio samples through the TTS and audio processing
    pipeline, maintaining the relationship between generated audio and its
    source text. It also includes end-of-stream signaling for proper pipeline
    management.
    
    Attributes:
        audio (NDArray[np.float32]): Audio samples as a numpy array of 32-bit
            floating point values. Expected to be mono audio at the sample
            rate specified by the TTS model.
        text (str): The text content that generated this audio, used for
            logging, interruption handling, and conversation history.
        is_eos (bool): End-of-stream flag indicating this is the final
            message in a response sequence. Used to trigger conversation
            history updates and processing state changes. Defaults to False.
    
    Notes:
        - Audio arrays should be mono (single channel) at the TTS sample rate
        - Empty audio arrays with is_eos=True are used for signaling only
        - Text content is preserved for accurate interruption handling
        - Used throughout the audio processing and playback pipeline
        
    Example:
        >>> # Regular audio message
        >>> audio_msg = AudioMessage(
        ...     audio=np.array([0.1, 0.2, 0.0, -0.1], dtype=np.float32),
        ...     text="Hello there",
        ...     is_eos=False
        ... )
        >>> 
        >>> # End-of-stream signal
        >>> eos_msg = AudioMessage(
        ...     audio=np.array([], dtype=np.float32),
        ...     text="",
        ...     is_eos=True
        ... )
    """
    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False
