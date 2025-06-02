from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioMessage:
    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False


@dataclass
class AudioInputMessage:
    audio_sample: NDArray[np.float32]
    vad_confidence: bool = False