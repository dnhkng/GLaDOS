from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioMessage:
    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False
