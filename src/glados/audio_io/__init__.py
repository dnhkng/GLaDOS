"""Audio input/output components.

This package provides an abstraction layer for audio input and output operations,
allowing the Glados engine to work with different audio backends interchangeably.

Classes:
    AudioIO: Abstract interface for audio input/output operations
    SoundDeviceAudioIO: Implementation using the sounddevice library
    WebSocketAudioIO: Implementation using WebSockets for network streaming

Functions:
    create_audio_io: Factory function to create AudioIO instances
"""

import queue
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .vad import VAD


class AudioProtocol(Protocol):
    def __init__(self, model_path: str, *args: str, **kwargs: dict[str, str]) -> None: ...
    def start_listening(self) -> None: ...
    def stop_listening(self) -> None: ...
    def start_speaking(self, audio_data: NDArray[np.float32], sample_rate: int, text: str = "") -> None: ...
    def measure_percentage_spoken(self, total_samples: int) -> tuple[bool, int]: ...
    def check_if_speaking(self) -> bool: ...
    def stop_speaking(self) -> None: ...
    def _get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]: ...


# Factory function
def get_audio_system(
    backend_type: str = "sounddevice", vad_threshold: float | None = None
) -> AudioProtocol:  # Return type is now a Union of concrete types
    """
    Factory function to get an instance of an audio transcriber based on the specified engine type.

    Parameters:
        backend_type (str): The type of audio backend to use:
            - "sounddevice": Connectionist Temporal Classification model (faster, good accuracy)
            - "websocket": ToDo
        **kwargs: Additional keyword arguments to pass to the transcriber constructor

    Returns:
        TranscriberProtocol: An instance of the requested audio transcriber

    Raises:
        ValueError: If the specified engine type is not supported
    """
    if backend_type == "sounddevice":
        from .sounddevice_io import SoundDeviceAudioIO

        return SoundDeviceAudioIO(
            vad_threshold=vad_threshold,
        )
    elif backend_type == "websocket":
        raise ValueError("WebSocket audio backend is not yet implemented.")
    else:
        raise ValueError(f"Unsupported ASR engine type: {backend_type}")


__all__ = [
    "VAD",
    "AudioProtocol",
    "get_audio_system",
]
