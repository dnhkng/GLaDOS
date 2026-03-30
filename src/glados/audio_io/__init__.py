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
from typing import Protocol, Any

import numpy as np
from numpy.typing import NDArray

from .vad import VAD


class AudioProtocol(Protocol):
    def __init__(self, vad_threshold: float | None = None) -> None: ...
    def start_listening(self) -> None: ...
    def stop_listening(self) -> None: ...
    def start_speaking(
        self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = "", wait: bool = False
    ) -> None: ...
    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]: ...
    def check_if_speaking(self) -> bool: ...
    def stop_speaking(self) -> None: ...
    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]: ...


# Factory function
def get_audio_system(backend_type: str = "sounddevice", backend_options: dict[str, Any] | None = None, vad_threshold: float | None = None) -> AudioProtocol:
    """
    Factory function to get an instance of an audio I/O system based on the specified backend type.

    Parameters:
        backend_type (str): The type of audio backend to use:
            - "sounddevice": Uses the sounddevice library for local audio I/O
            - "websocket": Network-based audio I/O
        backend_options: Options for the specified backend.
            - "sounddevice": No options are allowed.
            - "websocket": The following options are allowed:
                - server: Websocket listening address (default: 0.0.0.0)
                - port: Websocket listening port (default: 5050)
                - speaker_sync_delay_ms: Milliseconds to add to each speak start time to account for speaker synchronisation (default: 250)
                - mic_max_silence_chunks: How many consecutive VAD chunks must be silent so that the current microphone relinquishes control (default: 10)
        vad_threshold (float | None): Optional threshold for voice activity detection

    Returns:
        AudioProtocol: An instance of the requested audio I/O system

    Raises:
        ValueError: If the specified backend type is not supported
    """
    if backend_type == "sounddevice":
        from .sounddevice_io import SoundDeviceAudioIO

        if backend_options is not None:
            raise ValueError("Sounddevice backend does not support options")

        # noinspection PyTypeChecker
        return SoundDeviceAudioIO(
            vad_threshold=vad_threshold,
        )
    elif backend_type == "websocket":
        from .websocket_io import WebsocketAudioIO

        # noinspection PyTypeChecker
        return WebsocketAudioIO(vad_threshold=vad_threshold, options=backend_options)
    else:
        raise ValueError(f"Unsupported audio backend type: {backend_type}")


__all__ = [
    "VAD",
    "AudioProtocol",
    "get_audio_system",
]
