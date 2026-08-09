"""Audio input/output backends.

This package provides an abstraction layer for audio input/output so the Glados
engine can run on any backend interchangeably (local hardware, WebSocket, ...).

Classes:
    AudioIO: Abstract base class for audio I/O backends
    SoundDeviceAudioIO: Local hardware via the sounddevice library
    WebsocketAudioIO: Network streamed audio via a WebSocket server

Functions:
    get_audio_system: Factory that returns an AudioIO instance by name
"""

import queue
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .base import AudioIO
from .vad import VAD


class AudioProtocol(Protocol):
    """Legacy structural interface accepted by the engine.

    Implementations don't have to inherit :class:`AudioIO`; the ABC is
    available to new in-tree and third-party backends as a convenience.
    """

    def start_listening(self) -> None:
        """Start capturing microphone samples."""
        ...

    def stop_listening(self) -> None:
        """Stop capturing microphone samples."""
        ...

    def start_speaking(self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = "") -> None:
        """Queue audio for playback."""
        ...

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """Wait for playback and return interruption state and progress."""
        ...

    def check_if_speaking(self) -> bool:
        """Return whether audio is currently playing."""
        ...

    def stop_speaking(self) -> None:
        """Interrupt current playback."""
        ...

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Return queued microphone samples and VAD decisions."""
        ...


# Factory function
def get_audio_system(
    backend_type: str = "sounddevice",
    backend_options: dict[str, Any] | None = None,
    vad_threshold: float | None = None,
) -> AudioIO:
    """
    Factory function to get an instance of an audio I/O system based on the specified backend type.

    Parameters:
        backend_type (str): The type of audio backend to use:
            - "sounddevice": Uses the sounddevice library for local audio I/O
            - "websocket": Network-based audio I/O (starts a WebSocket server)
        backend_options (dict | None): Backend-specific options.
            - "sounddevice": No options are allowed.
            - "websocket" accepted options:
                - server: listen address (default: 127.0.0.1)
                - port: listen port (default: 5051)
                - rooms: enable multi-microphone room choreography (default: False)
                - segregate_speakers: only play to speakers in the active room
                - default_room_tag: room tag fallback (default: "office")
                - speaker_sync_delay_ms: add to start time for speaker sync (default: 250)
                - mic_max_silence_chunks: silent chunks before a mic yields (default: 10)
                - mic_queue_max_chunks: buffered 32 ms chunks before dropping oldest (default: 256)
        vad_threshold (float | None): Optional threshold for voice activity detection

    Returns:
        AudioIO: An instance of the requested audio I/O system

    Raises:
        ValueError: If the specified backend type is not supported
    """
    if backend_type == "sounddevice":
        from .sounddevice_io import SoundDeviceAudioIO

        if backend_options is not None:
            raise ValueError("Sounddevice backend does not support options")

        return SoundDeviceAudioIO(
            vad_threshold=vad_threshold,
        )
    elif backend_type == "websocket":
        from .websocket_io import WebsocketAudioIO

        return WebsocketAudioIO(vad_threshold=vad_threshold, options=backend_options)
    else:
        raise ValueError(f"Unsupported audio backend type: {backend_type}")


__all__ = [
    "VAD",
    "AudioIO",
    "AudioProtocol",
    "get_audio_system",
]
