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

from typing import Any

from .base import AudioIO
from .vad import VAD

# Backwards-compatible alias: AudioProtocol now refers to the AudioIO ABC.
AudioProtocol = AudioIO
"""Alias for :class:`AudioIO` kept for callers that previously used a Protocol."""

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
