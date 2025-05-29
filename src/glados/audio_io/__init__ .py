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

from .audio_io import AudioIO
from .factory import create_audio_io
from .sounddevice_io import SoundDeviceAudioIO

# from .websocket_io import WebSocketAudioIO

__all__ = [
    'AudioIO',
    'SoundDeviceAudioIO',
    # 'WebSocketAudioIO',
    'create_audio_io',
]
