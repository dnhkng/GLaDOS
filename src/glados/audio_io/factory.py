from collections.abc import Callable
from typing import Literal

from .audio_io import AudioIO
from .sounddevice_io import SoundDeviceAudioIO
from .websocket_io import WebSocketAudioIO


def create_audio_io(
    io_type: Literal["sounddevice", "websocket"],
    sample_rate: int,
    vad_model: Callable,
    vad_size: int = 32,
    vad_threshold: float = 0.8,
    websocket_host: str = "0.0.0.0",
    websocket_port: int = 8000,
    session_id: str | None = None,
) -> AudioIO:
    """Create and return an AudioIO instance of the specified type.

    This factory function simplifies the creation of different AudioIO implementations
    by providing a unified interface with appropriate default values. It handles
    the instantiation details for each implementation type.

    Parameters:
        io_type: Type of AudioIO to create ('sounddevice' or 'websocket')
        sample_rate: Sample rate for audio processing in Hz
        vad_model: Voice Activity Detection model callable
        vad_size: Size of each VAD window in milliseconds (default: 32)
        vad_threshold: Threshold for VAD detection (default: 0.8)
        websocket_host: Host address for WebSocket server (default: '0.0.0.0')
        websocket_port: Port for WebSocket server (default: 8000)
        session_id: Optional session identifier for multi-user setups

    Returns:
        An instance of AudioIO implementation

    Raises:
        ValueError: If io_type is not recognized
        ImportError: If dependencies for the requested implementation are not available

    Examples:
        # Create a SoundDeviceAudioIO instance
        audio_io = create_audio_io("sounddevice", 16000, vad_model)

        # Create a WebSocketAudioIO instance
        audio_io = create_audio_io(
            "websocket",
            16000,
            vad_model,
            websocket_port=8080,
            session_id="user123"
        )
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if vad_size <= 0:
        raise ValueError("VAD size must be positive")
    if not 0 <= vad_threshold <= 1:
        raise ValueError("VAD threshold must be between 0 and 1")

    if io_type == "sounddevice":
        return SoundDeviceAudioIO(
            sample_rate=sample_rate, vad_size=vad_size, vad_model=vad_model, vad_threshold=vad_threshold
        )
    elif io_type == "websocket":
        return WebSocketAudioIO(
            sample_rate=sample_rate,
            vad_model=vad_model,
            host=websocket_host,
            port=websocket_port,
            vad_threshold=vad_threshold,
            session_id=session_id,
        )
    else:
        raise ValueError(f"Unknown AudioIO type: {io_type}")
