"""Abstract audio input/output backend interface.

This is the single contract every audio backend implements, so the Glados
engine can swap between local hardware (sounddevice), a network backend
(WebSocket), or any future backend without changing engine code.
"""
from abc import ABC, abstractmethod
import queue

import numpy as np
from numpy.typing import NDArray


class AudioIO(ABC):
    """Abstract interface for audio input and output operations.

    Implementations provided by this package:
      - SoundDeviceAudioIO: local microphone/speaker via sounddevice
      - WebsocketAudioIO: network streamed audio via WebSockets
    """

    @abstractmethod
    def start_listening(self) -> None:
        """Start capturing audio input and queueing VAD-annotated samples."""

    @abstractmethod
    def stop_listening(self) -> None:
        """Stop capturing audio input."""

    @abstractmethod
    def start_speaking(
        self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = ""
    ) -> None:
        """Queue audio for playback (non-blocking)."""

    @abstractmethod
    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """Block until playback finishes (or is interrupted) and report progress.

        Returns:
            tuple[bool, int]: (interrupted, percentage_played 0-100)
        """

    @abstractmethod
    def check_if_speaking(self) -> bool:
        """Return True while audio is currently being played."""

    @abstractmethod
    def stop_speaking(self) -> None:
        """Interrupt any ongoing playback."""

    @abstractmethod
    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Return the thread-safe queue of ``(audio_samples, vad_confidence)``."""
