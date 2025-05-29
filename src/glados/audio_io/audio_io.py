from abc import ABC, abstractmethod
import queue

import numpy as np
from numpy.typing import NDArray


class AudioIO(ABC):
    """Abstract interface for audio input/output operations.
    
    This class defines a common interface for different audio input/output
    implementations, allowing the Glados engine to work with different
    audio backends (such as sounddevice or WebSockets) interchangeably.
    
    Implementations of this interface must provide methods for:
    - Starting and stopping audio input capture
    - Playing audio data
    - Checking audio playback status
    - Accessing a queue of captured audio samples
    
    The interface is designed to support real-time audio processing
    with voice activity detection (VAD) for intelligent speech recognition.
    """
    
    @abstractmethod
    def start_listening(self) -> None:
        """Start capturing audio input.
        
        Initializes and starts the audio input capture process. Implementation
        details depend on the specific audio backend being used.
        
        Raises:
            RuntimeError: If audio input capture cannot be started
        """
        pass
    
    @abstractmethod
    def stop_listening(self) -> None:
        """Stop capturing audio input.
        
        Stops any ongoing audio input capture and releases associated resources.
        This method should be called when audio input is no longer needed or
        before application shutdown.
        """
        pass
    
    @abstractmethod
    def start_speaking(self, audio_data: NDArray[np.float32], sample_rate: int, text: str = "") -> None:
        """Play audio data through the audio output device or stream.
        
        Parameters:
            audio_data: The audio data to play as a numpy float32 array
            sample_rate: The sample rate of the audio data in Hz
            text: Optional text associated with the audio
            
        Raises:
            RuntimeError: If audio playback cannot be initiated
        """
        pass
    
    @abstractmethod
    def check_if_speaking(self) -> bool:
        """Check if audio is currently being played.
        
        Returns:
            bool: True if audio is currently playing, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_speaking(self) -> None:
        """Stop any ongoing audio playback.
        
        Interrupts the current audio playback if any is in progress. This method
        should clean up any resources associated with the playback.
        """
        pass
    
    @abstractmethod
    def _get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence.
        
        The queue contains tuples of (audio_sample, vad_confidence) where:
        - audio_sample is a numpy array containing a chunk of audio data
        - vad_confidence is a boolean indicating whether voice activity is detected
        
        Returns:
            queue.Queue: A thread-safe queue containing audio samples with VAD results
        """
        pass
