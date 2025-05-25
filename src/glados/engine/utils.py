"""GLaDOS Engine Utils - Utility functions, properties, and helper methods.

This module contains utility functions, property accessors, and helper methods
that support the GLaDOS voice assistant engine. It includes conversation
management, startup functions, and audio callback utilities.
"""

import numpy as np
import sounddevice as sd
from sounddevice import CallbackFlags
from loguru import logger

from .config import GladosConfig


class UtilsMixin:
    """Mixin class containing utility methods and properties for the Glados class.
    
    This mixin provides utility functionality including:
    - Conversation history access through properties
    - Audio input stream callback functions
    - Helper methods for common operations
    
    Note: This is designed as a mixin to be used with the main Glados class.
    It expects certain attributes and methods to be available from the parent class.
    """

    @property
    def messages(self) -> list[dict[str, str]]:
        """Retrieve the current conversation message history.

        This property provides read-only access to the conversation history,
        which contains the complete dialogue between the user and assistant
        in OpenAI chat message format.

        Returns:
        - list[dict[str, str]]: List of message dictionaries with 'role' and 'content' keys.
          Each message represents one turn in the conversation with roles like 'system',
          'user', or 'assistant'.

        Notes:
        - Messages are stored in chronological order
        - Includes system prompts, user inputs, and assistant responses
        - Used for maintaining conversation context across interactions
        - Format is compatible with OpenAI chat completion APIs

        Example:
        >>> glados = Glados.from_yaml('config.yaml')
        >>> print(glados.messages)
        [
            {'role': 'system', 'content': 'You are a helpful AI assistant...'},
            {'role': 'user', 'content': 'Hello, how are you?'},
            {'role': 'assistant', 'content': 'I am doing well, thank you!'}
        ]
        """
        return self._messages

    def create_audio_callback(self):
        """Create audio input stream callback function for sounddevice.

        This method returns a properly configured callback function for processing
        audio input from the sounddevice input stream. The callback handles voice
        activity detection and queues audio samples for processing.

        Returns:
        - Callable: Audio callback function configured for this Glados instance

        Notes:
        - The callback is bound to this specific Glados instance
        - Handles real-time audio processing and VAD
        - Queues audio samples with confidence values for processing pipeline
        """
        def audio_callback_for_sd_input_stream(
            indata: np.dtype[np.float32],
            frames: int,
            time: sd.CallbackStop,
            status: CallbackFlags,
        ) -> None:
            """Callback function for processing audio input from sounddevice stream.

            This callback function handles incoming audio samples, performs voice activity
            detection (VAD), and queues the processed audio data for further analysis.
            It runs in real-time as part of the audio input stream processing.

            Parameters:
            - indata (np.ndarray): Input audio data from the sounddevice stream
            - frames (int): Number of audio frames in the current chunk  
            - time (sd.CallbackStop): Timing information for the audio callback
            - status (CallbackFlags): Status flags indicating stream conditions

            Side Effects:
            - Processes audio through VAD model
            - Puts audio samples and confidence values into sample queue
            - Converts multi-channel audio to single channel if necessary

            Notes:
            - Runs in audio thread context with real-time constraints
            - Must complete processing quickly to avoid audio dropouts
            - Copies input data to prevent modification of stream buffers
            - VAD confidence is computed and thresholded for voice detection
            """
            # Copy and ensure single-channel audio
            data = np.array(indata).copy().squeeze()
            
            # Perform voice activity detection
            vad_value = self._vad_model(np.expand_dims(data, 0))
            vad_confidence = vad_value > self.VAD_THRESHOLD
            
            # Queue audio sample with VAD result
            self._sample_queue.put((data, bool(vad_confidence)))

        return audio_callback_for_sd_input_stream



# Make start function available at module level for convenience
__all__ = ["UtilsMixin"]