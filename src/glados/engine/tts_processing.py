"""GLaDOS Engine TTS Processing - Text-to-speech processing and audio generation.

This module handles text-to-speech conversion, including text preprocessing,
speech synthesis, performance monitoring, and audio queue management for
the GLaDOS voice assistant pipeline.
"""

import queue
import time

import numpy as np
from loguru import logger

from .audio_models import AudioMessage


class TTSProcessingMixin:
    """Mixin class containing TTS processing methods for the Glados class.
    
    This mixin separates text-to-speech processing concerns from the main Glados class,
    providing methods for:
    - Text-to-speech conversion and audio generation
    - Performance monitoring and timing metrics
    - Audio queue management and EOS signal handling
    - Integration with the spoken text converter for natural speech
    
    Note: This is designed as a mixin to be used with the main Glados class.
    It expects certain attributes and methods to be available from the parent class.
    """

    def process_tts_thread(self) -> None:
        """Process text-to-speech conversion in a dedicated background thread.

        This method runs continuously as a daemon thread, retrieving text from the TTS
        queue and converting it to speech audio. It handles the complete TTS pipeline
        including text preprocessing, speech synthesis, performance monitoring, and
        audio queue management.

        The method processes several types of input:
        - Regular text strings for speech synthesis
        - End-of-stream ("<EOS>") signals to indicate completion
        - Empty strings which are logged as warnings

        Key Behaviors:
        - Continuously monitors TTS queue for new text to process
        - Converts text to spoken form using SpokenTextConverter
        - Generates speech audio using configured TTS model
        - Measures and logs TTS performance metrics
        - Queues audio messages for playback pipeline
        - Handles end-of-stream signaling for conversation completion

        Performance Monitoring:
        - Tracks TTS inference time for optimization
        - Calculates audio duration for timing analysis
        - Logs comprehensive performance metrics

        Side Effects:
        - Puts AudioMessage objects into the audio queue
        - Logs TTS performance and processing information
        - Processes text through spoken text converter for natural speech

        Error Handling:
        - Handles empty queue timeouts gracefully
        - Continues operation despite individual processing errors
        - Logs warnings for empty or invalid text input

        Notes:
        - Runs until shutdown event is triggered
        - Uses timeout-based queue processing to allow shutdown detection
        - Audio generation only occurs for non-empty text after conversion
        - EOS signals are passed through without audio generation
        """
        while not self.shutdown_event.is_set():
            try:
                # Retrieve text from TTS queue with timeout
                generated_text = self.tts_queue.get(timeout=self.PAUSE_TIME)

                # Handle end-of-stream signal
                if generated_text == "<EOS>":
                    # Signal audio pipeline that response is complete
                    self.audio_queue.put(AudioMessage(np.array([]), "", is_eos=True))
                    
                elif not generated_text:
                    # Log warning for empty text input
                    logger.warning("Empty string sent to TTS")
                    
                else:
                    # Process regular text for speech synthesis
                    logger.info(f"LLM text: {generated_text}")

                    # Start performance timing
                    start_time = time.time()
                    
                    # Convert text to spoken form (handles abbreviations, numbers, etc.)
                    spoken_text = self._stc.text_to_spoken(generated_text)
                    
                    # Generate speech audio using TTS model
                    audio = self._tts.generate_speech_audio(spoken_text)
                    
                    # Calculate and log performance metrics
                    inference_time = time.time() - start_time
                    audio_duration = len(audio) / self._tts.sample_rate
                    
                    logger.info(
                        f"TTS Complete - Inference: {inference_time:.2f}s, "
                        f"Audio Length: {audio_duration:.2f}s"
                    )

                    # Queue audio for playback if generation was successful
                    if len(audio) > 0:
                        self.audio_queue.put(AudioMessage(audio, spoken_text))

            except queue.Empty:
                # No text available, continue monitoring
                pass