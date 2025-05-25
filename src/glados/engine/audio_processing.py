"""GLaDOS Engine Audio Processing - Audio input handling and processing pipeline.

This module handles all aspects of audio input processing including voice activity
detection, speech recognition, wake word detection, audio buffering, and playback
management. It contains the core audio processing loop and state management for
the voice assistant.
"""

import copy
import queue
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd
from sounddevice import CallbackFlags
from Levenshtein import distance
from loguru import logger

from .audio_models import AudioMessage


class AudioProcessingMixin:
    """Mixin class containing audio processing methods for the Glados class.
    
    This mixin separates audio processing concerns from the main Glados class,
    providing methods for:
    - Audio input stream management
    - Voice activity detection processing
    - Speech recognition and wake word detection
    - Audio playback and interruption handling
    - Buffer management and state tracking
    
    Note: This is designed as a mixin to be used with the main Glados class.
    It expects certain attributes and methods to be available from the parent class.
    """

    def start_listen_event_loop(self) -> None:
        """Start the voice assistant's listening event loop for continuous audio processing.

        This method initializes the audio input stream and enters an infinite loop to handle
        incoming audio samples. The loop retrieves audio samples and their voice activity
        detection (VAD) confidence from a queue and processes each sample using the
        `_handle_audio_sample` method.

        Behavior:
        - Starts the audio input stream for real-time audio capture
        - Logs successful initialization of audio modules
        - Enters an infinite listening loop with timeout-based queue processing
        - Retrieves audio samples from the sample queue with VAD confidence
        - Processes each audio sample through the audio processing pipeline
        - Handles keyboard interrupts and shutdown events gracefully

        Side Effects:
        - Starts the sounddevice input stream
        - Begins continuous audio processing in the main thread
        - Logs operational status and listening state
        - Sets shutdown event on termination

        Raises:
        - KeyboardInterrupt: Allows graceful termination of the listening loop
        - Exception: Any errors during audio processing are logged but don't crash the loop

        Notes:
        - Uses timeout-based queue processing to allow for shutdown event checking
        - All processing threads continue running in the background
        - The method blocks until shutdown is requested or keyboard interrupt occurs
        - Cleanup is handled automatically through the finally block
        """
        self.input_stream.start()
        logger.success("Audio Modules Operational")
        logger.success("Listening...")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Use timeout to prevent blocking and allow shutdown checking
                    sample, vad_confidence = self._sample_queue.get(timeout=self.PAUSE_TIME)
                    self._handle_audio_sample(sample, vad_confidence)
                except queue.Empty:
                    # Timeout occurred, loop again to check shutdown_event
                    continue
                except Exception as e:
                    if not self.shutdown_event.is_set():
                        logger.error(f"Error in listen loop: {e}")
                    continue

            logger.info("Shutdown event detected in listen loop, exiting loop.")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt in listen loop.")
        finally:
            logger.info("Listen event loop is stopping/exiting.")
            self.stop_listen_event_loop()

    def stop_listen_event_loop(self) -> None:
        """Stop the voice assistant's listening event loop and clean up resources.

        This method performs a graceful shutdown of the audio processing system by
        stopping the audio input stream and setting the shutdown event to terminate
        any ongoing processing threads. It ensures that all resources are released
        properly and all background threads receive the shutdown signal.

        Side Effects:
        - Sets the shutdown event to signal all threads to terminate
        - Stops all sounddevice audio activity globally
        - Triggers cleanup in all background processing threads
        - Releases audio stream resources

        Notes:
        - Called automatically by start_listen_event_loop() during cleanup
        - Can be called manually to programmatically stop the assistant
        - Uses global sd.stop() to ensure all audio activity ceases
        - Threads are daemon threads so they will terminate with the main process

        Raises:
        - No explicit exceptions raised; handles cleanup gracefully
        """
        logger.info("Setting Shutdown event")
        self.shutdown_event.set()

        logger.info("Calling global sd.stop() to halt all sounddevice activity.")
        sd.stop()

        logger.info("Glados engine stop sequence initiated. Threads should terminate.")

    def _handle_audio_sample(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """Handle processing of individual audio samples based on recording state.

        This method serves as the central dispatcher for audio sample processing,
        routing samples to the appropriate handler based on whether voice recording
        has been activated. It maintains the state machine for voice detection and
        recording management.

        Parameters:
        - sample (NDArray[np.float32]): Single audio sample from the input stream,
          typically containing VAD_SIZE milliseconds of audio data
        - vad_confidence (bool): Voice activity detection result indicating whether
          speech is detected in this sample

        Side Effects:
        - Routes samples to pre-activation or activated audio processing
        - Maintains recording state through the voice detection pipeline
        - Triggers state transitions based on voice activity

        Notes:
        - Acts as a state machine dispatcher for audio processing
        - Pre-activation handles buffering before voice detection
        - Activated processing handles speech capture and recognition
        - State transitions are managed by the individual processing methods
        """
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """Manage pre-activation audio buffer and handle voice activity detection.

        This method maintains a circular buffer of audio samples before voice activation,
        ensuring that audio preceding voice detection is preserved for complete utterance
        capture. When voice activity is detected, it transitions to recording mode and
        prepares the accumulated buffer for speech recognition.

        Parameters:
        - sample (NDArray[np.float32]): Current audio sample to buffer
        - vad_confidence (bool): Whether voice activity is detected in this sample

        Side Effects:
        - Maintains circular buffer by discarding oldest samples when full
        - Stops audio stream when voice activity is detected
        - Transitions from buffering to recording mode
        - Copies buffer contents to samples list for processing
        - Respects interruptible setting when assistant is speaking

        Notes:
        - Buffer size is configured by BUFFER_SIZE constant
        - Voice detection triggers immediate transition to recording mode
        - Interruption handling prevents new input during non-interruptible speech
        - Audio stream is stopped to prevent overlap during processing
        """
        # Maintain circular buffer of pre-activation audio
        if self._buffer.full():
            self._buffer.get()  # Discard oldest sample
        self._buffer.put(sample)

        if vad_confidence:  # Voice activity detected
            # Check if interruption is allowed
            if not self.interruptible and self.currently_speaking.is_set():
                logger.info("Interruption is disabled, and the assistant is currently speaking, ignoring new input.")
                return

            # Transition to recording mode
            sd.stop()  # Stop audio stream to prevent overlap
            self.processing = False  # Signal processing threads to halt
            self._samples = list(self._buffer.queue)  # Copy buffer to samples
            self._recording_started = True

    def _process_activated_audio(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """Process audio samples during active speech recording.

        This method accumulates audio samples during active speech recording and
        monitors voice activity detection to determine when a complete utterance
        has been captured. It tracks silence gaps to trigger speech processing
        when the speaker has finished talking.

        Parameters:
        - sample (NDArray[np.float32]): Current audio sample from the input stream
        - vad_confidence (bool): Whether voice activity is currently detected

        Side Effects:
        - Appends audio samples to the samples list for later processing
        - Increments gap counter during silence periods
        - Resets gap counter when voice activity resumes
        - Triggers speech processing when pause limit is reached

        Notes:
        - Pause detection uses PAUSE_LIMIT and VAD_SIZE constants
        - Gap counter tracks consecutive silent samples
        - Processing is triggered after sufficient silence indicating end of utterance
        - Samples are accumulated for complete utterance recognition
        """
        self._samples.append(sample)

        if not vad_confidence:
            # Track silence duration
            self._gap_counter += 1
            if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                self._process_detected_audio()
        else:
            # Reset silence counter on voice activity
            self._gap_counter = 0

    def _process_detected_audio(self) -> None:
        """Process complete audio utterance after pause detection.

        This method handles the complete audio processing pipeline after a pause
        has been detected, including speech recognition, wake word validation,
        and queuing for language model processing. It manages the transition
        back to listening mode after processing.

        Side Effects:
        - Performs automatic speech recognition on accumulated samples
        - Validates wake word if configured
        - Queues recognized text for LLM processing
        - Sets processing flags for background threads
        - Resets audio recording state for next utterance

        Notes:
        - Only processes non-empty transcription results
        - Wake word validation uses Levenshtein distance matching
        - Processing flag signals background threads to begin LLM processing
        - Speaking event is set to indicate assistant response will begin
        """
        logger.debug("Detected pause after speech. Processing...")

        # Perform speech recognition
        detected_text = self.asr(self._samples)

        if detected_text:
            logger.success(f"ASR text: '{detected_text}'")

            # Validate wake word if configured
            if self.wake_word and not self._wakeword_detected(detected_text):
                logger.info(f"Required wake word {self.wake_word=} not detected.")
            else:
                # Queue for LLM processing
                self.llm_queue.put(detected_text)
                self.processing = True
                self.currently_speaking.set()

        # Reset for next utterance
        self.reset()

    def asr(self, samples: list[NDArray[np.float32]]) -> str:
        """Perform automatic speech recognition on accumulated audio samples.

        This method converts a list of audio sample arrays into a single continuous
        audio stream, applies normalization to prevent clipping, and performs
        speech recognition using the configured ASR model.

        Parameters:
        - samples (list[NDArray[np.float32]]): List of audio sample arrays to transcribe

        Returns:
        - str: Transcribed text from the input audio samples

        Notes:
        - Concatenates all samples into a single continuous audio array
        - Normalizes audio to [-0.5, 0.5] range to prevent clipping
        - Uses the pre-configured ASR model for transcription
        - Normalization ensures consistent audio levels for recognition
        """
        # Concatenate all audio samples
        audio = np.concatenate(samples)

        # Normalize audio to prevent clipping and ensure consistent levels
        audio = audio / np.max(np.abs(audio)) / 2

        # Perform speech recognition
        detected_text = self._asr_model.transcribe(audio)
        return detected_text

    def _wakeword_detected(self, text: str) -> bool:
        """Check for wake word similarity using Levenshtein distance matching.

        This method validates whether the transcribed text contains a word that
        closely matches the configured wake word, accounting for potential
        speech recognition errors and variations in pronunciation.

        Parameters:
        - text (str): Transcribed text to check for wake word presence

        Returns:
        - bool: True if a word in the text matches the wake word within the
          similarity threshold, False otherwise

        Raises:
        - AssertionError: If wake word is not configured (None)

        Notes:
        - Uses Levenshtein distance to measure text similarity
        - Compares each word in the text against the wake word
        - Similarity threshold is configured by SIMILARITY_THRESHOLD constant
        - Case-insensitive comparison for robust matching
        """
        assert self.wake_word is not None, "Wake word should not be None"

        words = text.split()
        closest_distance = min([distance(word.lower(), self.wake_word) for word in words])
        return bool(closest_distance < self.SIMILARITY_THRESHOLD)

    def reset(self) -> None:
        """Reset voice recording state and clear all audio buffers.

        This method performs a complete reset of the audio recording system,
        clearing all accumulated samples and returning to the pre-activation
        listening state. It uses thread-safe operations to prevent race
        conditions in the multi-threaded audio processing environment.

        Side Effects:
        - Sets recording_started flag to False
        - Clears the accumulated samples list
        - Resets the gap counter for pause detection
        - Safely empties the circular buffer queue

        Notes:
        - Called after each complete utterance processing
        - Uses mutex lock for thread-safe buffer clearing
        - Prepares the system for the next voice input cycle
        - Essential for proper state management between utterances
        """
        logger.debug("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        
        # Thread-safe buffer clearing
        with self._buffer.mutex:
            self._buffer.queue.clear()

    def play_announcement(self, interruptible: bool | None = None) -> None:
        """
        Play the configured announcement using text-to-speech synthesis.

        This method converts the announcement text to speech and plays it through 
        the default audio output device. The playback behavior can be configured 
        to be interruptible or blocking.

        Parameters:
            interruptible (bool | None, optional): Whether the announcement can be 
                interrupted. If None, uses the instance's interruptible setting.
                If True, allows other audio to interrupt. If False, blocks until 
                announcement completes. Defaults to None.

        Returns:
            None

        Side Effects:
            - Generates speech audio from announcement text
            - Plays audio through default sound device
            - Logs announcement status and content
            - May block execution if interruptible is False

        Notes:
            - Only plays if announcement text is configured (not None)
            - Uses the configured TTS model for speech synthesis
            - Blocking behavior useful for ensuring important announcements complete
            - Non-blocking behavior allows for responsive interaction

        Example:
            >>> glados.play_announcement(interruptible=False)  # Blocks until complete
            >>> glados.play_announcement(interruptible=True)   # Can be interrupted
        """
        if interruptible is None:
            interruptible = self.interruptible
            
        logger.success("Playing announcement...")
        
        if self.announcement:
            # Generate speech audio from announcement text
            audio = self._tts.generate_speech_audio(self.announcement)
            logger.success(f"TTS text: {self.announcement}")
            
            # Play audio through default sound device
            sd.play(audio, self._tts.sample_rate)
            
            # Block until completion if not interruptible
            if not interruptible:
                sd.wait()

    def percentage_played(self, total_samples: int) -> tuple[bool, int]:
        """Monitor audio playback progress with real-time interrupt detection.

        This method tracks audio playback progress through a callback-based system,
        allowing for real-time interruption detection and progress monitoring.
        It uses sounddevice's callback system to monitor sample-level playback
        progress and detect processing or shutdown interruptions.

        Parameters:
        - total_samples (int): Total number of audio samples to be played.
          For example, 1 second of 48kHz audio would be 48,000 samples.

        Returns:
        - tuple[bool, int]: A tuple containing:
          - bool: True if playback was interrupted, False if completed normally
          - int: Percentage of samples played (0-100)

        Side Effects:
        - Creates and manages an audio output stream
        - Monitors processing and shutdown flags for interruption
        - Uses completion event for synchronization

        Notes:
        - Uses callback system for real-time sample tracking
        - Handles interruption via processing flag and shutdown event
        - Implements timeout based on audio duration plus buffer
        - Progress percentage is capped at 100 even if more samples processed
        - Gracefully handles audio stream errors

        Example:
        >>> interrupted, progress = self.percentage_played(48000)
        >>> print(f"Interrupted: {interrupted}, Progress: {progress}%")
        """
        interrupted = False
        progress = 0
        completion_event = threading.Event()

        def stream_callback(
            outdata: NDArray[np.float32], 
            frames: int, 
            time: dict[str, Any], 
            status: sd.CallbackFlags
        ) -> tuple[NDArray[np.float32], sd.CallbackStop | None]:
            nonlocal progress, interrupted
            
            progress += frames
            
            # Check for interruption conditions
            if self.processing is False or self.shutdown_event.is_set():
                interrupted = True
                completion_event.set()
                return outdata, sd.CallbackStop
                
            # Check for completion
            if progress >= total_samples:
                completion_event.set()
                
            return outdata, None

        try:
            stream = sd.OutputStream(
                callback=stream_callback,
                samplerate=self._tts.sample_rate,
                channels=1,
                finished_callback=completion_event.set,
            )
            
            with stream:
                # Wait with timeout to allow for interruption
                timeout = total_samples / self._tts.sample_rate + 1
                completion_event.wait(timeout=timeout)

        except (sd.PortAudioError, RuntimeError):
            logger.debug("Audio stream already closed or invalid")

        # Calculate and cap percentage
        percentage_played = min(int(progress / total_samples * 100), 100)
        return interrupted, percentage_played

    def process_audio_thread(self) -> None:
        """Process audio messages from TTS queue and manage playback with interruption handling.

        This method runs as a background thread, continuously processing audio messages
        from the audio queue. It handles audio playback, interruption detection,
        conversation history management, and end-of-stream processing.

        Side Effects:
        - Plays audio through the default sound device
        - Manages conversation history by accumulating assistant messages
        - Handles interruption detection and message clipping
        - Clears speaking event when responses complete
        - Clears remaining audio queue on interruption

        Notes:
        - Runs until shutdown event is set
        - Accumulates assistant text for complete conversation messages
        - Handles end-of-stream signals to finalize conversation history
        - Interruption handling preserves partial message content
        - Uses mutex lock for thread-safe queue clearing on interruption

        Attributes Used:
        - assistant_text: Accumulates text for current assistant response
        - system_text: Stores text when response is interrupted
        """
        assistant_text: list[str] = []
        system_text: list[str] = []

        while not self.shutdown_event.is_set():
            try:
                audio_msg = self.audio_queue.get(timeout=self.PAUSE_TIME)

                # Handle end-of-stream signal
                if audio_msg.is_eos:
                    logger.debug("Processing end of stream")
                    
                    # Append complete assistant message to conversation
                    if assistant_text:
                        logger.debug(f"Appending assistant message: {' '.join(assistant_text)}")
                        self.messages.append({"role": "assistant", "content": " ".join(assistant_text)})
                    
                    assistant_text = []
                    self.currently_speaking.clear()
                    logger.debug("Speaking event cleared")
                    continue

                # Process audio message
                if len(audio_msg.audio):
                    sd.play(audio_msg.audio, self._tts.sample_rate)
                    total_samples = len(audio_msg.audio)

                    logger.success(f"TTS text: {audio_msg.text}")

                    # Monitor playback for interruption
                    interrupted, percentage_played = self.percentage_played(total_samples)

                    if interrupted:
                        # Handle interrupted playback
                        clipped_text = self.clip_interrupted_sentence(audio_msg.text, percentage_played)
                        logger.success(f"TTS interrupted at {percentage_played}%: {clipped_text}")

                        # Prepare interrupted message
                        system_text = copy.deepcopy(assistant_text)
                        system_text.append(clipped_text)

                        # Add interrupted message to conversation
                        self.messages.append({"role": "assistant", "content": " ".join(system_text)})
                        assistant_text = []

                        self.currently_speaking.clear()
                        logger.debug("Speaking event cleared")

                        # Clear remaining audio queue
                        with self.audio_queue.mutex:
                            self.audio_queue.queue.clear()
                    else:
                        # Accumulate text for complete response
                        assistant_text.append(audio_msg.text)

            except queue.Empty:
                pass

    def clip_interrupted_sentence(self, generated_text: str, percentage_played: float) -> str:
        """Clip text proportionally to audio playback progress with interruption marker.

        This method truncates the generated text based on the percentage of audio
        that was played before interruption, providing an accurate representation
        of what the user actually heard. It appends an interruption marker when
        text is cut short.

        Parameters:
        - generated_text (str): Complete text that was generated by the language model
        - percentage_played (float): Percentage of audio played before interruption (0-100)

        Returns:
        - str: Truncated text with optional interruption marker

        Notes:
        - Calculates word count proportional to audio percentage
        - Appends "<INTERRUPTED>" marker when text is truncated
        - Handles edge cases where no audio was played
        - Preserves word boundaries for clean text representation

        Example:
        >>> text = "Hello world how are you today"
        >>> clipped = self.clip_interrupted_sentence(text, 50)
        >>> print(clipped)
        "Hello world<INTERRUPTED>"
        """
        tokens = generated_text.split()
        words_to_print = round((percentage_played / 100) * len(tokens))
        text = " ".join(tokens[:words_to_print])

        # Add interruption marker if text was cut short
        if words_to_print < len(tokens):
            text = text + "<INTERRUPTED>"
            
        return text