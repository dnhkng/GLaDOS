import copy
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any

from Levenshtein import distance
from loguru import logger
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, HttpUrl
import requests
import sounddevice as sd  # type: ignore
from sounddevice import CallbackFlags
import yaml

from .ASR import VAD, AudioTranscriber
from .TTS import tts_glados, tts_kokoro
from .utils import spoken_text_converter as stc

try:
    logger.remove(0)
except ValueError:
    pass  # Handler with ID 0 does not exist


logger.add(sys.stderr, level="INFO")


class PersonalityPrompt(BaseModel):
    system: str | None = None
    user: str | None = None
    assistant: str | None = None

    def to_chat_message(self) -> dict[str, str]:
        """Convert the prompt to a chat message format.

        Returns:
            dict[str, str]: A single chat message dictionary

        Raises:
            ValueError: If the prompt does not contain exactly one non-null field
        """
        for field, value in self.model_dump(exclude_none=True).items():
            return {"role": field, "content": value}
        raise ValueError("PersonalityPrompt must have exactly one non-null field")


class GladosConfig(BaseModel):
    completion_url: HttpUrl
    model: str
    api_key: str | None = None
    interruptible: bool = True
    wake_word: str | None = None
    voice: str
    announcement: str | None = None
    personality_preprompt: list[PersonalityPrompt]

    @classmethod
    def from_yaml(
        cls, path: str | Path, key_to_config: tuple[str, ...] = ("Glados",)
    ) -> "GladosConfig":
        """
        Load a GladosConfig instance from a YAML configuration file.

        Parameters:
            path: Path to the YAML configuration file
            key_to_config: Tuple of keys to navigate nested configuration

        Returns:
            GladosConfig: Configuration object with validated settings

        Raises:
            ValueError: If the YAML content is invalid
            OSError: If the file cannot be read
            pydantic.ValidationError: If the configuration is invalid
        """
        path = Path(path)

        # Try different encodings
        for encoding in ["utf-8", "utf-8-sig"]:
            try:
                data = yaml.safe_load(path.read_text(encoding=encoding))
                break
            except UnicodeDecodeError:
                if encoding == "utf-8-sig":
                    raise

        # Navigate through nested keys
        config = data
        for key in key_to_config:
            config = config[key]

        return cls(**cls.model_validate(config).dict())

    def to_chat_messages(self) -> list[dict[str, str]]:
        """Convert personality preprompt to chat message format."""
        return [prompt.to_chat_message() for prompt in self.personality_preprompt]


@dataclass
class AudioMessage:
    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False


class Glados:
    PAUSE_TIME: float = 0.05  # Time to wait between processing loops
    SAMPLE_RATE: int = 16000  # Sample rate for input stream
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection
    BUFFER_SIZE: int = 800  # Milliseconds of buffer BEFORE VAD detection
    PAUSE_LIMIT: int = 640  # Milliseconds of pause allowed before processing
    SIMILARITY_THRESHOLD: int = 2  # Threshold for wake word similarity

    NEUROTOXIN_RELEASE_ALLOWED: bool = (
        False  # preparation for function calling, see issue #13
    )
    DEFAULT_PERSONALITY_PREPROMPT: tuple[dict[str, str], ...] = (
        {
            "role": "system",
            "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
        },
    )

    def __init__(
        self,
        asr_model: AudioTranscriber,
        tts_model: tts_glados.Synthesizer | tts_kokoro.Synthesizer,
        vad_model: VAD,
        completion_url: str,
        model: str,
        api_key: str | None = None,
        interruptible: bool = True,
        wake_word: str | None = None,
        personality_preprompt: tuple[
            dict[str, str], ...
        ] = DEFAULT_PERSONALITY_PREPROMPT,
        announcement: str | None = None,
    ) -> None:
        """
        Initialize the Glados voice assistant with configuration parameters.

        This method sets up the voice recognition system, including voice activity detection (VAD),
        automatic speech recognition (ASR), text-to-speech (TTS), and language model processing.
        The initialization configures various components and starts background threads for
        processing LLM responses and TTS output.

        Args:
            voice_model (str): Path to the voice model for text-to-speech synthesis.
            speaker_id (int | None): Identifier for the specific speaker voice, if applicable.
            completion_url (str): URL endpoint for language model completions.
            model (str): Identifier for the language model being used.
            api_key (str | None, optional): Authentication key for the language model API. Defaults to None.
            wake_word (str | None, optional): Activation word to trigger voice assistant. Defaults to None.
            personality_preprompt (list[dict[str, str]], optional): Initial context or personality
                configuration for the language model. Defaults to DEFAULT_PERSONALITY_PREPROMPT.
            announcement (str | None, optional): Initial announcement to be spoken upon initialization.
                Defaults to None.
            interruptible (bool, optional): Whether the assistant's speech can be interrupted.
                Defaults to True.
        """
        self.completion_url = completion_url
        self.model = model
        self.wake_word = wake_word
        self._vad_model = vad_model
        self._tts = tts_model
        self._asr_model = asr_model
        self._stc = stc.SpokenTextConverter()

        # warm up onnx ASR model
        self._asr_model.transcribe_file("data/0.wav")

        # LLAMA_SERVER_HEADERS
        self.prompt_headers = {
            "Authorization": (
                f"Bearer {api_key}" if api_key else "Bearer your_api_key_here"
            ),
            "Content-Type": "application/json",
        }

        # Initialize sample queues and state flags
        self._samples: list[NDArray[np.float32]] = []
        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = (
            queue.Queue()
        )
        self._buffer: queue.Queue[NDArray[np.float32]] = queue.Queue(
            maxsize=self.BUFFER_SIZE // self.VAD_SIZE
        )
        self._recording_started = False
        self._gap_counter = 0

        self._messages: list[dict[str, str]] = list(personality_preprompt)

        self.llm_queue: queue.Queue[str] = queue.Queue()
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.audio_queue: queue.Queue[AudioMessage] = queue.Queue()

        self.processing = False
        self.interruptible = interruptible

        self.currently_speaking = threading.Event()
        self.shutdown_event = threading.Event()

        llm_thread = threading.Thread(target=self.process_llm)
        llm_thread.start()

        tts_thread = threading.Thread(target=self.process_tts_thread)
        tts_thread.start()

        audio_thread = threading.Thread(target=self.process_audio_thread)
        audio_thread.start()

        if announcement:
            audio = self._tts.generate_speech_audio(announcement)
            logger.success(f"TTS text: {announcement}")
            sd.play(audio, self._tts.sample_rate)
            if not self.interruptible:
                sd.wait()

        def audio_callback_for_sd_input_stream(
            indata: np.dtype[np.float32],
            frames: int,
            time: sd.CallbackStop,
            status: CallbackFlags,
        ) -> None:
            """
            Callback function for processing audio input from a sounddevice input stream.

            This method is responsible for handling incoming audio samples, performing voice activity detection (VAD),
            and queuing the processed audio data for further analysis.

            Parameters:
                indata (np.ndarray): Input audio data from the sounddevice stream
                frames (int): Number of audio frames in the current chunk
                time (sd.CallbackStop): Timing information for the audio callback
                status (CallbackFlags): Status flags for the audio callback

            Returns:
                None

            Notes:
                - Copies and squeezes the input data to ensure single-channel processing
                - Applies voice activity detection to determine speech presence
                - Puts processed audio samples and VAD confidence into a thread-safe queue
            """

            data = (
                np.array(indata).copy().squeeze()
            )  # Reduce to single channel if necessary
            vad_value = self._vad_model(np.expand_dims(data, 0))
            vad_confidence = vad_value > self.VAD_THRESHOLD
            self._sample_queue.put((data, bool(vad_confidence)))

        self.input_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            callback=audio_callback_for_sd_input_stream,
            blocksize=int(self.SAMPLE_RATE * self.VAD_SIZE / 1000),
        )

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        Retrieve the current list of conversation messages.

        Returns:
            list[dict[str, str]]: A list of message dictionaries representing the conversation history.
        """
        return self._messages

    @classmethod
    def from_config(cls, config: GladosConfig) -> "Glados":
        """
        Create a Glados instance from a GladosConfig configuration object.

        Parameters:
            config (GladosConfig): Configuration object containing Glados initialization parameters

        Returns:
            Glados: A new Glados instance configured with the provided settings
        """
        asr_model = AudioTranscriber()
        vad_model = VAD()

        tts_model: tts_glados.Synthesizer | tts_kokoro.Synthesizer
        if config.voice == "glados":
            tts_model = tts_glados.Synthesizer()
        else:
            assert (
                config.voice in tts_kokoro.get_voices()
            ), f"Voice '{config.wake_word}' not available"
            tts_model = tts_kokoro.Synthesizer(voice=config.voice)

        return cls(
            asr_model=asr_model,
            tts_model=tts_model,
            vad_model=vad_model,
            completion_url=config.completion_url,
            model=config.model,
            api_key=config.api_key,
            interruptible=config.interruptible,
            wake_word=config.wake_word,
            announcement=config.announcement,
            personality_preprompt=tuple(config.to_chat_messages()),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Glados":
        """
        Create a Glados instance from a configuration file.

        Parameters:
            path (str): Path to the YAML configuration file containing Glados settings.

        Returns:
            Glados: A new Glados instance configured with settings from the specified YAML file.

        Example:
            glados = Glados.from_yaml('config/default.yaml')
        """
        return cls.from_config(GladosConfig.from_yaml(path))

    def start_listen_event_loop(self) -> None:
        """
        Start the voice assistant's listening event loop, continuously processing audio input.

        This method initializes the audio input stream and enters an infinite loop to handle incoming audio samples.
        The loop retrieves audio samples and their voice activity detection (VAD) confidence from a queue and processes
        each sample using the `_handle_audio_sample` method.

        Behavior:
        - Starts the audio input stream
        - Logs successful initialization of audio modules
        - Enters an infinite listening loop
        - Retrieves audio samples from a queue
        - Processes each audio sample with VAD confidence
        - Handles keyboard interrupts by stopping the input stream and setting a shutdown event

        Raises:
            KeyboardInterrupt: Allows graceful termination of the listening loop
        """
        self.input_stream.start()
        logger.success("Audio Modules Operational")
        logger.success("Listening...")
        # Loop forever, but is 'paused' when new samples are not available
        try:
            while True:
                sample, vad_confidence = self._sample_queue.get()
                self._handle_audio_sample(sample, vad_confidence)
        except KeyboardInterrupt:
            self.shutdown_event.set()
            self.input_stream.stop()

    def start_text(self) -> None:
        """
        Start a text-based interaction loop with Glados using terminal input.

        This method allows the user to communicate with Glados via text input from the terminal.
        It reads input from the user, sends it to the LLM queue, and prints the assistant's responses.
        """
        logger.success("Starting text-based interaction with Glados...")
        logger.success(
            "Type your message and press Enter to send. Type 'exit' to quit."
        )

        while not self.shutdown_event.is_set():
            try:
                # Read input from the terminal
                user_input = input("You: ").strip()

                # Exit condition
                if user_input.lower() == "exit":
                    logger.info("Exiting text-based interaction.")
                    break

                # Send the input to the LLM queue
                self.llm_queue.put(user_input)

                # Wait for the assistant's response
                self.processing = True
                self.currently_speaking.set()

                # Wait until the assistant finishes speaking
                while self.currently_speaking.is_set():
                    continue

            except KeyboardInterrupt:
                logger.info("Exiting text-based interaction.")
                break

    def _handle_audio_sample(
        self, sample: NDArray[np.float32], vad_confidence: bool
    ) -> None:
        """
        Handles the processing of each audio sample.

        If the recording has not started, the sample is added to the circular buffer.

        If the recording has started, the sample is added to the samples list, and the pause
        limit is checked to determine when to process the detected audio.

        Args:
            sample (np.ndarray): The audio sample to process.
            vad_confidence (bool): Whether voice activity is detected in the sample.
        """
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(
        self, sample: NDArray[np.float32], vad_confidence: bool
    ) -> None:
        """
        Manages the pre-activation audio buffer and handles voice activity detection.

        This method maintains a circular buffer of audio samples before voice activation,
        discarding the oldest sample when the buffer is full. When voice activity is detected,
        it stops the audio stream and prepares for audio processing.

        Args:
            sample (np.ndarray): The current audio sample to be added to the buffer.
            vad_confidence (bool): Indicates whether voice activity is detected in the sample.

        Side Effects:
            - Modifies the internal circular buffer
            - Stops the audio stream when voice is detected
            - Disables processing on LLM and TTS threads
            - Prepares samples for recording when voice is detected
        """
        if self._buffer.full():
            self._buffer.get()  # Discard the oldest sample to make room for new ones
        self._buffer.put(sample)

        if vad_confidence:  # Voice activity detected
            if not self.interruptible and self.currently_speaking.is_set():
                logger.info(
                    "Interruption is disabled, and the assistant is currently speaking, ignoring new input."
                )
                return

            sd.stop()  # Stop the audio stream to prevent overlap
            self.processing = (
                False  # Turns off processing on threads for the LLM and TTS!!!
            )
            self._samples = list(self._buffer.queue)
            self._recording_started = True

    def _process_activated_audio(
        self, sample: NDArray[np.float32], vad_confidence: bool
    ) -> None:
        """
        Process audio samples, tracking speech pauses to capture complete utterances.

        This method accumulates audio samples and monitors voice activity detection (VAD) confidence to determine
        when a complete speech segment has been captured. It appends incoming samples to the internal buffer and
        tracks silent gaps to trigger audio processing.

        Parameters:
            sample (np.ndarray): A single audio sample from the input stream
            vad_confidence (bool): Indicates whether voice activity is currently detected

        Side Effects:
            - Appends audio samples to self._samples
            - Increments or resets self._gap_counter
            - Triggers audio processing via self._process_detected_audio() when pause limit is reached
        """

        self._samples.append(sample)

        if not vad_confidence:
            self._gap_counter += 1
            if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                self._process_detected_audio()
        else:
            self._gap_counter = 0

    def _wakeword_detected(self, text: str) -> bool:
        """
        Check if the detected text contains a close match to the wake word using Levenshtein distance.

        This method helps handle variations in wake word detection by calculating the minimum edit distance
        between detected words and the configured wake word. It accounts for potential misheard
        variations during speech recognition.

        Parameters:
            text (str): The transcribed text to check for wake word similarity

        Returns:
            bool: True if a word in the text is sufficiently similar to the wake word, False otherwise

        Raises:
            AssertionError: If the wake word is not configured (None)

        Notes:
            - Uses Levenshtein distance to measure text similarity
            - Compares each word in the text against the wake word
            - Considers a match if the distance is below a predefined similarity threshold
        """
        assert self.wake_word is not None, "Wake word should not be None"

        words = text.split()
        closest_distance = min(
            [distance(word.lower(), self.wake_word) for word in words]
        )
        return bool(closest_distance < self.SIMILARITY_THRESHOLD)

    def reset(self) -> None:
        """
        Reset the voice recording state and clear all audio buffers.

        This method performs the following actions:
        - Logs a debug message indicating the reset process
        - Stops the current recording by setting `_recording_started` to False
        - Clears the collected audio samples
        - Resets the gap counter used for detecting speech pauses
        - Empties the thread-safe audio buffer queue

        Note:
            Uses a mutex lock to safely clear the shared buffer queue to prevent
            potential race conditions in multi-threaded audio processing.
        """
        logger.debug("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        with self._buffer.mutex:
            self._buffer.queue.clear()

    def _process_detected_audio(self) -> None:
        """
        Process detected audio and generate a response after speech pause.

        Transcribes audio samples and handles wake word detection and LLM processing. Manages the
        audio input stream and processing state throughout the interaction.

        Args:
            None

        Returns:
            None

        Side Effects:
            - Stops the input audio stream
            - Performs automatic speech recognition (ASR)
            - Potentially sends text to LLM queue
            - Resets audio recording state
            - Restarts input audio stream

        Raises:
            No explicit exceptions raised
        """
        logger.debug("Detected pause after speech. Processing...")

        detected_text = self.asr(self._samples)

        if detected_text:
            logger.success(f"ASR text: '{detected_text}'")

            if self.wake_word and not self._wakeword_detected(detected_text):
                logger.info(f"Required wake word {self.wake_word=} not detected.")
            else:
                self.llm_queue.put(detected_text)
                self.processing = True
                self.currently_speaking.set()

        self.reset()

    def asr(self, samples: list[NDArray[np.float32]]) -> str:
        """
        Perform automatic speech recognition (ASR) on the provided audio samples.

        Parameters:
            samples (list[np.dtype[np.float32]]): A list of numpy arrays containing audio samples to be transcribed.

        Returns:
            str: The transcribed text from the input audio samples.

        Notes:
            - Concatenates multiple audio samples into a single continuous audio array
            - Uses the pre-configured ASR model to transcribe the audio
        """
        audio = np.concatenate(samples)

        # Normalize audio to [-0.5, 0.5] range to prevent clipping and ensure consistent levels
        audio = audio / np.max(np.abs(audio)) / 2

        detected_text = self._asr_model.transcribe(audio)
        return detected_text

    def percentage_played(self, total_samples: int) -> tuple[bool, int]:
        """
        Monitor audio playback progress and return completion status with interrupt detection.

        Streams audio samples through PortAudio and actively tracks the number of samples
        that have been played. The playback can be interrupted by setting self.processing
        to False or self.shutdown_event. Uses a non-blocking callback system with a completion event for
        synchronization.

        Args:
            total_samples: Number of audio samples to be played in total. For example,
                for 1 second of 48kHz audio, this would be 48000.

        Returns:
            A tuple containing:
            - bool: True if playback was interrupted, False if completed normally
            - int: Percentage of samples played (0-100), calculated as
            (played_samples / total_samples * 100)

        Raises:
            sd.PortAudioError: If the audio stream encounters initialization or
                playback errors
            RuntimeError: If stream management fails during execution

        Examples:
            For 1 second of audio at 48kHz:
            >>> interrupted, progress = audio.percentage_played(48000)
            >>> print(f"Interrupted: {interrupted}, Progress: {progress}%")
            Interrupted: False, Progress: 100%

        Implementation Details:
            - Uses a stream callback system to track sample count in real-time
            - Handles interruption via self.processing flag
            - Implements timeout based on audio duration plus 1 second buffer
            - Caps progress percentage at 100 even if more samples are processed
        """
        interrupted = False
        progress = 0
        completion_event = threading.Event()

        def stream_callback(
            outdata: NDArray[np.float32],
            frames: int,
            time: dict[str, Any],
            status: sd.CallbackFlags,
        ) -> tuple[NDArray[np.float32], sd.CallbackStop | None]:
            nonlocal progress, interrupted
            progress += frames
            if self.processing is False or self.shutdown_event.is_set():
                interrupted = True
                completion_event.set()
                return outdata, sd.CallbackStop
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
                completion_event.wait(timeout=total_samples / self._tts.sample_rate + 1)

        except (sd.PortAudioError, RuntimeError):
            logger.debug("Audio stream already closed or invalid")

        percentage_played = min(int(progress / total_samples * 100), 100)
        return interrupted, percentage_played

    def process_llm(self) -> None:
        """
        Process text through the Language Model (LLM) and generate conversational responses.

        This method runs in a continuous loop, retrieving detected text from a queue and sending it to an LLM server.
        It streams the response, processes each chunk, and sends processed sentences to the text-to-speech (TTS) queue.

        Key Behaviors:
        - Continuously polls the LLM queue for detected text
        - Sends text to LLM server with streaming enabled
        - Processes response chunks in real-time
        - Breaks sentences at punctuation marks
        - Handles interruptions and processing flags
        - Adds end-of-stream token to TTS queue after processing

        Exceptions:
        - Handles empty queue timeouts
        - Catches and logs errors during line processing
        - Stops processing if shutdown event is set or processing flag is False

        Side Effects:
        - Modifies `self.messages` by appending user messages
        - Puts processed sentences into `self.tts_queue`
        - Logs debug and error information

        Note:
        - Uses a timeout mechanism to prevent blocking
        - Supports graceful interruption of LLM processing
        """
        while not self.shutdown_event.is_set():
            try:
                detected_text = self.llm_queue.get(timeout=0.1)
                self.messages.append({"role": "user", "content": detected_text})

                data = {
                    "model": self.model,
                    "stream": True,
                    "messages": self.messages,
                }
                logger.debug(f"starting request on {self.messages=}")
                logger.debug("Performing request to LLM server...")

                # Perform the request and process the stream

                with requests.post(
                    self.completion_url,
                    headers=self.prompt_headers,
                    json=data,
                    stream=True,
                ) as response:
                    sentence = []
                    for line in response.iter_lines():
                        if self.processing is False:
                            break  # If the stop flag is set from new voice input, halt processing
                        if line:  # Filter out empty keep-alive new lines
                            try:
                                cleaned_line = self._clean_raw_bytes(line)
                                if cleaned_line:  # Add check for empty cleaned line
                                    chunk = self._process_chunk(cleaned_line)
                                    if chunk:
                                        sentence.append(chunk)
                                        # If there is a pause token, send the sentence to the TTS queue
                                        if (
                                            chunk
                                            in [
                                                ".",
                                                "!",
                                                "?",
                                                ":",
                                                ";",
                                                "?!",
                                                "\n",
                                                "\n\n",
                                            ]
                                            and sentence[-2].isdigit() is False
                                        ):  # Don't split on numbers!
                                            logger.debug(f"Chunk: {chunk}")
                                            self._process_sentence(sentence)
                                            sentence = []
                            except Exception as e:
                                logger.error(f"Error processing line: {e}")
                                continue

                    if self.processing and sentence:
                        self._process_sentence(sentence)
                    self.tts_queue.put("<EOS>")  # Add end of stream token to the queue
            except queue.Empty:
                time.sleep(self.PAUSE_TIME)

    def _process_sentence(self, current_sentence: list[str]) -> None:
        """
        Process a sentence for text-to-speech by cleaning and formatting the input text.

        This method handles text preprocessing for the TTS system, removing special formatting
        and cleaning up the text before adding it to the TTS queue.

        Args:
            current_sentence (list[str]): A list of text fragments to be processed.

        Notes:
            - Removes text enclosed in asterisks (*) and parentheses ()
            - Replaces newlines with periods
            - Removes extra whitespace and colons
            - Only adds non-empty sentences to the TTS queue
        """
        sentence = "".join(current_sentence)
        sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
        sentence = (
            sentence.replace("\n\n", ". ")
            .replace("\n", ". ")
            .replace("  ", " ")
            .replace(":", " ")
        )
        if sentence:
            self.tts_queue.put(sentence)

    def _clean_raw_bytes(self, line: bytes) -> dict[str, Any] | None:
        """
        Cleans the raw bytes from the server and converts to OpenAI format.

        Args:
            line (bytes): The raw bytes from the server

        Returns:
            dict or None: Parsed JSON response in OpenAI format, or None if parsing fails
        """
        try:
            # Handle OpenAI format
            if line.startswith(b"data: "):
                json_str = line.decode("utf-8")[6:]  # Remove 'data: ' prefix
                parsed_json: dict[str, Any] = json.loads(json_str)
                return parsed_json
            # Handle Ollama format
            else:
                parsed_json = json.loads(line.decode("utf-8"))
                if isinstance(parsed_json, dict):
                    return parsed_json
                return None
        except Exception as e:
            logger.warning(f"Failed to parse server response: {e}")
            return None

    def _process_chunk(self, line: dict[str, Any]) -> str | None:
        """
        Process a single chunk of text from the LLM server, extracting content from different response formats.

        This method handles text chunks from two different LLM server response formats: OpenAI and Ollama.
        It safely extracts the text content, handling potential missing or malformed data.

        Args:
            line (dict[str, Any]): A dictionary containing the LLM server response chunk.

        Returns:
            str | None: The extracted text content, or None if no content is found or an error occurs.

        Raises:
            No explicit exceptions are raised; errors are logged and result in returning None.
        """
        if not line or not isinstance(line, dict):
            return None

        try:
            # Handle OpenAI format
            if "choices" in line:
                content = line.get("choices", [{}])[0].get("delta", {}).get("content")
                return content if content else None
            # Handle Ollama format
            else:
                content = line.get("message", {}).get("content")
                return content if content else None
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return None

    def process_tts_thread(self) -> None:
        """
        Processes text-to-speech (TTS) generation and playback in a dedicated thread.

        This method continuously retrieves generated text from the TTS queue and converts it to spoken audio.
        It manages the lifecycle of TTS output, including handling interruptions, tracking playback
        progress, and updating conversation messages.

        The method runs until the shutdown event is triggered and handles several key scenarios:
        - Generating speech audio from text
        - Playing audio through the default sound device
        - Detecting and handling audio interruptions
        - Tracking and logging TTS performance metrics
        - Managing conversation message history

        Attributes:
            assistant_text (list[str]): Accumulates text generated by the assistant for current response
            system_text (list[str]): Stores text logged when TTS is interrupted
            finished (bool): Indicates completion of TTS generation
            interrupted (bool): Signals whether TTS playback was interrupted

        Raises:
            queue.Empty: When no text is available in the TTS queue within the specified timeout
        """
        while not self.shutdown_event.is_set():
            try:
                generated_text = self.tts_queue.get(timeout=self.PAUSE_TIME)

                if generated_text == "<EOS>":
                    self.audio_queue.put(AudioMessage(np.array([]), "", is_eos=True))
                elif not generated_text:
                    logger.warning("Empty string sent to TTS")
                else:
                    logger.debug(f"LLM text: {generated_text}")

                    start = time.time()
                    spoken_text = self._stc.text_to_spoken(generated_text)
                    audio = self._tts.generate_speech_audio(spoken_text)
                    logger.debug(
                        f"TTS Complete, inference: {(time.time() - start):.2f}, "
                        f"length: {len(audio) / self._tts.sample_rate:.2f}s"
                    )

                    if len(audio):
                        self.audio_queue.put(AudioMessage(audio, spoken_text))

            except queue.Empty:
                pass

    def process_audio_thread(self) -> None:
        """Process audio from the TTS queue and play it through the default sound device

        This method continuously retrieves audio messages from the audio queue and plays them through the default sound
        device. It manages the lifecycle of audio output, including handling interruptions, tracking playback progress,
        and updating conversation messages.

        Attributes:
            assistant_text (list[str]): Accumulates text generated by the assistant for current response
            system_text (list[str]): Stores text logged when TTS is interrupted

        Raises:
            queue.Empty: When no audio is available in the audio queue within the specified timeout.
        """
        assistant_text: list[str] = []
        system_text: list[str] = []

        while not self.shutdown_event.is_set():
            try:
                audio_msg = self.audio_queue.get(timeout=self.PAUSE_TIME)

                if audio_msg.is_eos:
                    logger.debug("Processing end of stream")
                    # End of stream - append complete message
                    if assistant_text:
                        logger.debug(
                            f"Appending assistant message: {' '.join(assistant_text)}"
                        )
                        self.messages.append(
                            {"role": "assistant", "content": " ".join(assistant_text)}
                        )
                    assistant_text = []
                    self.currently_speaking.clear()
                    logger.debug("Speaking event cleared")
                    continue

                if len(audio_msg.audio):
                    sd.play(audio_msg.audio, self._tts.sample_rate)
                    total_samples = len(audio_msg.audio)

                    logger.success(f"TTS text: {audio_msg.text}")

                    interrupted, percentage_played = self.percentage_played(
                        total_samples
                    )

                    if interrupted:
                        clipped_text = self.clip_interrupted_sentence(
                            audio_msg.text, percentage_played
                        )
                        logger.success(
                            f"TTS interrupted at {percentage_played}%: {clipped_text}"
                        )

                        system_text = copy.deepcopy(assistant_text)
                        system_text.append(clipped_text)

                        # Add interrupted message
                        self.messages.append(
                            {"role": "assistant", "content": " ".join(system_text)}
                        )
                        assistant_text = []

                        self.currently_speaking.clear()
                        logger.debug("Speaking event cleared")

                        # Clear remaining audio queue
                        with self.audio_queue.mutex:
                            self.audio_queue.queue.clear()
                    else:
                        assistant_text.append(audio_msg.text)

            except queue.Empty:
                pass

    def clip_interrupted_sentence(
        self, generated_text: str, percentage_played: float
    ) -> str:
        """
        Clips the generated text based on the percentage of audio played before interruption.

        Truncates the text proportionally to the percentage of audio played and appends an
        interruption marker if the text was cut short.

        Args:
            generated_text (str): The complete text generated by the language model.
            percentage_played (float): Percentage of audio played before interruption (0-100).

        Returns:
            str: Truncated text with an optional interruption marker.

        Example:
            >>> assistant.clip_interrupted_sentence("Hello world how are you today", 50)
            "Hello world<INTERRUPTED>"
        """
        tokens = generated_text.split()
        words_to_print = round((percentage_played / 100) * len(tokens))
        text = " ".join(tokens[:words_to_print])

        # If the TTS was cut off, make that clear
        if words_to_print < len(tokens):
            text = text + "<INTERRUPTED>"
        return text


def start() -> None:
    """Set up the LLM server and start GlaDOS.

    This function reads the configuration file, initializes the Glados voice assistant,
    and starts the listening event loop.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there is an error parsing the YAML configuration file.
    """
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Start Glados with audio or text input."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["audio", "text"],
        default="audio",
        help="Choose the interaction mode: 'audio' for microphone input, 'text' for terminal input.",
    )
    args = parser.parse_args()

    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    glados = Glados.from_config(glados_config)

    if args.mode == "audio":
        glados.start_listen_event_loop()
    elif args.mode == "text":
        glados.start_text()


if __name__ == "__main__":
    start()
