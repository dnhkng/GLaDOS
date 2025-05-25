"""
Core GLaDOS Engine - Main Glados class initialization and factory methods.

This module contains the primary Glados class with its initialization logic,
factory methods for creating instances from configuration, and core constants.
The core module handles the setup of all voice assistant components including
ASR, TTS, VAD models, and background processing threads.
"""

import sys
import threading
import queue
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import HttpUrl
import sounddevice as sd
from sounddevice import CallbackFlags
from loguru import logger

from glados.ASR import VAD, TranscriberProtocol, get_audio_transcriber
from glados.engine.tui_integration import TUIIntegrationMixin
from ..TTS import tts_glados, tts_kokoro
from ..utils import spoken_text_converter as stc
from ..utils.resources import resource_path
from .audio_processing import AudioProcessingMixin
from .llm_processing import LLMProcessingMixin  
from .tts_processing import TTSProcessingMixin
from .utils import UtilsMixin
from .config import GladosConfig
from .audio_models import AudioMessage


class Glados(AudioProcessingMixin, LLMProcessingMixin, TTSProcessingMixin, TUIIntegrationMixin, UtilsMixin):
    """
    Main GLaDOS voice assistant class handling initialization and core functionality.
    
    This class orchestrates all components of the voice assistant including:
    - Automatic Speech Recognition (ASR)
    - Text-to-Speech (TTS) 
    - Voice Activity Detection (VAD)
    - Language Model (LLM) processing
    - Audio input/output management
    - TUI integration for real-time updates
    
    The class manages multiple background threads for processing different aspects
    of the voice interaction pipeline and maintains state for conversation history
    and processing flags.
    """
    
    # Core timing and processing constants
    PAUSE_TIME: float = 0.05  # Time to wait between processing loops (seconds)
    SAMPLE_RATE: int = 16000  # Sample rate for input stream (Hz)
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection (0.0-1.0)
    
    # Audio buffer configuration
    BUFFER_SIZE: int = 800  # Milliseconds of buffer BEFORE VAD detection
    PAUSE_LIMIT: int = 640  # Milliseconds of pause allowed before processing
    
    # Wake word detection
    SIMILARITY_THRESHOLD: int = 2  # Threshold for wake word similarity (Levenshtein distance)
    
    # Text processing configuration
    PUNCTUATION_SET: tuple[str, ...] = (
        ".", "!", "?", ":", ";", "?!", "\n", "\n\n"
    )  # Sentence splitting punctuation marks
    
    # Feature flags
    NEUROTOXIN_RELEASE_ALLOWED: bool = False  # Preparation for function calling, see issue #13
    
    # Default personality configuration
    DEFAULT_PERSONALITY_PREPROMPT: tuple[dict[str, str], ...] = (
        {
            "role": "system",
            "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
        },
    )

    def __init__(
        self,
        asr_model: TranscriberProtocol,
        tts_model: tts_glados.Synthesizer | tts_kokoro.Synthesizer,
        vad_model: VAD,
        completion_url: HttpUrl,
        model: str,
        api_key: str | None = None,
        interruptible: bool = True,
        wake_word: str | None = None,
        personality_preprompt: tuple[dict[str, str], ...] = DEFAULT_PERSONALITY_PREPROMPT,
        announcement: str | None = None,
    ) -> None:
        """
        Initialize the Glados voice assistant with all required components.

        This method sets up the complete voice recognition system, including voice activity 
        detection (VAD), automatic speech recognition (ASR), text-to-speech (TTS), and 
        language model processing. The initialization configures various components and 
        starts background threads for processing LLM responses and TTS output.

        Parameters:
            asr_model (TranscriberProtocol): Automatic speech recognition model for transcribing audio
            tts_model (tts_glados.Synthesizer | tts_kokoro.Synthesizer): Text-to-speech synthesis model
            vad_model (VAD): Voice activity detection model for identifying speech segments
            completion_url (HttpUrl): URL endpoint for language model completions
            model (str): Identifier for the language model being used
            api_key (str | None, optional): Authentication key for the language model API. 
                Defaults to None.
            interruptible (bool, optional): Whether the assistant's speech can be interrupted 
                by new input. Defaults to True.
            wake_word (str | None, optional): Activation word to trigger voice assistant. 
                If None, always processes detected speech. Defaults to None.
            personality_preprompt (tuple[dict[str, str], ...], optional): Initial context or 
                personality configuration for the language model in chat message format. 
                Defaults to DEFAULT_PERSONALITY_PREPROMPT.
            announcement (str | None, optional): Initial announcement to be spoken upon 
                initialization. Defaults to None.

        Returns:
            None

        Side Effects:
            - Warms up the ASR model with a test audio file
            - Initializes audio input stream with callback
            - Starts three background daemon threads for LLM, TTS, and audio processing
            - Sets up various queues for inter-thread communication
            - Configures HTTP headers for LLM API communication

        Notes:
            - The ASR model is warmed up using a test file to ensure optimal performance
            - All processing threads are started as daemon threads for clean shutdown
            - Audio callback is configured for real-time voice activity detection
            - Thread-safe queues are used for communication between processing components

        Raises:
            FileNotFoundError: If the test audio file for ASR warmup is not found
            RuntimeError: If audio stream initialization fails


                    Initialize the Glados voice assistant with configuration parameters.

        This method sets up the voice recognition system, including voice activity detection (VAD),
        automatic speech recognition (ASR), text-to-speech (TTS), and language model processing.
        The initialization configures various components and starts background threads for
        processing LLM responses and TTS output.

        Args:
            asr_model: Transcriber for automatic speech recognition
            tts_model: Text-to-speech synthesizer 
            vad_model: Voice activity detection model
            completion_url: URL endpoint for language model completions
            model: Identifier for the language model being used
            api_key: Authentication key for the language model API
            interruptible: Whether the assistant's speech can be interrupted
            wake_word: Activation word to trigger voice assistant
            personality_preprompt: Initial context for the language model
            announcement: Initial announcement to be spoken upon initialization
        """        
        # Initialize parent mixins first (TUIIntegrationMixin needs to be called)
        super().__init__()
        
        # Core configuration
        self.completion_url = completion_url
        self.model = model
        self.wake_word = wake_word
        self.announcement = announcement
        self.interruptible = interruptible
        
        # Component models
        self._vad_model = vad_model
        self._tts = tts_model
        self._asr_model = asr_model
        self._stc = stc.SpokenTextConverter()

        # Warm up ONNX ASR model for optimal performance
        self._asr_model.transcribe_file(resource_path("data/0.wav"))

        # Configure HTTP headers for LLM API communication
        self.prompt_headers = {
            "Authorization": (f"Bearer {api_key}" if api_key else "Bearer your_api_key_here"),
            "Content-Type": "application/json",
        }

        # Initialize audio processing state
        self._samples: list[NDArray[np.float32]] = []
        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()
        self._buffer: queue.Queue[NDArray[np.float32]] = queue.Queue(
            maxsize=self.BUFFER_SIZE // self.VAD_SIZE
        )
        self._recording_started = False
        self._gap_counter = 0

        # Initialize conversation state
        self._messages: list[dict[str, str]] = list(personality_preprompt)

        # Initialize inter-thread communication queues
        self.llm_queue: queue.Queue[str] = queue.Queue()
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.audio_queue: queue.Queue[AudioMessage] = queue.Queue()

        # Initialize processing state flags
        self.processing = False
        self.currently_speaking = threading.Event()
        self.shutdown_event = threading.Event()

        # Start background processing threads
        llm_thread = threading.Thread(target=self.process_llm, daemon=True)
        llm_thread.start()

        tts_thread = threading.Thread(target=self.process_tts_thread, daemon=True)
        tts_thread.start()

        audio_thread = threading.Thread(target=self.process_audio_thread, daemon=True)
        audio_thread.start()

        # Set up audio input stream with callback
        audio_callback = self.create_audio_callback()
        self.input_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=int(self.SAMPLE_RATE * self.VAD_SIZE / 1000),
        )

    @classmethod
    def from_config(cls, config: GladosConfig) -> "Glados":
        """
        Create a Glados instance from a configuration object.

        This factory method instantiates all required models and components based on
        the provided configuration, then creates and returns a fully configured
        Glados instance.

        Parameters:
            config (GladosConfig): Configuration object containing all settings for
                Glados initialization including model types, API endpoints, and
                behavior configurations.

        Returns:
            Glados: A new Glados instance configured with the provided settings.

        Raises:
            AssertionError: If specified voice is not available in the TTS system
            Exception: If model initialization fails for any component

        Notes:
            - Automatically selects appropriate ASR model based on engine type
            - Initializes VAD model with default settings
            - Chooses TTS model based on voice configuration
            - Converts personality prompts to required chat message format
            - All models are warmed up and ready for use upon return

        Example:
            >>> config = GladosConfig.from_yaml("config.yaml")
            >>> glados = Glados.from_config(config)
            >>> glados.start_listen_event_loop()
        """
        # Initialize ASR model based on configuration
        asr_model = get_audio_transcriber(engine_type=config.asr_engine)

        # Initialize voice activity detection model
        vad_model = VAD()

        # Initialize TTS model based on voice selection
        tts_model: tts_glados.Synthesizer | tts_kokoro.Synthesizer
        if config.voice == "glados":
            tts_model = tts_glados.Synthesizer()
        else:
            assert config.voice in tts_kokoro.get_voices(), f"Voice '{config.voice}' not available"
            tts_model = tts_kokoro.Synthesizer(voice=config.voice)

        # Create and return configured Glados instance
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
        Create a Glados instance from a YAML configuration file.

        This convenience factory method loads configuration from a YAML file and
        creates a fully configured Glados instance. It combines configuration
        loading and instance creation in a single method call.

        Parameters:
            path (str): Path to the YAML configuration file containing Glados 
                settings. The file should contain valid YAML with configuration
                parameters matching GladosConfig fields.

        Returns:
            Glados: A new Glados instance configured with settings from the 
                specified YAML file.

        Raises:
            FileNotFoundError: If the configuration file is not found
            yaml.YAMLError: If there is an error parsing the YAML configuration
            pydantic.ValidationError: If the configuration contains invalid values
            Exception: If model initialization fails for any component

        Notes:
            - Supports both UTF-8 and UTF-8-BOM encoded files
            - Validates all configuration parameters during loading
            - Creates and initializes all required models
            - Provides a simple interface for file-based configuration

        Example:
            >>> glados = Glados.from_yaml('config/glados_config.yaml')
            >>> glados.play_announcement()
            >>> glados.start_listen_event_loop()
        """
        return cls.from_config(GladosConfig.from_yaml(path))


def start() -> None:
    """
    Initialize and start the GLaDOS voice assistant from configuration file.

    This function provides a simple entry point for starting GLaDOS by loading
    configuration from the default YAML file, creating a Glados instance,
    playing the announcement, and starting the main listening loop.

    Parameters:
        None

    Returns:
        None

    Side Effects:
        - Loads configuration from "glados_config.yaml"
        - Creates and initializes Glados instance
        - Plays configured announcement
        - Starts the main audio listening event loop
        - Blocks until user interrupts or shutdown

    Raises:
        FileNotFoundError: If the configuration file "glados_config.yaml" is not found
        yaml.YAMLError: If there is an error parsing the YAML configuration file
        pydantic.ValidationError: If the configuration contains invalid values
        Exception: If any component initialization fails

    Notes:
        - Expects "glados_config.yaml" to exist in the current working directory
        - This is typically used as the main entry point for the application
        - The function will block until the user terminates the program
        - All background threads are automatically cleaned up on exit

    Example:
        >>> start()  # Starts GLaDOS with default configuration
        # Audio Modules Operational
        # Listening...
        # (blocks until user interrupts)
    """
    # Load configuration from default YAML file
    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    
    # Create Glados instance from configuration
    glados = Glados.from_config(glados_config)
    
    # Play initial announcement if configured
    glados.play_announcement()
    
    # Start main listening event loop
    glados.start_listen_event_loop()