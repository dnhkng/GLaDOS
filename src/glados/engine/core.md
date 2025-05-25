# GLaDOS Engine Core Module

## Overview

The `core.py` module contains the main `Glados` class and core initialization functionality for the GLaDOS voice assistant. This module serves as the primary entry point and orchestrates all components of the voice interaction system.

## Architecture

The core module handles:
- **Component Initialization**: Sets up ASR, TTS, VAD, and LLM components
- **Thread Management**: Manages background processing threads
- **Configuration Loading**: Factory methods for creating instances from config
- **Audio Stream Setup**: Initializes real-time audio input processing
- **State Management**: Maintains conversation history and processing flags

## Key Components

### Glados Class

The main class that orchestrates the entire voice assistant system.

**Core Responsibilities:**
- Initialize all AI models (ASR, TTS, VAD)
- Set up background processing threads
- Manage audio input/output streams
- Maintain conversation state
- Handle configuration and factory creation

**Threading Architecture:**
- **LLM Thread**: Processes language model requests and responses
- **TTS Thread**: Handles text-to-speech generation
- **Audio Thread**: Manages audio playback and interruption detection
- **Main Thread**: Handles audio input and voice activity detection

### Factory Methods

#### `from_config(config: GladosConfig) -> Glados`
- Creates instance from configuration object
- Initializes all required models based on config
- Returns fully configured Glados instance

#### `from_yaml(path: str) -> Glados`
- Loads configuration from YAML file
- Convenience method combining config loading and instance creation

### Class Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `PAUSE_TIME` | 0.05 | Processing loop delay (seconds) |
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `VAD_SIZE` | 32 | VAD window size (milliseconds) |
| `VAD_THRESHOLD` | 0.8 | Voice activity detection threshold |
| `BUFFER_SIZE` | 800 | Pre-activation buffer (milliseconds) |
| `PAUSE_LIMIT` | 640 | Speech pause detection (milliseconds) |
| `SIMILARITY_THRESHOLD` | 2 | Wake word Levenshtein distance |

## Dependencies

### Internal Dependencies
- `config.GladosConfig`: Configuration model
- `audio_models.AudioMessage`: Audio data structure
- `../TTS.tts_glados`: GLaDOS voice synthesis
- `../TTS.tts_kokoro`: Kokoro voice synthesis
- `../utils.spoken_text_converter`: Text preprocessing
- `glados.ASR`: Speech recognition components

### External Dependencies
- `sounddevice`: Real-time audio I/O
- `numpy`: Audio data processing
- `pydantic`: Configuration validation
- `loguru`: Logging system
- `threading`: Background task management
- `queue`: Inter-thread communication

## Coupling Analysis

### High Coupling
- **Audio Processing**: Tight integration with sounddevice for real-time I/O
- **Model Components**: Direct dependency on ASR, TTS, and VAD models
- **Background Threads**: Manages lifecycle of processing threads

### Medium Coupling
- **Configuration**: Uses GladosConfig for initialization parameters
- **Queue Communication**: Interfaces with processing modules via queues

### Low Coupling
- **Logging**: Uses loguru for status reporting
- **Utilities**: Minimal dependency on utility functions

## Usage Examples

### Basic Initialization
```python
from glados.engine.core import Glados
from glados.engine.config import GladosConfig

# From configuration file
glados = Glados.from_yaml("config.yaml")

# From configuration object
config = GladosConfig.from_yaml("config.yaml")
glados = Glados.from_config(config)
```

### Manual Initialization
```python
from glados.ASR import get_audio_transcriber, VAD
from glados.TTS import tts_kokoro

asr_model = get_audio_transcriber("ctc")
tts_model = tts_kokoro.Synthesizer(voice="af_alloy")
vad_model = VAD()

glados = Glados(
    asr_model=asr_model,
    tts_model=tts_model,
    vad_model=vad_model,
    completion_url="http://localhost:8080/v1/chat/completions",
    model="llama-3.2-3b-instruct",
    interruptible=True
)
```

### Application Entry Point
```python
from glados.engine.core import start

# Start with default configuration
start()
```

## Error Handling

The core module handles several error conditions:

- **Configuration Errors**: Invalid YAML or missing required fields
- **Model Initialization**: Failed loading of ASR, TTS, or VAD models
- **Audio Stream Errors**: Sounddevice initialization failures
- **Thread Management**: Background thread startup issues

## Performance Considerations

- **Model Warmup**: ASR model is warmed up during initialization
- **Thread Efficiency**: Background threads use timeouts to prevent blocking
- **Memory Management**: Queues are bounded to prevent excessive memory usage
- **Audio Latency**: Real-time audio processing with minimal buffering

## Configuration Requirements

The core module requires these configuration parameters:

### Required Fields
- `completion_url`: LLM API endpoint
- `model`: LLM model identifier
- `voice`: TTS voice selection
- `personality_preprompt`: Initial conversation context

### Optional Fields
- `api_key`: LLM API authentication
- `wake_word`: Voice activation trigger
- `announcement`: Startup message
- `interruptible`: Interruption behavior
- `asr_engine`: ASR model type

## Thread Safety

The core module implements thread-safe communication:
- **Queue-based Communication**: All inter-thread data transfer uses queues
- **Event Synchronization**: Threading.Event for state coordination
- **Mutex Protection**: Critical sections protected where necessary
- **Graceful Shutdown**: Coordinated thread termination via shutdown events