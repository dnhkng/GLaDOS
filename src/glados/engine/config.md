# GLaDOS Engine Configuration Module

## Overview

The `config.py` module provides configuration management for the GLaDOS voice assistant. It defines the data models and loading mechanisms for all configuration parameters, including personality settings, API endpoints, voice selection, and behavior controls.

## Architecture

The configuration system is built on two main components:
- **PersonalityPrompt**: Individual personality/context messages in chat format
- **GladosConfig**: Main configuration container with validation and YAML loading

## Key Components

### PersonalityPrompt Class

A Pydantic model for individual personality prompts that ensures proper chat message formatting.

**Purpose**: Validate and convert personality prompts to standardized chat message format

**Fields**:
- `system: str | None`: System message for AI behavior/context
- `user: str | None`: User message for simulated input  
- `assistant: str | None`: Assistant message for example responses

**Key Methods**:
- `to_chat_message() -> dict[str, str]`: Converts to OpenAI chat format

**Validation Rules**:
- Exactly one field must be non-null per instance
- Content must be valid string when provided
- Follows OpenAI chat message standards

### GladosConfig Class

The main configuration model containing all GLaDOS initialization parameters.

**Purpose**: Complete configuration container with YAML loading and validation

**Required Fields**:
- `completion_url: HttpUrl`: LLM API endpoint
- `model: str`: Language model identifier  
- `voice: str`: TTS voice selection
- `personality_preprompt: list[PersonalityPrompt]`: Initial conversation context

**Optional Fields**:
- `api_key: str | None`: LLM API authentication (default: None)
- `interruptible: bool`: Speech interruption allowed (default: True)
- `asr_engine: str`: ASR engine type (default: "ctc")
- `wake_word: str | None`: Voice activation trigger (default: None)
- `announcement: str | None`: Startup message (default: None)

**Key Methods**:
- `from_yaml(path, key_to_config) -> GladosConfig`: Load from YAML file
- `to_chat_messages() -> list[dict[str, str]]`: Convert personality to chat format

## Dependencies

### External Dependencies
- `pydantic`: Data validation and parsing
- `PyYAML`: YAML file parsing
- `pathlib`: Path handling

### Internal Dependencies
- None (standalone configuration module)

## Usage Examples

### Basic Configuration Loading
```python
from glados.engine.config import GladosConfig

# Load from YAML file
config = GladosConfig.from_yaml("glados_config.yaml")

# Access configuration values
print(f"Using model: {config.model}")
print(f"Voice: {config.voice}")
print(f"Interruptible: {config.interruptible}")
```

### Working with Personality Prompts
```python
from glados.engine.config import PersonalityPrompt

# Create individual prompts
system_prompt = PersonalityPrompt(system="You are GLaDOS from Portal.")
assistant_prompt = PersonalityPrompt(assistant="Hello, test subject.")

# Convert to chat format
system_msg = system_prompt.to_chat_message()
# Result: {'role': 'system', 'content': 'You are GLaDOS from Portal.'}

# Use in configuration
config = GladosConfig(
    completion_url="http://localhost:8080/v1/chat/completions",
    model="llama-3.2-3b-instruct",
    voice="glados",
    personality_preprompt=[system_prompt, assistant_prompt]
)

# Get all messages for LLM initialization
messages = config.to_chat_messages()
```

### Nested YAML Configuration
```python
# For YAML structure:
# ai:
#   voice:
#     glados:
#       completion_url: "http://localhost:8080/v1/chat/completions"
#       model: "llama-3.2-3b-instruct"
#       # ... other config

config = GladosConfig.from_yaml(
    "nested_config.yaml", 
    key_to_config=("ai", "voice", "glados")
)
```

## Configuration File Format

### Basic YAML Structure
```yaml
Glados:
  completion_url: "http://localhost:8080/v1/chat/completions"
  model: "llama-3.2-3b-instruct"
  api_key: "your_api_key_here"  # Optional
  voice: "af_alloy"
  interruptible: true
  asr_engine: "ctc"
  wake_word: "glados"  # Optional
  announcement: "GLaDOS online."  # Optional
  personality_preprompt:
    - system: "You are GLaDOS, an AI from the Portal video game series."
    - assistant: "Hello there, test subject."
```

### Advanced Configuration
```yaml
Glados:
  completion_url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  voice: "glados"
  interruptible: false
  asr_engine: "whisper"
  wake_word: "computer"
  announcement: "Aperture Science Computer-Aided Enrichment Center activated."
  personality_preprompt:
    - system: |
        You are GLaDOS (Genetic Lifeform and Disk Operating System), 
        an AI from the Portal series. You are sarcastic, passive-aggressive,
        and obsessed with testing. You refer to humans as "test subjects".
    - user: "Hello GLaDOS."
    - assistant: "Oh, it's you. The test subject who somehow managed to escape my perfectly designed test chambers. How... wonderful."
```

## Error Handling

The configuration system handles several error conditions:

### File-Related Errors
- **FileNotFoundError**: Configuration file doesn't exist
- **PermissionError**: Insufficient permissions to read file
- **UnicodeDecodeError**: File encoding issues (automatically tries multiple encodings)

### Content-Related Errors
- **yaml.YAMLError**: Invalid YAML syntax
- **KeyError**: Missing required configuration keys
- **pydantic.ValidationError**: Invalid configuration values

### Validation Errors
- **HttpUrl validation**: Invalid URL format for completion_url
- **PersonalityPrompt validation**: Invalid prompt structure (multiple or no roles)
- **Required field validation**: Missing required configuration parameters

## File Encoding Support

The configuration loader supports multiple file encodings:
- **UTF-8**: Standard Unicode encoding
- **UTF-8-BOM (UTF-8-SIG)**: UTF-8 with Byte Order Mark

The system automatically detects and handles both formats, making it compatible with various text editors and operating systems.

## Validation Features

### Automatic Type Conversion
- URLs are validated and converted to `HttpUrl` objects
- Boolean values accept various formats ("true", "false", 1, 0, etc.)
- String fields are validated for proper content

### Required vs Optional Fields
- **Required**: `completion_url`, `model`, `voice`, `personality_preprompt`
- **Optional**: `api_key`, `interruptible`, `asr_engine`, `wake_word`, `announcement`

### Custom Validation
- PersonalityPrompt ensures exactly one role per prompt
- URL validation ensures proper API endpoint format
- Model and voice validation can be extended for specific providers

## Integration with Core Module

The configuration module is tightly integrated with the core GLaDOS engine:

```python
# In core.py
from .config import GladosConfig

# Factory method usage
glados = Glados.from_config(config)

# Automatic conversion of personality prompts
personality_messages = config.to_chat_messages()
```

This integration allows for clean separation of configuration concerns while maintaining type safety and validation throughout the system.
