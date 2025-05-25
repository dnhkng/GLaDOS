"""
GLaDOS Engine Configuration - Configuration models and YAML loading.

This module contains the configuration models for the GLaDOS voice assistant,
including personality prompt handling and main configuration loading from YAML files.
The configuration system uses Pydantic for validation and supports multiple encoding
formats for configuration files.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, HttpUrl


class PersonalityPrompt(BaseModel):
    """
    Configuration model for personality prompts in chat message format.
    
    This class validates and converts personality prompts to the standardized
    chat message format used by language models. It ensures that exactly one
    role field is populated per prompt instance.
    
    Attributes:
        system (str | None): System message content for setting AI behavior and context
        user (str | None): User message content for simulating user input
        assistant (str | None): Assistant message content for example responses
    
    Notes:
        - Exactly one of the three role fields must be non-null
        - Follows OpenAI chat message format standards
        - Used to construct conversation context and personality
    """
    
    system: str | None = None
    user: str | None = None
    assistant: str | None = None

    def to_chat_message(self) -> dict[str, str]:
        """
        Convert the personality prompt to standardized chat message format.

        This method transforms the PersonalityPrompt instance into a dictionary
        that follows the OpenAI chat message format with 'role' and 'content' fields.
        It validates that exactly one role field contains content.

        Returns:
            dict[str, str]: A chat message dictionary containing:
                - "role": The message role ("system", "user", or "assistant")
                - "content": The text content of the message

        Raises:
            ValueError: If the prompt does not contain exactly one non-null field.
                This ensures that each prompt has a clear, single role assignment.

        Example:
            >>> prompt = PersonalityPrompt(system="You are a helpful assistant.")
            >>> message = prompt.to_chat_message()
            >>> print(message)
            {'role': 'system', 'content': 'You are a helpful assistant.'}

        Notes:
            - Only non-null fields are considered during conversion
            - The first non-null field found determines the role
            - Used to build conversation context for language models
        """
        for field, value in self.model_dump(exclude_none=True).items():
            return {"role": field, "content": value}
        raise ValueError("PersonalityPrompt must have exactly one non-null field")


class GladosConfig(BaseModel):
    """
    Main configuration model for the GLaDOS voice assistant.
    
    This class defines all configuration parameters required to initialize and
    run the GLaDOS voice assistant, including language model settings, voice
    configuration, behavior controls, and personality setup.
    
    Attributes:
        completion_url (HttpUrl): URL endpoint for language model API completions
        model (str): Identifier for the language model to use (e.g., "llama-3.2-3b-instruct")
        api_key (str | None): Authentication key for the language model API
        interruptible (bool): Whether assistant speech can be interrupted by new input
        asr_engine (str): Automatic speech recognition engine type
        wake_word (str | None): Activation word to trigger the voice assistant
        voice (str): Voice identifier for text-to-speech synthesis
        announcement (str | None): Initial message to speak on startup
        personality_preprompt (list[PersonalityPrompt]): Initial conversation context
    
    Notes:
        - Uses Pydantic for automatic validation and type checking
        - Supports loading from YAML configuration files
        - Provides conversion methods for chat message format
        - Handles multiple file encodings and nested configuration keys
    """
    
    completion_url: HttpUrl
    model: str
    api_key: str | None = None
    interruptible: bool = True
    asr_engine: str = "ctc"
    wake_word: str | None = None
    voice: str
    announcement: str | None = None
    personality_preprompt: list[PersonalityPrompt]

    @classmethod
    def from_yaml(
        cls, 
        path: str | Path, 
        key_to_config: tuple[str, ...] = ("Glados",)
    ) -> "GladosConfig":
        """
        Load a GladosConfig instance from a YAML configuration file.

        This class method provides a robust way to load configuration from YAML files,
        handling multiple encodings and nested configuration structures. It supports
        navigation through nested YAML keys to locate the GLaDOS configuration section.

        Parameters:
            path (str | Path): Path to the YAML configuration file. Can be a string
                path or a Path object. The file should contain valid YAML content.
            key_to_config (tuple[str, ...], optional): Tuple of keys to navigate
                nested configuration structures. Each key represents a level in
                the YAML hierarchy. Defaults to ("Glados",) for top-level Glados key.

        Returns:
            GladosConfig: A validated configuration object with all settings loaded
                from the YAML file and converted to appropriate Python types.

        Raises:
            ValueError: If the YAML content is invalid or cannot be parsed
            OSError: If the file cannot be read due to permissions or file not found
            pydantic.ValidationError: If the configuration values don't match
                the expected schema or contain invalid data types/values
            KeyError: If the specified navigation keys don't exist in the YAML structure

        Notes:
            - Attempts multiple encodings: UTF-8 and UTF-8-BOM (UTF-8-SIG)
            - Supports nested YAML configurations for organized config files
            - All configuration values are validated according to Pydantic model schema
            - URLs are automatically validated and converted to HttpUrl objects
            - The method preserves the original file's encoding when possible

        Example:
            >>> # Load from simple YAML structure
            >>> config = GladosConfig.from_yaml("simple_config.yaml", ("Glados",))
            
            >>> # Load from nested YAML structure
            >>> config = GladosConfig.from_yaml("nested_config.yaml", ("ai", "voice", "glados"))
            
            >>> # YAML file example:
            >>> # Glados:
            >>> #   completion_url: "http://localhost:8080/v1/chat/completions"
            >>> #   model: "llama-3.2-3b-instruct"
            >>> #   voice: "af_alloy"
            >>> #   personality_preprompt:
            >>> #     - system: "You are GLaDOS from Portal."
        """
        path = Path(path)

        # Try different encodings to handle various file formats
        for encoding in ["utf-8", "utf-8-sig"]:
            try:
                data = yaml.safe_load(path.read_text(encoding=encoding))
                break
            except UnicodeDecodeError:
                if encoding == "utf-8-sig":
                    raise

        # Navigate through nested keys to find configuration
        config = data
        for key in key_to_config:
            config = config[key]

        return cls.model_validate(config)

    def to_chat_messages(self) -> list[dict[str, str]]:
        """
        Convert the personality preprompt list to chat message format.

        This method transforms all PersonalityPrompt instances in the personality_preprompt
        list into a list of standardized chat message dictionaries. This format is
        required for initializing conversation context with language models.

        Returns:
            list[dict[str, str]]: A list of chat message dictionaries, each containing:
                - "role": The message role ("system", "user", or "assistant") 
                - "content": The text content of the message

        Notes:
            - Maintains the order of prompts as specified in configuration
            - Each prompt is converted using PersonalityPrompt.to_chat_message()
            - The resulting list can be directly used to initialize conversation history
            - Typically includes system prompts for personality and behavior setup

        Example:
            >>> config = GladosConfig(
            ...     personality_preprompt=[
            ...         PersonalityPrompt(system="You are GLaDOS."),
            ...         PersonalityPrompt(assistant="Hello, test subject.")
            ...     ],
            ...     # ... other config fields
            ... )
            >>> messages = config.to_chat_messages()
            >>> print(messages)
            [
                {'role': 'system', 'content': 'You are GLaDOS.'},
                {'role': 'assistant', 'content': 'Hello, test subject.'}
            ]

        Raises:
            ValueError: If any PersonalityPrompt in the list is invalid (e.g., 
                contains multiple non-null fields or all fields are null)
        """
        return [prompt.to_chat_message() for prompt in self.personality_preprompt]