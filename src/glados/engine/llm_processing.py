"""GLaDOS Engine LLM Processing - Language model interaction and response processing.

This module handles all aspects of language model processing including request
formatting, streaming response handling, content extraction from different LLM
server formats (OpenAI and Ollama), and sentence processing for the TTS pipeline.
"""

import json
import queue
import re
import time
from typing import Any

import requests
from loguru import logger


class LLMProcessingMixin:
    """Mixin class containing LLM processing methods for the Glados class.
    
    This mixin separates language model processing concerns from the main Glados class,
    providing methods for:
    - LLM server communication with streaming responses
    - Response format parsing (OpenAI and Ollama compatibility)
    - Real-time sentence processing and text cleanup
    - Integration with the TTS pipeline through queue management
    
    Note: This is designed as a mixin to be used with the main Glados class.
    It expects certain attributes and methods to be available from the parent class.
    """

    def process_llm(self) -> None:
        """Process text through Language Model and generate conversational responses.

        This method runs as a background thread, continuously retrieving detected text from
        the LLM queue and sending it to an LLM server. It streams the response in real-time,
        processes each chunk, and sends processed sentences to the text-to-speech queue.

        The method handles two primary LLM server formats:
        - OpenAI-compatible APIs (with 'data: ' prefixed responses)
        - Ollama format (direct JSON responses)

        Key Behaviors:
        - Continuously polls the LLM queue for detected text
        - Appends user messages to conversation history
        - Sends streaming requests to LLM server
        - Processes response chunks in real-time for immediate TTS
        - Breaks sentences at punctuation marks for natural speech flow
        - Handles interruptions through processing flags
        - Adds end-of-stream token to signal completion

        Side Effects:
        - Modifies `self.messages` by appending user messages
        - Puts processed sentences into `self.tts_queue`
        - Logs debug information and processing errors
        - Respects interruption signals from audio processing

        Error Handling:
        - Handles empty queue timeouts gracefully
        - Catches and logs errors during line processing without crashing
        - Stops processing when shutdown event is set or processing flag is False
        - Maintains robust operation despite network or parsing errors

        Notes:
        - Uses timeout mechanism to prevent blocking and allow shutdown
        - Supports graceful interruption of LLM processing for responsive interaction
        - Sentence segmentation preserves natural speech patterns
        - Number handling prevents inappropriate sentence breaks (e.g., "1.5")
        """
        while not self.shutdown_event.is_set():
            try:
                # Wait for detected text from speech recognition
                detected_text = self.llm_queue.get(timeout=0.1)
                
                # Add user message to conversation history
                self.messages.append({"role": "user", "content": detected_text})

                # Prepare request data for LLM server
                data = {
                    "model": self.model,
                    "stream": True,
                    "messages": self.messages,
                }
                
                logger.debug(f"Starting LLM request with messages: {self.messages}")
                logger.debug("Performing request to LLM server...")

                # Send streaming request to LLM server
                with requests.post(
                    str(self.completion_url),
                    headers=self.prompt_headers,
                    json=data,
                    stream=True,
                ) as response:
                    sentence = []
                    
                    # Process each line of the streaming response
                    for line in response.iter_lines():
                        # Check for interruption signal
                        if self.processing is False:
                            break  # Halt processing if interrupted by new voice input
                            
                        if line:  # Filter out empty keep-alive lines
                            try:
                                # Parse and clean the response line
                                cleaned_line = self._clean_raw_bytes(line)
                                if cleaned_line:
                                    chunk = self._process_chunk(cleaned_line)
                                    if chunk:
                                        sentence.append(chunk)
                                        
                                        # Check for sentence completion markers
                                        if chunk in self.PUNCTUATION_SET and (
                                            len(sentence) < 2 or not sentence[-2].strip().isdigit()
                                        ):  # Don't split on numbers like "1.5"
                                            logger.info(f"Sentence complete with punctuation: {chunk}")
                                            self._process_sentence(sentence)
                                            sentence = []
                                            
                            except Exception as e:
                                logger.error(f"Error processing LLM response line: {e}")
                                continue

                    # Process any remaining sentence content
                    if self.processing and sentence:
                        self._process_sentence(sentence)
                        
                    # Signal end of response to TTS pipeline
                    self.tts_queue.put("<EOS>")
                    
            except queue.Empty:
                # No new text to process, continue monitoring
                time.sleep(self.PAUSE_TIME)

    def _process_sentence(self, current_sentence: list[str]) -> None:
        """Process and clean sentence for text-to-speech synthesis.

        This method handles text preprocessing for the TTS system, removing formatting
        artifacts and normalizing text before sending it to the TTS queue. It ensures
        that only clean, speakable text reaches the speech synthesis pipeline.

        Parameters:
        - current_sentence (list[str]): List of text fragments forming a complete sentence

        Side Effects:
        - Puts cleaned sentence text into the TTS queue
        - Only processes non-empty sentences after cleaning

        Text Cleaning Operations:
        - Removes text enclosed in asterisks (*) for action descriptions
        - Removes text enclosed in parentheses () for side comments
        - Replaces double newlines with periods for proper sentence breaks
        - Replaces single newlines with periods to maintain flow
        - Normalizes multiple spaces to single spaces
        - Removes colons that might interfere with natural speech

        Notes:
        - Preserves sentence meaning while removing formatting artifacts
        - Optimized for natural speech synthesis output
        - Handles markdown-style formatting commonly found in LLM responses
        - Empty sentences are discarded to prevent audio gaps
        """
        # Join sentence fragments into complete text
        sentence = "".join(current_sentence)
        
        # Remove action descriptions and side comments
        sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
        
        # Normalize line breaks and spacing for speech
        sentence = sentence.replace("\n\n", ". ").replace("\n", ". ").replace("  ", " ").replace(":", " ")
        
        # Only queue non-empty sentences for TTS
        if sentence:
            self.tts_queue.put(sentence)

    def _clean_raw_bytes(self, line: bytes) -> dict[str, Any] | None:
        """Parse and clean raw server response bytes into standardized format.

        This method handles response parsing for multiple LLM server formats,
        converting both OpenAI and Ollama response formats into a consistent
        dictionary structure for downstream processing.

        Parameters:
        - line (bytes): Raw response line from the LLM server

        Returns:
        - dict[str, Any] | None: Parsed JSON response or None if parsing fails

        Supported Formats:
        - OpenAI format: Lines starting with "data: " followed by JSON
        - Ollama format: Direct JSON responses without prefix

        Error Handling:
        - Gracefully handles malformed JSON responses
        - Logs warnings for parsing failures without crashing
        - Returns None for unparseable content to allow processing continuation

        Notes:
        - Maintains compatibility with multiple LLM server implementations
        - Provides unified interface for diverse response formats
        - Essential for supporting both local and cloud LLM services
        """
        try:
            # Handle OpenAI-compatible format with "data: " prefix
            if line.startswith(b"data: "):
                json_str = line.decode("utf-8")[6:]  # Remove 'data: ' prefix
                parsed_json: dict[str, Any] = json.loads(json_str)
                return parsed_json
                
            # Handle Ollama format (direct JSON)
            else:
                parsed_json = json.loads(line.decode("utf-8"))
                if isinstance(parsed_json, dict):
                    return parsed_json
                return None
                
        except Exception as e:
            logger.warning(f"Failed to parse LLM server response: {e}")
            return None

    def _process_chunk(self, line: dict[str, Any]) -> str | None:
        """Extract text content from LLM server response chunk.

        This method safely extracts the actual text content from parsed LLM server
        responses, handling the different JSON structures used by OpenAI and Ollama
        format responses.

        Parameters:
        - line (dict[str, Any]): Parsed JSON response chunk from LLM server

        Returns:
        - str | None: Extracted text content or None if no content found

        Response Format Handling:
        - OpenAI format: Extracts from choices[0].delta.content
        - Ollama format: Extracts from message.content
        - Handles missing or malformed response structures gracefully

        Error Handling:
        - Returns None for empty or invalid input
        - Logs errors without crashing to maintain stream processing
        - Handles missing nested dictionary keys safely

        Notes:
        - Critical for real-time text extraction during streaming responses
        - Maintains robustness across different LLM server implementations
        - Enables seamless switching between local and cloud LLM services
        """
        if not line or not isinstance(line, dict):
            return None

        try:
            # Handle OpenAI-compatible format
            if "choices" in line:
                content = line.get("choices", [{}])[0].get("delta", {}).get("content")
                return content if content else None
                
            # Handle Ollama format
            else:
                content = line.get("message", {}).get("content")
                return content if content else None
                
        except Exception as e:
            logger.error(f"Error extracting content from LLM chunk: {e}")
            return None