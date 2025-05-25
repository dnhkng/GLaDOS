"""TUI integration mixin for the GLaDOS engine.

This module provides the TUIIntegrationMixin class that extends the GLaDOS
engine with capabilities to communicate with the TUI interface, sending
real-time updates and notifications.
"""

import time
from typing import Callable, Dict, Any
import threading
from loguru import logger


class TUIIntegrationMixin:
    """Mixin to provide TUI integration capabilities to the GLaDOS engine.
    
    This mixin extends the GLaDOS engine with methods to register TUI callbacks
    and send real-time updates to the user interface for various engine events
    such as TTS interruptions, chain of thought processes, and tool calling.
    
    Features:
    - Thread-safe callback registration and execution
    - Multiple event type support
    - Error handling for TUI callback failures
    - Enhanced method overrides with TUI notifications
    """

    def __init__(self, *args, **kwargs):
        """Initialize TUI integration capabilities.
        
        Sets up callback storage and thread synchronization for safe
        TUI communication from engine threads.
        """
        super().__init__(*args, **kwargs)
        self.tui_callbacks: Dict[str, Callable] = {}
        self._tui_update_lock = threading.Lock()

    def register_tui_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback function for TUI updates.
        
        Allows the TUI to register callback functions that will be called
        when specific engine events occur, enabling real-time UI updates.
        
        Args:
            event_type: Type of event to listen for
                       ('tts_interruption', 'chain_of_thought', 'tool_calling', etc.)
            callback: Function to call when the event occurs
        
        Thread Safety:
            Uses a lock to ensure thread-safe callback registration
        """
        with self._tui_update_lock:
            self.tui_callbacks[event_type] = callback
            logger.debug(f"Registered TUI callback for event type: {event_type}")

    def unregister_tui_callback(self, event_type: str) -> None:
        """Unregister a TUI callback for the specified event type.
        
        Args:
            event_type: Type of event to stop listening for
        """
        with self._tui_update_lock:
            if event_type in self.tui_callbacks:
                del self.tui_callbacks[event_type]
                logger.debug(f"Unregistered TUI callback for event type: {event_type}")

    def notify_tui(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send an update to the TUI if a callback is registered.
        
        Safely calls the registered TUI callback with the provided data,
        handling any errors that occur during callback execution.
        
        Args:
            event_type: Type of event that occurred
            data: Dictionary containing event-specific data
        
        Error Handling:
            Catches and logs any exceptions that occur during callback execution
            to prevent TUI errors from affecting engine operation
        """
        callback = None
        
        # Safely get callback reference
        with self._tui_update_lock:
            callback = self.tui_callbacks.get(event_type)
        
        if callback:
            try:
                # Call the TUI callback with event data
                callback(data)
                logger.debug(f"TUI notified of {event_type} event")
            except Exception as e:
                logger.error(f"TUI callback error for {event_type}: {e}")
        else:
            logger.debug(f"No TUI callback registered for event type: {event_type}")

    def clip_interrupted_sentence(self, generated_text: str, percentage_played: float) -> str:
        """Enhanced sentence clipping with TUI notification.
        
        Overrides the base implementation to add TUI notification when
        TTS is interrupted, providing real-time feedback to the user.
        
        Args:
            generated_text: The complete text that was being spoken
            percentage_played: Percentage of the text that was actually played
        
        Returns:
            str: The clipped text with interruption marker if applicable
        
        Side Effects:
            Sends a TUI notification with interruption details
        """
        # Call parent implementation to get clipped text
        clipped_text = super().clip_interrupted_sentence(generated_text, percentage_played)
        
        # Notify TUI of the interruption with detailed information
        self.notify_tui("tts_interruption", {
            "original_text": generated_text,
            "clipped_text": clipped_text,
            "percentage": percentage_played,
            "reason": "voice_activation",
            "timestamp": time.strftime("%H:%M:%S")
        })
        
        return clipped_text

    def _process_llm_stream_chunk(self, chunk: str, reasoning_step: str = None) -> None:
        """Process LLM stream chunks with chain of thought notifications.
        
        This method can be called during LLM processing to send chain of
        thought updates to the TUI, showing the AI's reasoning process.
        
        Args:
            chunk: The text chunk being processed
            reasoning_step: Optional description of the current reasoning step
        
        Note:
            This is a placeholder method that can be integrated into the
            LLM processing pipeline for chain of thought display
        """
        if reasoning_step:
            self.notify_tui("chain_of_thought", {
                "step": reasoning_step,
                "reasoning": chunk,
                "timestamp": time.strftime("%H:%M:%S")
            })

    def _notify_tool_calling(
        self, 
        tool_name: str, 
        status: str, 
        parameters: Dict[str, Any] = None,
        result: Any = None,
        error: str = None
    ) -> None:
        """Notify TUI of tool calling events.
        
        Sends tool calling status updates to the TUI for display in the
        dynamic content panel.
        
        Args:
            tool_name: Name of the tool being called
            status: Current status ('starting', 'running', 'completed', 'error')
            parameters: Optional parameters passed to the tool
            result: Optional result from tool execution
            error: Optional error message if tool execution failed
        """
        data = {
            "tool_name": tool_name,
            "status": status,
            "timestamp": time.strftime("%H:%M:%S")
        }
        
        if parameters is not None:
            data["parameters"] = str(parameters)
        
        if result is not None:
            data["result"] = str(result)
        
        if error is not None:
            data["error"] = error
        
        self.notify_tui("tool_calling", data)

    def clear_tui_callbacks(self) -> None:
        """Clear all registered TUI callbacks.
        
        Useful for cleanup when shutting down or when TUI is disconnected.
        """
        with self._tui_update_lock:
            callback_count = len(self.tui_callbacks)
            self.tui_callbacks.clear()
            logger.debug(f"Cleared {callback_count} TUI callbacks")