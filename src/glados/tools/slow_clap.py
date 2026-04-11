import queue
from typing import Any

from loguru import logger
import soundfile as sf

from glados.audio_io import get_audio_system

tool_definition = {
    "type": "function",
    "function": {
        "name": "slow clap",
        "description": "Performs a slow clap.",
        "parameters": {
            "type": "object",
            "properties": {
                "claps": {
                    "type": "number",
                    "description": "The number of slow claps to perform."
                }
            },
            "required": ["claps"]
        }
    }
}

class SlowClap:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the tool with a queue for communication with the LLM.

        Args:
            llm_queue: A queue for sending tool results to the language model.
            tool_config: Configuration dictionary containing tool settings.
        """
        self.llm_queue = llm_queue
        tool_config = tool_config or {}
        self.audio_path = tool_config.get("slow_clap_audio_path", "data/slow-clap.mp3")
        self.audio_io = tool_config.get("audio_io")
        if self.audio_io is None:
            self.audio_io = get_audio_system()

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        """
        Executes the slow clap by playing an audio file multiple times.

        Args:
            tool_call_id: Unique identifier for the tool call.
            call_args: Arguments passed by the LLM related to this tool call.
        """
        try:
            claps = int(call_args.get("claps", 1))
            claps = max(1, min(claps, 5))  # clamp between 1 and 5
        except (ValueError, TypeError):
            claps = 1

        try:
            data, sample_rate = sf.read(self.audio_path, dtype='float32')

            for _ in range(claps):
                self.audio_io.start_speaking(data, sample_rate=sample_rate, wait=True)
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Success. The tool played a slow clap audio to the user. You do not need to narrate the clapping.",
                    "type": "function_call_output",
                }
            )

        except FileNotFoundError:
            error_msg = f"error: audio file not found at {self.audio_path}"
            logger.error(f"SlowClap: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )

        except ValueError as ve:
            error_msg = f"error: invalid audio file - {ve}"
            logger.error(f"SlowClap: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )

        except Exception as other_error:
            error_msg = f"error: other (possibly audio device) - {other_error}"
            logger.error(f"SlowClap: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )
