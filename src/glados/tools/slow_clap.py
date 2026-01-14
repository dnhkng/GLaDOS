import queue
from typing import Any

from loguru import logger
import sounddevice as sd  # type: ignore
import soundfile as sf

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
            data, sample_rate = sf.read(self.audio_path)

            for _ in range(claps):
                sd.play(data, sample_rate)
                sd.wait()
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "success",
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

        except sd.PortAudioError as pa_err:
            error_msg = f"error: audio device error - {pa_err}"
            logger.error(f"SlowClap: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )
