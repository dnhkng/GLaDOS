import queue
from typing import Any

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
        audio_path: str = "data/slow-clap.mp3"
    ):
        """
        Initializes the tool with a queue for communication with the LLM.

        Args:
            audio_path: A path to the slow clap audio file
            llm_queue: A queue for sending tool results to the language model.
        """
        self.llm_queue = llm_queue
        self.audio_path = audio_path


    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        """
        Executes the slow clap by playing an audio file multiple times.

        Args:
            tool_call_id: Unique identifier for the tool call.
            call_args: Arguments passed by the LLM related to this tool call.
        """
        try:
            claps = int(
                call_args.get("claps", 1)
            )
            # clamp between 1 and 5
            claps = max(1, min(claps, 5))

        except (ValueError, TypeError):
            # default to 1 clap
            claps = 1

        try:
            # Load the audio file
            data, sample_rate = sf.read(self.audio_path)

            # Play the sound for each clap
            for _ in range(claps):
                sd.play(data, sample_rate)
                sd.wait()
            self.llm_queue.put({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "success",
                "type": "function_call_output"
            })

        except FileNotFoundError:
            # Raised if the audio file is not found at the specified path
            print(f"Error: Audio not found. Check the path: {self.audio_path}")

        except ValueError as ve:
            # Raised by soundfile for invalid file formats or parameters
            print(f"ValueError: {ve}")

        except sd.PortAudioError as pa_err:
            # Raised by sounddevice for audio device-related issues
            print(f"PortAudioError: {pa_err}")
