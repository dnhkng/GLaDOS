from __future__ import annotations

from typing import Final

# Instructions for the LLM to handle vision messages from the vision module.
# These instructions are essential for proper integration of vision observations into the conversation.
SYSTEM_PROMPT_VISION_HANDLING: Final[str] = (
    "Important vision instructions: "
    "- You receive the latest camera snapshot in a system message prefixed with '[vision]'. Treat it as context, not a user message. "
    "- Do not respond directly to the [vision] snapshot unless the user asks about the scene. "
    "- When a user asks for detailed visual inspection or verification, call the `vision_look` tool with a short prompt describing what to check. "
    "- Use the vision snapshot to ground answers, mentioning only relevant or changed elements."
)

# Default prompts for FastVLM inference.
VISION_DEFAULT_PROMPT: Final[str] = "Describe the image briefly, focusing on salient elements."
VISION_DETAIL_PROMPT: Final[str] = "Describe the image in detail."
