"""
LLM-driven message summarization for conversation compaction.

Uses the LLM to summarize messages and extract facts, following
the LLM-first principle (complex reasoning in prompts, not code).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from .llm_client import LLMConfig, llm_call


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimate token count for messages.

    Uses simple word-based estimation (4 chars per token average).
    For production, consider tiktoken for accuracy.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Handle multi-part messages
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
    return total_chars // 4


def summarize_messages(
    messages: list[dict[str, Any]],
    llm_config: LLMConfig,
) -> str | None:
    """
    Use LLM to summarize a list of conversation messages.

    Returns a concise summary preserving key information.
    """
    if not messages:
        return None

    # Format messages for the prompt
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            formatted.append(f"{role}: {content}")
        elif isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            text = " ".join(text_parts).strip()
            if text:
                formatted.append(f"{role}: {text}")

    if not formatted:
        return None

    conversation = "\n".join(formatted)

    system_prompt = """You are a summarization assistant. Summarize the conversation below.

Rules:
- Be concise but preserve important context
- Mention key topics discussed
- Note any decisions made or tasks assigned
- Mention any facts learned about the user
- Keep to 2-4 sentences maximum"""

    user_prompt = f"Summarize this conversation:\n\n{conversation}"

    response = llm_call(llm_config, system_prompt, user_prompt)
    if response:
        logger.debug("Summarized {} messages into summary", len(messages))
    return response


def extract_facts(
    messages: list[dict[str, Any]],
    llm_config: LLMConfig,
) -> list[str]:
    """
    Use LLM to extract factual information from messages.

    Returns a list of discrete facts worth remembering.
    """
    if not messages:
        return []

    # Format messages for the prompt
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            formatted.append(f"{role}: {content}")

    if not formatted:
        return []

    conversation = "\n".join(formatted)

    system_prompt = """You extract important facts from conversations.

Output one fact per line. Include facts like:
- User preferences (e.g., "User prefers dark mode")
- Personal information (e.g., "User's name is David")
- Important decisions (e.g., "Decided to use Python for the project")
- Technical context (e.g., "Project uses FastAPI")

Only output facts that are worth remembering long-term.
Output nothing if no important facts are present.
Do not include conversational pleasantries or ephemeral details."""

    user_prompt = f"Extract facts from this conversation:\n\n{conversation}"

    response = llm_call(llm_config, system_prompt, user_prompt)
    if not response:
        return []

    # Parse response into list of facts
    facts = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            # Remove bullet points if present
            if line.startswith("- "):
                line = line[2:]
            elif line.startswith("* "):
                line = line[2:]
            if line:
                facts.append(line)

    logger.debug("Extracted {} facts from {} messages", len(facts), len(messages))
    return facts
