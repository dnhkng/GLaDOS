"""
Async LLM decision-making helper.

Replaces hard-coded heuristics with LLM reasoning. All decision points
use this helper for consistent structured output and error handling.

Usage:
    class WeatherUrgency(BaseModel):
        notify_user: bool
        importance: float
        reason: str

    result = await llm_decide(
        prompt="Evaluate weather for notification: {weather}",
        context={"weather": weather_data},
        schema=WeatherUrgency,
        config=llm_config,
    )
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Shared executor for sync→async bridging
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm_decide")


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""
    url: str
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout: float = 30.0

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


async def llm_decide(
    prompt: str,
    context: dict[str, Any],
    schema: type[T],
    config: LLMConfig,
    system_prompt: str | None = None,
) -> T:
    """
    Make an async LLM decision with structured output.

    Args:
        prompt: Prompt template with {placeholders} for context
        context: Dict of values to format into prompt
        schema: Pydantic model for response validation
        config: LLM configuration
        system_prompt: Optional system prompt (defaults to decision-making prompt)

    Returns:
        Validated instance of the schema type

    Raises:
        LLMDecisionError: If LLM call fails or response is invalid
    """
    if system_prompt is None:
        system_prompt = (
            "You are a decision-making assistant. Analyze the input and respond "
            "with valid JSON matching the requested schema. Be concise and precise."
        )

    # Format the prompt with context
    formatted_prompt = prompt.format(**context)

    # Build the schema hint for the LLM
    schema_hint = _build_schema_hint(schema)
    user_message = f"{formatted_prompt}\n\nRespond with JSON matching this schema:\n{schema_hint}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    data = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                config.url,
                headers=config.headers,
                json=data,
            )
            response.raise_for_status()
            result = response.json()

        # Extract content from response
        content = _extract_content(result)
        if not content:
            raise LLMDecisionError("Empty response from LLM")

        # Parse and validate with Pydantic
        return schema.model_validate_json(content)

    except httpx.HTTPStatusError as e:
        raise LLMDecisionError(f"HTTP error: {e}") from e
    except httpx.TimeoutException as e:
        raise LLMDecisionError("Request timed out") from e
    except json.JSONDecodeError as e:
        raise LLMDecisionError(f"Invalid JSON response: {e}") from e
    except Exception as e:
        raise LLMDecisionError(f"Decision failed: {e}") from e


def llm_decide_sync(
    prompt: str,
    context: dict[str, Any],
    schema: type[T],
    config: LLMConfig,
    system_prompt: str | None = None,
    timeout: float = 30.0,
) -> T:
    """
    Synchronous wrapper for llm_decide().

    For use in sync code that needs LLM decisions.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - create one
        return asyncio.run(llm_decide(prompt, context, schema, config, system_prompt))

    # Running in async context - use executor
    future = asyncio.run_coroutine_threadsafe(
        llm_decide(prompt, context, schema, config, system_prompt),
        loop,
    )
    return future.result(timeout=timeout)


class LLMDecisionError(Exception):
    """Raised when an LLM decision fails."""
    pass


def _build_schema_hint(schema: type[BaseModel]) -> str:
    """Build a schema hint string for the LLM."""
    try:
        schema_dict = schema.model_json_schema()
        # Simplify for the LLM
        properties = schema_dict.get("properties", {})
        fields = []
        for name, prop in properties.items():
            field_type = prop.get("type", "any")
            description = prop.get("description", "")
            if description:
                fields.append(f'  "{name}": {field_type} // {description}')
            else:
                fields.append(f'  "{name}": {field_type}')
        return "{\n" + ",\n".join(fields) + "\n}"
    except Exception:
        return "{...}"


def _extract_content(result: dict[str, Any]) -> str | None:
    """Extract content from OpenAI or Ollama response format."""
    # OpenAI format
    if "choices" in result and result["choices"]:
        return result["choices"][0]["message"]["content"]

    # Ollama format
    if "message" in result:
        return result["message"].get("content")

    return None


# ============================================================================
# Pre-built decision schemas for common use cases
# ============================================================================


class WakeWordDecision(BaseModel):
    """Decision for wake word detection."""
    detected: bool
    confidence: float


class UrgencyDecision(BaseModel):
    """Decision for alert/notification urgency."""
    notify_user: bool
    importance: float
    reason: str


class TimingDecision(BaseModel):
    """Decision for autonomy timing."""
    should_speak: bool
    wait_seconds: int
    reason: str


class CompactionDecision(BaseModel):
    """Decision for message compaction."""
    indices_to_summarize: list[int]
    reason: str


class RelevanceDecision(BaseModel):
    """Decision for content relevance (news, etc.)."""
    relevant: bool
    importance: float
    summary: str
