"""
Simple LLM client for subagent use.

Provides a minimal interface for making LLM calls without the full
complexity of the main LLM processor.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import threading
import time
from typing import Any

from loguru import logger
import requests

from ..core.inference import InferenceCancelledError, InferenceScheduler


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""

    url: str
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout: float = 30.0
    request_options: dict[str, Any] = field(default_factory=dict)

    scheduler: InferenceScheduler | None = None
    shutdown_event: threading.Event | None = None
    owner: str = "background"
    lane: str = "autonomy"
    cancelled: Callable[[], bool] = lambda: False
    deadline: float | None = None

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def llm_call(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str | list,
    json_response: bool = False,
) -> str | None:
    """
    Make a simple LLM call.

    Returns the assistant's response text, or None on error.
    """
    if config.cancelled():
        return None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    data = {
        **config.request_options,
        "model": config.model,
        "messages": messages,
        "stream": False,
    }

    if json_response:
        data["response_format"] = {"type": "json_object"}

    try:
        guard = (
            config.scheduler.lease(
                config.owner,
                config.lane,
                config.model,
                lambda: config.cancelled() or bool(config.shutdown_event and config.shutdown_event.is_set())
                or (config.deadline is not None and time.monotonic() >= config.deadline),
            )
            if config.scheduler
            else nullcontext()
        )
        with guard:
            timeout = config.timeout
            if config.deadline is not None:
                timeout = min(timeout, config.deadline - time.monotonic())
                if timeout <= 0 or config.cancelled():
                    return None
            response = requests.post(
                config.url,
                headers=config.headers,
                json=data,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
        if config.cancelled():
            return None

        # Handle OpenAI-style response
        if result.get("choices"):
            return result["choices"][0]["message"]["content"]

        # Handle Ollama-style response
        if "message" in result:
            return result["message"].get("content")

        logger.warning("LLM call: unexpected response format")
        return None

    except InferenceCancelledError:
        return None
    except requests.Timeout:
        logger.warning("LLM call timed out")
        return None
    except requests.RequestException as e:
        logger.warning("LLM call failed: %s", e)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("LLM call: failed to parse response: %s", e)
        return None
