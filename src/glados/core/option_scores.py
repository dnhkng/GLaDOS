"""Shared one-token option scoring for routing and background PAD decisions."""

from collections.abc import Sequence
import math
from typing import Any

import requests


def token_ids(
    base_url: str, headers: dict[str, str], labels: Sequence[str], cache: dict[str, int], timeout: float
) -> dict[str, int]:
    """Each label's single token id, asked of the server once and then cached."""
    for label in labels:
        if label not in cache:
            response = requests.post(
                base_url + "/tokenize", headers=headers, json={"content": label, "add_special": False}, timeout=timeout
            )
            response.raise_for_status()
            tokens = response.json().get("tokens", [])
            if len(tokens) != 1 or type(tokens[0]) is not int:
                raise ValueError(f"Option {label} is not one token for this model")
            cache[label] = tokens[0]
    return {label: cache[label] for label in labels}


def option_request(model: str, messages: list[dict[str, Any]], ids: dict[str, int]) -> dict[str, Any]:
    """A one-token completion constrained to the option tokens, reporting every option's probability."""
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 1,
        "samplers": ["temperature"],
        "top_k": 0,
        "top_p": 1,
        "min_p": 0,
        "repeat_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "logit_bias": {str(token): 100 for token in ids.values()},
        "logprobs": True,
        "top_logprobs": len(ids),
        "post_sampling_probs": True,
    }


def request_scores(
    url: str, headers: dict[str, str], data: dict[str, Any], ids: dict[str, int], timeout: float
) -> list[float]:
    """The options' probabilities, in ``ids`` order and normalised to sum to 1."""
    response = requests.post(url, headers=headers, json=data, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    entries = result["choices"][0]["logprobs"]["content"][0].get("top_probs")
    if entries is None:
        raise ValueError("Server does not expose untruncated post-sampling option scores")
    scores = {entry["id"]: entry["prob"] for entry in entries}
    values: list[float] = []
    for token in ids.values():
        value = scores.get(token)
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError("Incomplete option scores; no decision authorized")
        values.append(value)
    total = sum(values)
    if total <= 0:
        raise ValueError("Empty option distribution")
    return [value / total for value in values]
