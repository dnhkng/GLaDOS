"""Shared one-token option scoring for routing and background PAD decisions."""

import math
import requests


def token_ids(base_url, headers, labels, cache, timeout):
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


def option_request(model, messages, ids):
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


def request_scores(url, headers, data, ids, timeout):
    response = requests.post(url, headers=headers, json=data, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    entries = result["choices"][0]["logprobs"]["content"][0].get("top_probs")
    if entries is None:
        raise ValueError("Server does not expose untruncated post-sampling option scores")
    scores = {entry["id"]: entry["prob"] for entry in entries}
    values = [scores.get(token) for token in ids.values()]
    if any(v is None or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Incomplete option scores; no decision authorized")
    total = sum(values)
    if total <= 0:
        raise ValueError("Empty option distribution")
    return [value / total for value in values]
