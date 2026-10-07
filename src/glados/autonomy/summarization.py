"""Estimate conversation token use for context budgeting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


from .token_estimator import TokenEstimator, get_default_estimator

if TYPE_CHECKING:
    pass


def estimate_tokens(
    messages: list[dict[str, Any]],
    estimator: TokenEstimator | None = None,
) -> int:
    """
    Estimate token count for messages.

    Args:
        messages: List of message dicts to estimate tokens for.
        estimator: Optional token estimator. Uses default if not provided.

    Returns:
        Estimated token count.
    """
    if estimator is None:
        estimator = get_default_estimator()
    return estimator.estimate(messages)
