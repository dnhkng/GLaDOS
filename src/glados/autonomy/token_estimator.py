"""
Pluggable token estimation for conversation management.

Provides different strategies for estimating token counts in messages,
from simple character-based estimation to accurate tiktoken counting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from .config import TokenConfig


class TokenEstimator(ABC):
    """Abstract base class for token estimation strategies."""

    @abstractmethod
    def estimate(self, messages: list[dict[str, Any]]) -> int:
        """
        Estimate the token count for a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Estimated token count.
        """
        ...

    @abstractmethod
    def estimate_text(self, text: str) -> int:
        """
        Estimate the token count for a single text string.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        ...


class SimpleTokenEstimator(TokenEstimator):
    """
    Simple character-based token estimator.

    Uses a configurable characters-per-token ratio (default 4.0).
    Fast but less accurate than tiktoken-based estimation.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        """
        Initialize the simple estimator.

        Args:
            chars_per_token: Average characters per token (default 4.0).
        """
        self._chars_per_token = chars_per_token

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        """Estimate tokens using character count divided by chars_per_token."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle multi-part messages (e.g., vision messages)
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total_chars += len(part["text"])
        return int(total_chars / self._chars_per_token)

    def estimate_text(self, text: str) -> int:
        """Estimate tokens for a single text string."""
        return int(len(text) / self._chars_per_token)


class TiktokenEstimator(TokenEstimator):
    """
    Accurate token estimator using tiktoken library.

    Provides exact token counts for OpenAI/Anthropic models.
    Falls back to simple estimation if tiktoken is not available.
    """

    def __init__(self, model: str = "cl100k_base", fallback_chars_per_token: float = 4.0) -> None:
        """
        Initialize the tiktoken estimator.

        Args:
            model: Tiktoken encoding name (default 'cl100k_base' for GPT-4/Claude).
            fallback_chars_per_token: Chars per token for fallback estimation.
        """
        self._encoding = None
        self._fallback = SimpleTokenEstimator(fallback_chars_per_token)

        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding(model)
            logger.debug("TiktokenEstimator initialized with encoding: {}", model)
        except ImportError:
            logger.warning(
                "tiktoken not installed, falling back to simple estimation. "
                "Install with: pip install tiktoken"
            )
        except Exception as e:
            logger.warning("Failed to load tiktoken encoding '{}': {}", model, e)

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        """Estimate tokens using tiktoken or fallback."""
        if self._encoding is None:
            return self._fallback.estimate(messages)

        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += len(self._encoding.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total_tokens += len(self._encoding.encode(part["text"]))
            # Account for message overhead (role, formatting)
            total_tokens += 4  # Approximate overhead per message
        return total_tokens

    def estimate_text(self, text: str) -> int:
        """Estimate tokens for a single text string."""
        if self._encoding is None:
            return self._fallback.estimate_text(text)
        return len(self._encoding.encode(text))


def create_estimator(config: TokenConfig) -> TokenEstimator:
    """
    Factory function to create the appropriate token estimator.

    Args:
        config: Token configuration specifying the estimator type.

    Returns:
        A TokenEstimator instance.
    """
    if config.estimator == "tiktoken":
        return TiktokenEstimator(fallback_chars_per_token=config.chars_per_token)
    return SimpleTokenEstimator(chars_per_token=config.chars_per_token)


# Default estimator instance for backward compatibility
_default_estimator: TokenEstimator | None = None


def get_default_estimator() -> TokenEstimator:
    """Get or create the default token estimator."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = SimpleTokenEstimator()
    return _default_estimator


def set_default_estimator(estimator: TokenEstimator) -> None:
    """Set the default token estimator."""
    global _default_estimator
    _default_estimator = estimator
