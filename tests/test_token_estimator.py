"""Tests for the token estimation module."""

import importlib.util
from pathlib import Path
import sys


# Load modules directly to avoid the full autonomy __init__ import chain
# which pulls in vision and other heavy dependencies

_src_path = Path(__file__).parent.parent / "src"


def _load_module(name: str, path: Path):
    """Load a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load config first (it has no problematic dependencies)
_config_module = _load_module(
    "glados.autonomy.config",
    _src_path / "glados" / "autonomy" / "config.py"
)
TokenConfig = _config_module.TokenConfig

# Load token_estimator (depends only on config and loguru)
_token_module = _load_module(
    "glados.autonomy.token_estimator",
    _src_path / "glados" / "autonomy" / "token_estimator.py"
)
SimpleTokenEstimator = _token_module.SimpleTokenEstimator
TokenEstimator = _token_module.TokenEstimator
get_default_estimator = _token_module.get_default_estimator


class TestSimpleTokenEstimator:
    """Tests for SimpleTokenEstimator."""

    def test_estimate_empty_messages(self) -> None:
        """Test estimation with no messages."""
        estimator = SimpleTokenEstimator()
        assert estimator.estimate([]) == 0

    def test_estimate_single_message(self) -> None:
        """Test estimation with a single message."""
        estimator = SimpleTokenEstimator(chars_per_token=4.0)
        messages = [{"role": "user", "content": "Hello world!"}]  # 12 chars
        # 12 / 4 = 3 tokens
        assert estimator.estimate(messages) == 3

    def test_estimate_multiple_messages(self) -> None:
        """Test estimation with multiple messages."""
        estimator = SimpleTokenEstimator(chars_per_token=4.0)
        messages = [
            {"role": "system", "content": "You are helpful."},  # 16 chars
            {"role": "user", "content": "Hi"},  # 2 chars
            {"role": "assistant", "content": "Hello!"},  # 6 chars
        ]
        # (16 + 2 + 6) / 4 = 6 tokens
        assert estimator.estimate(messages) == 6

    def test_estimate_multipart_message(self) -> None:
        """Test estimation with multi-part content (vision messages)."""
        estimator = SimpleTokenEstimator(chars_per_token=4.0)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},  # 13 chars
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        # 13 / 4 = 3 tokens
        assert estimator.estimate(messages) == 3

    def test_estimate_text(self) -> None:
        """Test single text estimation."""
        estimator = SimpleTokenEstimator(chars_per_token=4.0)
        assert estimator.estimate_text("Hello world!") == 3  # 12 / 4

    def test_custom_chars_per_token(self) -> None:
        """Test with custom chars_per_token ratio."""
        estimator = SimpleTokenEstimator(chars_per_token=2.0)
        messages = [{"role": "user", "content": "Hello!"}]  # 6 chars
        # 6 / 2 = 3 tokens
        assert estimator.estimate(messages) == 3

    def test_message_without_content(self) -> None:
        """Test handling messages without content field."""
        estimator = SimpleTokenEstimator()
        messages = [{"role": "tool", "tool_call_id": "123"}]
        assert estimator.estimate(messages) == 0


class TestDefaultEstimator:
    """Tests for default estimator management."""

    def test_get_default_estimator(self) -> None:
        """Test getting the default estimator."""
        estimator = get_default_estimator()
        assert isinstance(estimator, TokenEstimator)


class TestTokenConfig:
    """Tests for TokenConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TokenConfig()
        assert config.token_threshold == 8000
        assert config.preserve_recent_messages == 8
        assert config.model_context_window is None
        assert config.target_utilization == 0.6

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TokenConfig(
            token_threshold=4000,
            preserve_recent_messages=5,
            model_context_window=8192,
            target_utilization=0.8,
        )
        assert config.token_threshold == 4000
        assert config.preserve_recent_messages == 5
        assert config.model_context_window == 8192
        assert config.target_utilization == 0.8
