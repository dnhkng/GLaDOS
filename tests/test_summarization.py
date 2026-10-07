"""Tests for summarization utilities."""


from glados.autonomy.summarization import (
    estimate_tokens,
)


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_messages(self):
        """Test with empty message list."""
        assert estimate_tokens([]) == 0

    def test_simple_string_content(self):
        """Test with simple string content."""
        messages = [
            {"role": "user", "content": "Hello world"},  # 11 chars
            {"role": "assistant", "content": "Hi there"},  # 8 chars
        ]
        # 19 chars / 4 = 4 tokens
        assert estimate_tokens(messages) == 4

    def test_multipart_content(self):
        """Test with multipart content (list format)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},  # 5 chars
                    {"type": "text", "text": "World"},  # 5 chars
                ]
            }
        ]
        # 10 chars / 4 = 2 tokens
        assert estimate_tokens(messages) == 2

    def test_empty_content(self):
        """Test with empty content."""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant"},  # Missing content
        ]
        assert estimate_tokens(messages) == 0

    def test_longer_content(self):
        """Test with longer content for accuracy."""
        # 400 chars should be ~100 tokens
        content = "x" * 400
        messages = [{"role": "user", "content": content}]
        assert estimate_tokens(messages) == 100
