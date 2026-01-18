"""Tests for summarization utilities."""

from unittest.mock import patch, MagicMock

import pytest

from glados.autonomy.summarization import (
    estimate_tokens,
    summarize_messages,
    extract_facts,
)
from glados.autonomy.llm_client import LLMConfig


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


class TestSummarizeMessages:
    """Tests for message summarization."""

    def test_empty_messages(self):
        """Test with empty message list."""
        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        result = summarize_messages([], config)
        assert result is None

    def test_no_content_messages(self):
        """Test with messages that have no content."""
        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "   "},
        ]
        result = summarize_messages(messages, config)
        assert result is None

    @patch("glados.autonomy.summarization.llm_call")
    def test_successful_summarization(self, mock_llm_call):
        """Test successful summarization."""
        mock_llm_call.return_value = "User discussed project setup and debugging."

        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [
            {"role": "user", "content": "How do I set up the project?"},
            {"role": "assistant", "content": "Run npm install first."},
            {"role": "user", "content": "I'm getting an error."},
        ]

        result = summarize_messages(messages, config)

        assert result == "User discussed project setup and debugging."
        mock_llm_call.assert_called_once()
        call_args = mock_llm_call.call_args
        assert "summarize" in call_args[0][1].lower()  # system prompt
        assert "npm install" in call_args[0][2]  # user prompt contains conversation

    @patch("glados.autonomy.summarization.llm_call")
    def test_llm_failure(self, mock_llm_call):
        """Test handling of LLM failure."""
        mock_llm_call.return_value = None

        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [{"role": "user", "content": "Hello"}]

        result = summarize_messages(messages, config)
        assert result is None

    @patch("glados.autonomy.summarization.llm_call")
    def test_multipart_content_handling(self, mock_llm_call):
        """Test that multipart content is handled correctly."""
        mock_llm_call.return_value = "Summary"

        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ]
            }
        ]

        result = summarize_messages(messages, config)
        assert result == "Summary"

        # Check that the content was properly extracted
        call_args = mock_llm_call.call_args
        assert "Hello" in call_args[0][2] or "World" in call_args[0][2]


class TestExtractFacts:
    """Tests for fact extraction."""

    def test_empty_messages(self):
        """Test with empty message list."""
        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        result = extract_facts([], config)
        assert result == []

    def test_no_content_messages(self):
        """Test with messages that have no content."""
        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [{"role": "user", "content": ""}]
        result = extract_facts(messages, config)
        assert result == []

    @patch("glados.autonomy.summarization.llm_call")
    def test_successful_extraction(self, mock_llm_call):
        """Test successful fact extraction."""
        mock_llm_call.return_value = """- User's name is David
- User prefers dark mode
- Project uses Python"""

        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [
            {"role": "user", "content": "I'm David, I prefer dark mode."},
            {"role": "assistant", "content": "Nice to meet you, David!"},
        ]

        result = extract_facts(messages, config)

        assert len(result) == 3
        assert "User's name is David" in result
        assert "User prefers dark mode" in result
        assert "Project uses Python" in result

    @patch("glados.autonomy.summarization.llm_call")
    def test_extraction_with_asterisk_bullets(self, mock_llm_call):
        """Test extraction with asterisk-style bullets."""
        mock_llm_call.return_value = """* Fact one
* Fact two"""

        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [{"role": "user", "content": "Some content"}]

        result = extract_facts(messages, config)
        assert result == ["Fact one", "Fact two"]

    @patch("glados.autonomy.summarization.llm_call")
    def test_extraction_filters_empty_lines(self, mock_llm_call):
        """Test that empty lines are filtered out."""
        mock_llm_call.return_value = """Fact one

Fact two

"""

        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [{"role": "user", "content": "Some content"}]

        result = extract_facts(messages, config)
        assert result == ["Fact one", "Fact two"]

    @patch("glados.autonomy.summarization.llm_call")
    def test_llm_failure(self, mock_llm_call):
        """Test handling of LLM failure."""
        mock_llm_call.return_value = None

        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [{"role": "user", "content": "Hello"}]

        result = extract_facts(messages, config)
        assert result == []

    @patch("glados.autonomy.summarization.llm_call")
    def test_extraction_skips_comments(self, mock_llm_call):
        """Test that lines starting with # are skipped."""
        mock_llm_call.return_value = """# This is a comment
Actual fact
# Another comment"""

        config = LLMConfig(url="http://localhost:11434/v1/chat/completions")
        messages = [{"role": "user", "content": "Some content"}]

        result = extract_facts(messages, config)
        assert result == ["Actual fact"]
