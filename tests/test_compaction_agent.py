"""Tests for CompactionAgent."""

import threading
from unittest.mock import patch, MagicMock

import pytest

from glados.autonomy.agents.compaction_agent import CompactionAgent
from glados.autonomy.subagent import SubagentConfig
from glados.autonomy.llm_client import LLMConfig


@pytest.fixture
def agent_config():
    """Create a basic subagent config."""
    return SubagentConfig(
        agent_id="test_compaction",
        title="Test Compaction",
        role="context_management",
        loop_interval_s=60.0,
    )


@pytest.fixture
def llm_config():
    """Create a basic LLM config."""
    return LLMConfig(url="http://localhost:11434/v1/chat/completions")


class TestCompactionAgent:
    """Tests for the CompactionAgent class."""

    def test_tick_without_llm_config(self, agent_config):
        """Test that tick returns idle status without LLM config."""
        agent = CompactionAgent(config=agent_config, llm_config=None)
        result = agent.tick()

        assert result is not None
        assert result.status == "idle"
        assert "No LLM" in result.summary

    def test_tick_below_threshold(self, agent_config, llm_config):
        """Test that tick returns monitoring status when below threshold."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            token_threshold=10000,
        )

        result = agent.tick()

        assert result is not None
        assert result.status == "monitoring"
        assert "threshold" in result.summary.lower()

    def test_tick_preserves_system_messages(self, agent_config, llm_config):
        """Test that system messages are never compacted."""
        messages = [
            {"role": "system", "content": "You are GLaDOS."},
            {"role": "user", "content": "x" * 10000},  # Large to trigger compaction
            {"role": "assistant", "content": "y" * 10000},
        ]

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            token_threshold=100,  # Low threshold
            preserve_recent=0,  # Don't preserve recent
        )

        with patch("glados.autonomy.agents.compaction_agent.summarize_messages") as mock_summarize:
            mock_summarize.return_value = "Summary of conversation"

            with patch("glados.autonomy.agents.compaction_agent.extract_facts") as mock_extract:
                mock_extract.return_value = []

                result = agent.tick()

                # System message should not be in the messages to compact
                if mock_summarize.called:
                    compacted = mock_summarize.call_args[0][0]
                    for msg in compacted:
                        assert msg.get("role") != "system"

    def test_tick_preserves_recent_messages(self, agent_config, llm_config):
        """Test that recent messages are preserved."""
        messages = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
            {"role": "user", "content": "Recent 1"},
            {"role": "assistant", "content": "Recent 2"},
        ]

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            token_threshold=100,
            preserve_recent=2,
        )

        with patch("glados.autonomy.agents.compaction_agent.summarize_messages") as mock_summarize:
            mock_summarize.return_value = "Summary"

            with patch("glados.autonomy.agents.compaction_agent.extract_facts") as mock_extract:
                mock_extract.return_value = []

                agent.tick()

                # After compaction, recent messages should still be there
                assert any("Recent 1" in str(m.get("content", "")) for m in messages)
                assert any("Recent 2" in str(m.get("content", "")) for m in messages)

    def test_tick_not_enough_to_compact(self, agent_config, llm_config):
        """Test when there aren't enough messages to compact."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "x" * 10000},
        ]

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            token_threshold=100,
            preserve_recent=1,
        )

        result = agent.tick()

        # Only 1 compactable message (user), which is < 3 required
        assert result.status == "monitoring"

    @patch("glados.autonomy.agents.compaction_agent.summarize_messages")
    @patch("glados.autonomy.agents.compaction_agent.extract_facts")
    def test_successful_compaction(self, mock_extract, mock_summarize, agent_config, llm_config):
        """Test successful compaction replaces messages with summary."""
        mock_summarize.return_value = "User asked about project setup."
        mock_extract.return_value = ["User prefers Python"]

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "x" * 3000},
            {"role": "assistant", "content": "y" * 3000},
            {"role": "user", "content": "z" * 3000},
            {"role": "assistant", "content": "w" * 3000},
            {"role": "user", "content": "Recent message"},
        ]
        original_count = len(messages)

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            token_threshold=100,
            preserve_recent=1,
        )

        result = agent.tick()

        assert result.status == "compacted"
        assert "compacted" in result.summary.lower()

        # Should have fewer messages now
        assert len(messages) < original_count

        # Should have a summary message
        summary_msgs = [m for m in messages if "[summary]" in str(m.get("content", ""))]
        assert len(summary_msgs) >= 1

        # Recent message should be preserved
        assert any("Recent message" in str(m.get("content", "")) for m in messages)

    @patch("glados.autonomy.agents.compaction_agent.summarize_messages")
    def test_summarization_failure(self, mock_summarize, agent_config, llm_config):
        """Test handling of summarization failure."""
        mock_summarize.return_value = None

        messages = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
            {"role": "user", "content": "z" * 5000},
            {"role": "assistant", "content": "w" * 5000},
        ]
        original_count = len(messages)

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            token_threshold=100,
            preserve_recent=0,
        )

        result = agent.tick()

        assert result.status == "error"
        assert len(messages) == original_count  # No changes

    def test_skips_already_compacted(self, agent_config, llm_config):
        """Test that already compacted summaries are not re-compacted."""
        messages = [
            {"role": "system", "content": "[summary] Previous summary"},
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
            {"role": "user", "content": "z" * 5000},
            {"role": "assistant", "content": "w" * 5000},
        ]

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            token_threshold=100,
            preserve_recent=1,
        )

        with patch("glados.autonomy.agents.compaction_agent.summarize_messages") as mock_summarize:
            mock_summarize.return_value = "New summary"

            with patch("glados.autonomy.agents.compaction_agent.extract_facts") as mock_extract:
                mock_extract.return_value = []

                agent.tick()

                # The [summary] message should not be in the compacted set
                if mock_summarize.called:
                    compacted = mock_summarize.call_args[0][0]
                    for msg in compacted:
                        content = str(msg.get("content", ""))
                        assert not content.startswith("[summary]")

    def test_thread_safety(self, agent_config, llm_config):
        """Test that compaction is thread-safe."""
        messages = []
        lock = threading.Lock()

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            conversation_lock=lock,
            token_threshold=10000,
        )

        errors = []

        def add_messages():
            try:
                for i in range(50):
                    with lock:
                        messages.append({"role": "user", "content": f"Message {i}"})
            except Exception as e:
                errors.append(e)

        def tick_agent():
            try:
                for _ in range(10):
                    agent.tick()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_messages),
            threading.Thread(target=tick_agent),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    @patch("glados.autonomy.agents.compaction_agent.summarize_messages")
    @patch("glados.autonomy.agents.compaction_agent.extract_facts")
    def test_raw_output_contains_stats(self, mock_extract, mock_summarize, agent_config, llm_config):
        """Test that raw output contains compaction statistics."""
        mock_summarize.return_value = "Summary"
        mock_extract.return_value = ["Fact 1", "Fact 2"]

        messages = [
            {"role": "user", "content": "x" * 3000},
            {"role": "assistant", "content": "y" * 3000},
            {"role": "user", "content": "z" * 3000},
            {"role": "assistant", "content": "w" * 3000},
        ]

        agent = CompactionAgent(
            config=agent_config,
            llm_config=llm_config,
            conversation_history=messages,
            token_threshold=100,
            preserve_recent=0,
        )

        result = agent.tick()

        assert result.raw is not None
        assert "compacted_count" in result.raw
        assert "facts_extracted" in result.raw
        assert result.raw["facts_extracted"] == 2
        assert "tokens_before" in result.raw
        assert "tokens_after" in result.raw
