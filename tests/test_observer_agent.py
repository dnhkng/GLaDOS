"""Tests for ObserverAgent."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from glados.autonomy.agents.observer_agent import ObserverAgent, OBSERVER_SYSTEM_PROMPT
from glados.autonomy.constitution import ConstitutionalState, PromptModifier
from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.subagent import SubagentConfig


def make_config() -> SubagentConfig:
    """Create a test subagent config."""
    return SubagentConfig(
        agent_id="test_observer",
        title="Test Observer",
        role="meta_supervision",
        loop_interval_s=60.0,
    )


def make_llm_config() -> LLMConfig:
    """Create a test LLM config."""
    return LLMConfig(
        url="http://localhost:11434/api/chat",
        model="test-model",
    )


class TestObserverAgentInit:
    """Tests for ObserverAgent initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        config = make_config()
        agent = ObserverAgent(config)

        assert agent.config == config
        assert agent._llm_config is None
        assert agent._conversation_history == []
        assert isinstance(agent._constitutional_state, ConstitutionalState)
        assert agent._sample_count == 10
        assert agent._min_samples == 5

    def test_init_with_llm_config(self):
        """Test initialization with LLM config."""
        config = make_config()
        llm_config = make_llm_config()
        agent = ObserverAgent(config, llm_config=llm_config)

        assert agent._llm_config == llm_config

    def test_init_with_shared_history(self):
        """Test initialization with shared conversation history."""
        config = make_config()
        history = [{"role": "user", "content": "Hello"}]
        lock = threading.Lock()
        state = ConstitutionalState()

        agent = ObserverAgent(
            config,
            conversation_history=history,
            conversation_lock=lock,
            constitutional_state=state,
        )

        assert agent._conversation_history is history
        assert agent._conversation_lock is lock
        assert agent._constitutional_state is state

    def test_init_custom_sample_settings(self):
        """Test initialization with custom sample settings."""
        config = make_config()
        agent = ObserverAgent(
            config,
            sample_count=20,
            min_samples_for_analysis=10,
        )

        assert agent._sample_count == 20
        assert agent._min_samples == 10


class TestObserverAgentTick:
    """Tests for ObserverAgent tick behavior."""

    def test_tick_no_llm_config(self):
        """Test tick returns idle when no LLM configured."""
        config = make_config()
        agent = ObserverAgent(config)

        result = agent.tick()

        assert result is not None
        assert result.status == "idle"
        assert "No LLM configured" in result.summary

    def test_tick_insufficient_samples(self):
        """Test tick returns monitoring when not enough samples."""
        config = make_config()
        llm_config = make_llm_config()
        history = [
            {"role": "assistant", "content": "Hello"},
            {"role": "assistant", "content": "How are you?"},
        ]

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            min_samples_for_analysis=5,
        )

        result = agent.tick()

        assert result is not None
        assert result.status == "monitoring"
        assert "Collecting samples" in result.summary
        assert "2/5" in result.summary

    def test_tick_no_new_messages(self):
        """Test tick skips analysis when no new messages."""
        config = make_config()
        llm_config = make_llm_config()
        history = [{"role": "assistant", "content": f"Message {i}"} for i in range(5)]

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            min_samples_for_analysis=5,
        )

        # First tick analyzes
        agent._last_analysis_count = 5

        result = agent.tick()

        assert result is not None
        assert result.status == "monitoring"
        assert "No new messages" in result.summary

    @patch("glados.autonomy.agents.observer_agent.llm_call")
    def test_tick_analysis_stable(self, mock_llm_call):
        """Test tick when analysis returns no recommendation."""
        config = make_config()
        llm_config = make_llm_config()
        history = [{"role": "assistant", "content": f"Message {i}"} for i in range(6)]

        mock_llm_call.return_value = '{"analysis": "Behavior looks good", "recommendation": null}'

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            min_samples_for_analysis=5,
        )

        result = agent.tick()

        assert result is not None
        assert result.status == "stable"
        assert "Behavior looks good" in result.summary

    @patch("glados.autonomy.agents.observer_agent.llm_call")
    def test_tick_analysis_adjusted(self, mock_llm_call):
        """Test tick when analysis returns valid recommendation."""
        config = make_config()
        llm_config = make_llm_config()
        history = [{"role": "assistant", "content": f"Message {i}"} for i in range(6)]
        state = ConstitutionalState()

        mock_llm_call.return_value = """{
            "analysis": "Responses too verbose",
            "recommendation": {
                "field": "verbosity",
                "value": 0.3,
                "reason": "User prefers concise responses"
            }
        }"""

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            constitutional_state=state,
            min_samples_for_analysis=5,
        )

        result = agent.tick()

        assert result is not None
        assert result.status == "adjusted"
        assert result.notify_user is True
        assert "verbosity" in result.summary

        # Check state was modified
        assert "verbosity" in state.active_modifiers
        assert state.active_modifiers["verbosity"].value == 0.3

    @patch("glados.autonomy.agents.observer_agent.llm_call")
    def test_tick_analysis_rejected(self, mock_llm_call):
        """Test tick when recommendation is outside bounds."""
        config = make_config()
        llm_config = make_llm_config()
        history = [{"role": "assistant", "content": f"Message {i}"} for i in range(6)]
        state = ConstitutionalState()

        mock_llm_call.return_value = """{
            "analysis": "Trying invalid adjustment",
            "recommendation": {
                "field": "verbosity",
                "value": 2.0,
                "reason": "This should be rejected"
            }
        }"""

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            constitutional_state=state,
            min_samples_for_analysis=5,
        )

        result = agent.tick()

        assert result is not None
        assert result.status == "rejected"
        assert "outside constitutional bounds" in result.summary

        # State should not be modified
        assert "verbosity" not in state.active_modifiers

    @patch("glados.autonomy.agents.observer_agent.llm_call")
    def test_tick_analysis_failed(self, mock_llm_call):
        """Test tick when LLM call fails."""
        config = make_config()
        llm_config = make_llm_config()
        history = [{"role": "assistant", "content": f"Message {i}"} for i in range(6)]

        mock_llm_call.return_value = None  # Simulates failure

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            min_samples_for_analysis=5,
        )

        result = agent.tick()

        assert result is not None
        assert result.status == "error"
        assert "Analysis failed" in result.summary

    @patch("glados.autonomy.agents.observer_agent.llm_call")
    def test_tick_analysis_invalid_json(self, mock_llm_call):
        """Test tick when LLM returns invalid JSON."""
        config = make_config()
        llm_config = make_llm_config()
        history = [{"role": "assistant", "content": f"Message {i}"} for i in range(6)]

        mock_llm_call.return_value = "not valid json {{"

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            min_samples_for_analysis=5,
        )

        result = agent.tick()

        assert result is not None
        assert result.status == "error"


class TestObserverAgentAnalyze:
    """Tests for ObserverAgent behavior analysis."""

    def test_analyze_filters_non_assistant_messages(self):
        """Test that analysis only uses assistant messages."""
        config = make_config()
        llm_config = make_llm_config()
        history = [
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant response 1"},
            {"role": "user", "content": "User message 2"},
            {"role": "assistant", "content": "Assistant response 2"},
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant response 3"},
            {"role": "assistant", "content": "Assistant response 4"},
            {"role": "assistant", "content": "Assistant response 5"},
        ]

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            min_samples_for_analysis=5,
        )

        # Should return monitoring because only 5 assistant messages
        result = agent.tick()

        assert result is not None
        assert result.status == "monitoring"

    def test_analyze_ignores_empty_content(self):
        """Test that empty assistant messages are ignored."""
        config = make_config()
        llm_config = make_llm_config()
        history = [
            {"role": "assistant", "content": "Valid message 1"},
            {"role": "assistant", "content": ""},  # Empty
            {"role": "assistant", "content": "   "},  # Whitespace only
            {"role": "assistant", "content": "Valid message 2"},
            {"role": "assistant", "content": "Valid message 3"},
        ]

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            min_samples_for_analysis=5,
        )

        # Only 3 valid messages
        result = agent.tick()

        assert result is not None
        assert result.status == "monitoring"
        assert "3/5" in result.summary

    def test_analyze_ignores_non_string_content(self):
        """Test that non-string content is ignored."""
        config = make_config()
        llm_config = make_llm_config()
        history = [
            {"role": "assistant", "content": "Valid message"},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": {"tool_calls": []}},
            {"role": "assistant", "content": ["list", "content"]},
        ]

        agent = ObserverAgent(
            config,
            llm_config=llm_config,
            conversation_history=history,
            min_samples_for_analysis=5,
        )

        result = agent.tick()

        assert result is not None
        assert result.status == "monitoring"
        assert "1/5" in result.summary


class TestObserverAgentConstitutionalState:
    """Tests for ObserverAgent constitutional state access."""

    def test_constitutional_state_property(self):
        """Test constitutional_state property."""
        config = make_config()
        state = ConstitutionalState()

        agent = ObserverAgent(config, constitutional_state=state)

        assert agent.constitutional_state is state

    def test_constitutional_state_default(self):
        """Test default constitutional state is created."""
        config = make_config()
        agent = ObserverAgent(config)

        assert agent.constitutional_state is not None
        assert isinstance(agent.constitutional_state, ConstitutionalState)


class TestObserverSystemPrompt:
    """Tests for the observer system prompt template."""

    def test_prompt_contains_bounds_placeholder(self):
        """Test that prompt contains bounds placeholder."""
        assert "{bounds_summary}" in OBSERVER_SYSTEM_PROMPT

    def test_prompt_contains_json_format(self):
        """Test that prompt describes JSON output format."""
        assert '"analysis"' in OBSERVER_SYSTEM_PROMPT
        assert '"recommendation"' in OBSERVER_SYSTEM_PROMPT
        assert '"field"' in OBSERVER_SYSTEM_PROMPT
        assert '"value"' in OBSERVER_SYSTEM_PROMPT
        assert '"reason"' in OBSERVER_SYSTEM_PROMPT

    def test_prompt_mentions_modifiable_parameters(self):
        """Test that prompt mentions modifiable parameters."""
        assert "MODIFIABLE PARAMETERS" in OBSERVER_SYSTEM_PROMPT

    def test_prompt_mentions_constraints(self):
        """Test that prompt mentions constraints."""
        assert "CONSTRAINTS" in OBSERVER_SYSTEM_PROMPT
        assert "GLaDOS" in OBSERVER_SYSTEM_PROMPT
