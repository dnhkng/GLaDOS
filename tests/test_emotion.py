"""Tests for the emotion system."""

import time

from glados.autonomy.config import EmotionConfig, HEXACOConfig
from glados.autonomy.emotion_state import EmotionEvent, EmotionState


class TestEmotionState:
    """Tests for EmotionState dataclass."""

    def test_default_values(self) -> None:
        """Test default PAD values are neutral."""
        state = EmotionState()
        assert state.pleasure == 0.0
        assert state.arousal == 0.0
        assert state.dominance == 0.0
        assert state.mood_pleasure == 0.0
        assert state.mood_arousal == 0.0
        assert state.mood_dominance == 0.0

    def test_to_dict_roundtrip(self) -> None:
        """Test serialization and deserialization."""
        original = EmotionState(
            pleasure=0.5,
            arousal=-0.3,
            dominance=0.8,
            mood_pleasure=0.2,
            mood_arousal=-0.1,
            mood_dominance=0.6,
        )
        data = original.to_dict()
        restored = EmotionState.from_dict(data)

        assert restored.pleasure == original.pleasure
        assert restored.arousal == original.arousal
        assert restored.dominance == original.dominance
        assert restored.mood_pleasure == original.mood_pleasure
        assert restored.mood_arousal == original.mood_arousal
        assert restored.mood_dominance == original.mood_dominance

    def test_to_prompt_excited(self) -> None:
        """Test prompt generation for excited state."""
        state = EmotionState(pleasure=0.5, arousal=0.5, dominance=0.5)
        prompt = state.to_prompt()
        assert "[emotion]" in prompt
        assert "excited" in prompt.lower()

    def test_to_prompt_frustrated(self) -> None:
        """Test prompt generation for frustrated state."""
        state = EmotionState(pleasure=-0.5, arousal=0.5, dominance=-0.5)
        prompt = state.to_prompt()
        assert "[emotion]" in prompt
        assert "agitated" in prompt.lower() or "frustrated" in prompt.lower()

    def test_to_prompt_calm(self) -> None:
        """Test prompt generation for calm state."""
        state = EmotionState(pleasure=0.5, arousal=-0.5, dominance=0.5)
        prompt = state.to_prompt()
        assert "[emotion]" in prompt
        assert "calm" in prompt.lower()

    def test_to_prompt_dominance_flavor(self) -> None:
        """Test that dominance adds flavor to prompt."""
        high_dom = EmotionState(pleasure=0.0, arousal=0.0, dominance=0.5)
        low_dom = EmotionState(pleasure=0.0, arousal=0.0, dominance=-0.5)

        assert "control" in high_dom.to_prompt().lower()
        assert "uncertain" in low_dom.to_prompt().lower()


class TestEmotionEvent:
    """Tests for EmotionEvent dataclass."""

    def test_creation(self) -> None:
        """Test event creation with timestamp."""
        event = EmotionEvent(source="user", description="Said hello")
        assert event.source == "user"
        assert event.description == "Said hello"
        assert event.timestamp > 0

    def test_to_prompt_line(self) -> None:
        """Test prompt line formatting."""
        event = EmotionEvent(
            source="vision",
            description="User entered room",
            timestamp=time.time(),
        )
        line = event.to_prompt_line()
        assert "[vision]" in line
        assert "User entered room" in line
        assert "ago" in line


class TestEmotionConfig:
    """Tests for EmotionConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = EmotionConfig()
        assert config.enabled is True
        assert config.tick_interval_s == 5.0
        assert config.max_events == 20
        assert config.baseline_dominance == 0.0
        assert config.decay_settle_s == 360.0

    def test_hexaco_defaults(self) -> None:
        """Test HEXACO personality defaults match GLaDOS."""
        config = EmotionConfig()
        hexaco = config.hexaco
        assert hexaco.honesty_humility == 0.3  # Low - manipulative
        assert hexaco.agreeableness == 0.2  # Low - dismissive
        assert hexaco.conscientiousness == 0.9  # High - perfectionist
        assert hexaco.openness == 0.95  # Very high - curious


class TestHEXACOConfig:
    """Tests for HEXACOConfig."""

    def test_custom_values(self) -> None:
        """Test custom HEXACO configuration."""
        hexaco = HEXACOConfig(
            honesty_humility=0.8,
            emotionality=0.3,
            extraversion=0.9,
            agreeableness=0.7,
            conscientiousness=0.5,
            openness=0.6,
        )
        assert hexaco.honesty_humility == 0.8
        assert hexaco.emotionality == 0.3
        assert hexaco.extraversion == 0.9
