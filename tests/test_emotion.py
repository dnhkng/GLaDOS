"""Tests for the emotion system."""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

# Load modules directly to avoid the full autonomy __init__ import chain
_src_path = Path(__file__).parent.parent / "src"


def _load_module(name: str, path: Path):
    """Load a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load config first
_config_module = _load_module(
    "glados.autonomy.config",
    _src_path / "glados" / "autonomy" / "config.py",
)
EmotionConfig = _config_module.EmotionConfig
HEXACOConfig = _config_module.HEXACOConfig

# Load emotion_state
_emotion_state_module = _load_module(
    "glados.autonomy.emotion_state",
    _src_path / "glados" / "autonomy" / "emotion_state.py",
)
EmotionState = _emotion_state_module.EmotionState
EmotionEvent = _emotion_state_module.EmotionEvent

# Load constitution
_constitution_module = _load_module(
    "glados.autonomy.constitution",
    _src_path / "glados" / "autonomy" / "constitution.py",
)
Constitution = _constitution_module.Constitution
ConstitutionalState = _constitution_module.ConstitutionalState
PromptModifier = _constitution_module.PromptModifier
EmotionConstitutionBridge = _constitution_module.EmotionConstitutionBridge


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
        assert config.tick_interval_s == 30.0
        assert config.max_events == 20
        assert config.baseline_dominance == 0.6  # GLaDOS feels in control

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


class TestEmotionConstitutionBridge:
    """Tests for EmotionConstitutionBridge."""

    def test_low_pleasure_increases_snark(self) -> None:
        """Test that low pleasure increases snark level."""
        bridge = EmotionConstitutionBridge()
        emotion = EmotionState(pleasure=-0.5, arousal=0.0, dominance=0.0)
        constitution = Constitution.default()

        modifiers = bridge.compute_modifiers(emotion, constitution)

        snark_mods = [m for m in modifiers if m.field_name == "snark_level"]
        assert len(snark_mods) == 1
        assert snark_mods[0].value > bridge.default_snark

    def test_high_arousal_increases_proactivity(self) -> None:
        """Test that high arousal increases proactivity."""
        bridge = EmotionConstitutionBridge()
        emotion = EmotionState(pleasure=0.0, arousal=0.5, dominance=0.0)
        constitution = Constitution.default()

        modifiers = bridge.compute_modifiers(emotion, constitution)

        proactive_mods = [m for m in modifiers if m.field_name == "proactivity"]
        assert len(proactive_mods) == 1
        assert proactive_mods[0].value > bridge.default_proactivity

    def test_low_dominance_decreases_verbosity(self) -> None:
        """Test that low dominance decreases verbosity."""
        bridge = EmotionConstitutionBridge()
        emotion = EmotionState(pleasure=0.0, arousal=0.0, dominance=-0.5)
        constitution = Constitution.default()

        modifiers = bridge.compute_modifiers(emotion, constitution)

        verbosity_mods = [m for m in modifiers if m.field_name == "verbosity"]
        assert len(verbosity_mods) == 1
        assert verbosity_mods[0].value < bridge.default_verbosity

    def test_neutral_emotion_no_modifiers(self) -> None:
        """Test that neutral emotion produces no modifiers."""
        bridge = EmotionConstitutionBridge()
        emotion = EmotionState(pleasure=0.0, arousal=0.0, dominance=0.0)
        constitution = Constitution.default()

        modifiers = bridge.compute_modifiers(emotion, constitution)

        assert len(modifiers) == 0

    def test_apply_emotion_modifiers(self) -> None:
        """Test applying modifiers to constitutional state."""
        bridge = EmotionConstitutionBridge()
        emotion = EmotionState(pleasure=-0.5, arousal=0.5, dominance=-0.5)
        state = ConstitutionalState()

        applied = bridge.apply_emotion_modifiers(emotion, state)

        # Should have applied snark, proactivity, and verbosity
        assert "snark_level" in applied
        assert "proactivity" in applied
        assert "verbosity" in applied

        # Check modifiers are in state
        assert "snark_level" in state.active_modifiers
        assert "proactivity" in state.active_modifiers
        assert "verbosity" in state.active_modifiers

    def test_modifiers_respect_bounds(self) -> None:
        """Test that modifiers stay within constitutional bounds."""
        bridge = EmotionConstitutionBridge()
        # Extreme emotion
        emotion = EmotionState(pleasure=-1.0, arousal=1.0, dominance=-1.0)
        constitution = Constitution.default()

        modifiers = bridge.compute_modifiers(emotion, constitution)

        for modifier in modifiers:
            min_val, max_val = constitution.modifiable_bounds[modifier.field_name]
            assert min_val <= modifier.value <= max_val
