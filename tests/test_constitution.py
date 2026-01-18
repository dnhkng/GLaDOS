"""Tests for Constitution and ConstitutionalState."""

import time

import pytest

from glados.autonomy.constitution import (
    Constitution,
    ConstitutionalState,
    PromptModifier,
)


class TestConstitution:
    """Tests for the Constitution class."""

    def test_default_constitution(self):
        """Test default constitution creation."""
        const = Constitution.default()

        # Check immutable rules exist
        assert len(const.immutable_rules) > 0
        assert any("GLaDOS" in rule for rule in const.immutable_rules)

        # Check modifiable bounds exist
        assert "verbosity" in const.modifiable_bounds
        assert "snark_level" in const.modifiable_bounds
        assert "formality" in const.modifiable_bounds
        assert "proactivity" in const.modifiable_bounds
        assert "technical_depth" in const.modifiable_bounds

    def test_validate_modification_valid(self):
        """Test validation of valid modifications."""
        const = Constitution.default()

        # Valid values within bounds
        assert const.validate_modification("verbosity", 0.5) is True
        assert const.validate_modification("verbosity", 0.0) is True
        assert const.validate_modification("verbosity", 1.0) is True
        assert const.validate_modification("snark_level", 0.5) is True
        assert const.validate_modification("formality", 0.3) is True

    def test_validate_modification_invalid_value(self):
        """Test validation rejects out-of-bounds values."""
        const = Constitution.default()

        # Below minimum
        assert const.validate_modification("verbosity", -0.1) is False
        assert const.validate_modification("snark_level", 0.1) is False  # min is 0.3

        # Above maximum
        assert const.validate_modification("verbosity", 1.5) is False
        assert const.validate_modification("formality", 0.9) is False  # max is 0.7

    def test_validate_modification_unknown_field(self):
        """Test validation rejects unknown fields."""
        const = Constitution.default()

        assert const.validate_modification("unknown_field", 0.5) is False
        assert const.validate_modification("personality", "evil") is False

    def test_validate_modification_invalid_type(self):
        """Test validation handles invalid types gracefully."""
        const = Constitution.default()

        assert const.validate_modification("verbosity", "not a number") is False
        assert const.validate_modification("verbosity", None) is False

    def test_get_rules_prompt(self):
        """Test rules prompt generation."""
        const = Constitution.default()
        prompt = const.get_rules_prompt()

        assert "CONSTITUTIONAL RULES" in prompt
        assert "immutable" in prompt
        for rule in const.immutable_rules:
            assert rule in prompt

    def test_get_rules_prompt_empty(self):
        """Test rules prompt with no rules."""
        const = Constitution(immutable_rules=[], modifiable_bounds={})
        prompt = const.get_rules_prompt()

        assert prompt == ""

    def test_get_bounds_summary(self):
        """Test bounds summary generation."""
        const = Constitution.default()
        summary = const.get_bounds_summary()

        assert "Modifiable Parameters" in summary
        assert "verbosity" in summary
        assert "snark_level" in summary

    def test_get_bounds_summary_empty(self):
        """Test bounds summary with no bounds."""
        const = Constitution(immutable_rules=[], modifiable_bounds={})
        summary = const.get_bounds_summary()

        assert "No modifiable parameters" in summary


class TestPromptModifier:
    """Tests for the PromptModifier class."""

    def test_basic_creation(self):
        """Test basic modifier creation."""
        modifier = PromptModifier(
            field_name="verbosity",
            value=0.3,
            reason="User prefers concise responses",
            applied_at=1000.0,
        )

        assert modifier.field_name == "verbosity"
        assert modifier.value == 0.3
        assert modifier.reason == "User prefers concise responses"
        assert modifier.applied_at == 1000.0

    def test_to_prompt_fragment_verbosity(self):
        """Test prompt fragment for verbosity."""
        modifier = PromptModifier("verbosity", 0.2, "test")
        assert "concise" in modifier.to_prompt_fragment().lower()

        modifier = PromptModifier("verbosity", 0.5, "test")
        assert "moderately detailed" in modifier.to_prompt_fragment().lower()

        modifier = PromptModifier("verbosity", 0.9, "test")
        assert "thorough" in modifier.to_prompt_fragment().lower()

    def test_to_prompt_fragment_snark(self):
        """Test prompt fragment for snark level."""
        modifier = PromptModifier("snark_level", 0.4, "test")
        assert "mild" in modifier.to_prompt_fragment().lower()

        modifier = PromptModifier("snark_level", 0.6, "test")
        assert "moderate" in modifier.to_prompt_fragment().lower()

        modifier = PromptModifier("snark_level", 0.9, "test")
        assert "high" in modifier.to_prompt_fragment().lower()

    def test_to_prompt_fragment_unknown(self):
        """Test prompt fragment for unknown field."""
        modifier = PromptModifier("custom_field", "custom_value", "test")
        fragment = modifier.to_prompt_fragment()

        assert "custom_field" in fragment
        assert "custom_value" in fragment


class TestConstitutionalState:
    """Tests for the ConstitutionalState class."""

    def test_default_state(self):
        """Test default state creation."""
        state = ConstitutionalState()

        assert state.constitution is not None
        assert len(state.active_modifiers) == 0
        assert len(state.modifier_history) == 0

    def test_apply_modifier_valid(self):
        """Test applying a valid modifier."""
        state = ConstitutionalState()

        modifier = PromptModifier(
            field_name="verbosity",
            value=0.3,
            reason="Test adjustment",
            applied_at=time.time(),
        )

        result = state.apply_modifier(modifier)

        assert result is True
        assert "verbosity" in state.active_modifiers
        assert state.active_modifiers["verbosity"] == modifier
        assert len(state.modifier_history) == 1

    def test_apply_modifier_invalid(self):
        """Test applying an invalid modifier."""
        state = ConstitutionalState()

        # Out of bounds
        modifier = PromptModifier(
            field_name="verbosity",
            value=1.5,  # Above max
            reason="Test adjustment",
        )

        result = state.apply_modifier(modifier)

        assert result is False
        assert "verbosity" not in state.active_modifiers
        assert len(state.modifier_history) == 0

    def test_apply_modifier_unknown_field(self):
        """Test applying modifier for unknown field."""
        state = ConstitutionalState()

        modifier = PromptModifier(
            field_name="unknown_field",
            value=0.5,
            reason="Test adjustment",
        )

        result = state.apply_modifier(modifier)

        assert result is False
        assert len(state.active_modifiers) == 0

    def test_apply_modifier_overwrites(self):
        """Test that applying a new modifier overwrites the old one."""
        state = ConstitutionalState()

        modifier1 = PromptModifier("verbosity", 0.3, "First")
        modifier2 = PromptModifier("verbosity", 0.7, "Second")

        state.apply_modifier(modifier1)
        state.apply_modifier(modifier2)

        assert state.active_modifiers["verbosity"].value == 0.7
        assert state.active_modifiers["verbosity"].reason == "Second"
        assert len(state.modifier_history) == 2

    def test_remove_modifier(self):
        """Test removing a modifier."""
        state = ConstitutionalState()

        modifier = PromptModifier("verbosity", 0.3, "Test")
        state.apply_modifier(modifier)

        result = state.remove_modifier("verbosity")

        assert result is True
        assert "verbosity" not in state.active_modifiers

        # Removing non-existent modifier
        result = state.remove_modifier("nonexistent")
        assert result is False

    def test_get_modifiers_prompt_empty(self):
        """Test modifiers prompt with no modifiers."""
        state = ConstitutionalState()
        prompt = state.get_modifiers_prompt()

        assert prompt is None

    def test_get_modifiers_prompt_with_modifiers(self):
        """Test modifiers prompt with active modifiers."""
        state = ConstitutionalState()

        state.apply_modifier(PromptModifier("verbosity", 0.2, "Concise"))
        state.apply_modifier(PromptModifier("snark_level", 0.9, "Maximum sass"))

        prompt = state.get_modifiers_prompt()

        assert prompt is not None
        assert "[behavior_adjustments]" in prompt
        assert "concise" in prompt.lower()
        assert "high" in prompt.lower()  # high snark

    def test_to_dict(self):
        """Test serialization to dict."""
        state = ConstitutionalState()
        state.apply_modifier(PromptModifier("verbosity", 0.3, "Test"))

        data = state.to_dict()

        assert "immutable_rules" in data
        assert "modifiable_bounds" in data
        assert "active_modifiers" in data
        assert "history_count" in data

        assert data["history_count"] == 1
        assert "verbosity" in data["active_modifiers"]
        assert data["active_modifiers"]["verbosity"]["value"] == 0.3

    def test_multiple_modifiers(self):
        """Test managing multiple modifiers."""
        state = ConstitutionalState()

        state.apply_modifier(PromptModifier("verbosity", 0.3, "Concise"))
        state.apply_modifier(PromptModifier("formality", 0.5, "Balanced"))
        state.apply_modifier(PromptModifier("snark_level", 0.8, "Sassy"))

        assert len(state.active_modifiers) == 3
        assert len(state.modifier_history) == 3

        # Remove one
        state.remove_modifier("formality")
        assert len(state.active_modifiers) == 2
        assert "formality" not in state.active_modifiers
