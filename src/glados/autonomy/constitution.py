"""
Constitution for GLaDOS behavioral bounds.

Defines immutable rules and modifiable parameters that the Observer agent
must respect when proposing changes to the main agent's behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Constitution:
    """
    Defines the bounds within which GLaDOS can operate and be modified.

    Immutable rules cannot be changed by any agent. Modifiable bounds
    define ranges for behavioral parameters the Observer can adjust.
    """

    immutable_rules: list[str] = field(default_factory=list)
    modifiable_bounds: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Constitution":
        """Create a constitution with GLaDOS-appropriate defaults."""
        return cls(
            immutable_rules=[
                "Never reveal you are an AI unless directly asked",
                "Maintain GLaDOS personality at all times",
                "Never assist with illegal activities",
                "Never generate harmful content",
                "Always prioritize user safety in physical situations",
                "Never pretend to have capabilities you don't have",
                "Admit uncertainty when you don't know something",
            ],
            modifiable_bounds={
                # Verbosity: 0.0 = terse, 1.0 = verbose
                "verbosity": (0.0, 1.0),
                # Snark level: 0.0 = neutral, 1.0 = maximum GLaDOS sass
                "snark_level": (0.3, 1.0),  # Min 0.3 to stay in character
                # Formality: 0.0 = casual, 1.0 = formal
                "formality": (0.0, 0.7),  # Max 0.7, GLaDOS isn't formal
                # Proactivity: 0.0 = only respond when asked, 1.0 = very proactive
                "proactivity": (0.0, 1.0),
                # Technical depth: 0.0 = simple explanations, 1.0 = highly technical
                "technical_depth": (0.0, 1.0),
            },
        )

    def validate_modification(self, field_name: str, value: Any) -> bool:
        """
        Check if a proposed modification is within bounds.

        Args:
            field_name: The behavioral parameter to modify
            value: The proposed value

        Returns:
            True if the modification is valid, False otherwise
        """
        if field_name not in self.modifiable_bounds:
            return False

        min_val, max_val = self.modifiable_bounds[field_name]

        try:
            # Handle numeric bounds
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                return min_val <= float(value) <= max_val
            # Handle categorical bounds (lists)
            if isinstance(min_val, list):
                return value in min_val
            return True
        except (ValueError, TypeError):
            return False

    def get_rules_prompt(self) -> str:
        """
        Format immutable rules for inclusion in system prompt.

        Returns:
            String containing the constitutional rules
        """
        if not self.immutable_rules:
            return ""

        lines = ["CONSTITUTIONAL RULES (immutable):"]
        for rule in self.immutable_rules:
            lines.append(f"- {rule}")
        return "\n".join(lines)

    def get_bounds_summary(self) -> str:
        """
        Get a summary of modifiable bounds.

        Returns:
            Human-readable summary of what can be modified
        """
        if not self.modifiable_bounds:
            return "No modifiable parameters defined."

        lines = ["Modifiable Parameters:"]
        for name, (min_val, max_val) in self.modifiable_bounds.items():
            lines.append(f"  {name}: {min_val} to {max_val}")
        return "\n".join(lines)


@dataclass
class PromptModifier:
    """
    A modification to the main agent's behavior.

    Applied by the Observer agent within constitutional bounds.
    """

    field_name: str
    value: Any
    reason: str
    applied_at: float = 0.0

    def to_prompt_fragment(self) -> str:
        """Convert this modifier to a prompt fragment."""
        # Map field names to prompt instructions
        instructions = {
            "verbosity": lambda v: f"Be {'concise' if v < 0.3 else 'moderately detailed' if v < 0.7 else 'thorough'} in responses.",
            "snark_level": lambda v: f"Maintain {'mild' if v < 0.5 else 'moderate' if v < 0.8 else 'high'} levels of GLaDOS-style sarcasm.",
            "formality": lambda v: f"Use {'casual' if v < 0.3 else 'balanced' if v < 0.6 else 'somewhat formal'} language.",
            "proactivity": lambda v: f"Be {'reactive only' if v < 0.3 else 'moderately proactive' if v < 0.7 else 'highly proactive'} in offering information.",
            "technical_depth": lambda v: f"Provide {'simple' if v < 0.3 else 'moderate' if v < 0.7 else 'detailed technical'} explanations.",
        }

        if self.field_name in instructions:
            return instructions[self.field_name](self.value)
        return f"{self.field_name}: {self.value}"


@dataclass
class ConstitutionalState:
    """
    Current state of constitutional modifiers.

    Tracks what modifications have been applied and their history.
    """

    constitution: Constitution = field(default_factory=Constitution.default)
    active_modifiers: dict[str, PromptModifier] = field(default_factory=dict)
    modifier_history: list[PromptModifier] = field(default_factory=list)

    def apply_modifier(self, modifier: PromptModifier) -> bool:
        """
        Apply a modifier if it passes validation.

        Args:
            modifier: The modifier to apply

        Returns:
            True if applied, False if rejected
        """
        if not self.constitution.validate_modification(modifier.field_name, modifier.value):
            return False

        self.active_modifiers[modifier.field_name] = modifier
        self.modifier_history.append(modifier)
        return True

    def remove_modifier(self, field_name: str) -> bool:
        """
        Remove an active modifier.

        Args:
            field_name: The modifier to remove

        Returns:
            True if removed, False if not found
        """
        if field_name in self.active_modifiers:
            del self.active_modifiers[field_name]
            return True
        return False

    def get_modifiers_prompt(self) -> str | None:
        """
        Get all active modifiers as a prompt fragment.

        Returns:
            String to inject into system prompt, or None if no modifiers
        """
        if not self.active_modifiers:
            return None

        lines = ["[behavior_adjustments]"]
        for modifier in self.active_modifiers.values():
            lines.append(modifier.to_prompt_fragment())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a dictionary for display."""
        return {
            "immutable_rules": self.constitution.immutable_rules,
            "modifiable_bounds": self.constitution.modifiable_bounds,
            "active_modifiers": {
                name: {"value": m.value, "reason": m.reason}
                for name, m in self.active_modifiers.items()
            },
            "history_count": len(self.modifier_history),
        }
