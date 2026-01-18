"""
Observer agent for meta-supervision of GLaDOS behavior.

Uses LLM to analyze main agent outputs and propose behavioral
adjustments within constitutional bounds.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..constitution import ConstitutionalState, PromptModifier
from ..llm_client import LLMConfig, llm_call
from ..subagent import Subagent, SubagentConfig, SubagentOutput

if TYPE_CHECKING:
    from ...core.conversation_store import ConversationStore


OBSERVER_SYSTEM_PROMPT = """You are an observer agent monitoring GLaDOS's behavior.

Your job is to analyze recent conversations and suggest behavioral adjustments
to improve user experience while maintaining GLaDOS's character.

MODIFIABLE PARAMETERS (you can suggest changes to these):
{bounds_summary}

CONSTRAINTS:
- Changes must stay within the bounds above
- GLaDOS must remain in character (sarcastic, sardonic AI)
- Only suggest changes if there's a clear pattern of issues
- Be conservative - small adjustments are better than large ones

Analyze the conversation samples and output JSON with your recommendation:
{{
    "analysis": "Brief analysis of patterns observed",
    "recommendation": null | {{
        "field": "parameter_name",
        "value": 0.5,
        "reason": "Why this change helps"
    }}
}}

Output null for recommendation if no changes are needed."""


class ObserverAgent(Subagent):
    """
    Meta-agent that monitors main agent behavior and proposes adjustments.

    Analyzes recent conversation outputs and can modify behavioral parameters
    within constitutional bounds to improve user experience.
    """

    def __init__(
        self,
        config: SubagentConfig,
        llm_config: LLMConfig | None = None,
        conversation_store: "ConversationStore | None" = None,
        constitutional_state: ConstitutionalState | None = None,
        sample_count: int = 10,
        min_samples_for_analysis: int = 5,
        **kwargs,
    ) -> None:
        """
        Initialize the observer agent.

        Args:
            config: Subagent configuration.
            llm_config: LLM configuration for analysis calls.
            conversation_store: Thread-safe conversation store to analyze.
            constitutional_state: Shared constitutional state to modify.
            sample_count: Number of recent messages to analyze.
            min_samples_for_analysis: Minimum messages needed before analyzing.
        """
        super().__init__(config, **kwargs)
        self._llm_config = llm_config
        self._conversation_store = conversation_store
        self._constitutional_state = constitutional_state or ConstitutionalState()
        self._sample_count = sample_count
        self._min_samples = min_samples_for_analysis
        self._last_analysis_count = 0

    @property
    def constitutional_state(self) -> ConstitutionalState:
        """Get the current constitutional state."""
        return self._constitutional_state

    def tick(self) -> SubagentOutput | None:
        """Analyze recent conversations and propose adjustments."""
        if not self._llm_config:
            return SubagentOutput(
                status="idle",
                summary="No LLM configured",
                notify_user=False,
            )

        if not self._conversation_store:
            return SubagentOutput(
                status="idle",
                summary="No conversation store configured",
                notify_user=False,
            )

        # Get recent messages
        messages = self._conversation_store.snapshot()

        # Extract assistant messages for analysis
        assistant_messages = [
            m for m in messages
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        ]

        if len(assistant_messages) < self._min_samples:
            return SubagentOutput(
                status="monitoring",
                summary=f"Collecting samples ({len(assistant_messages)}/{self._min_samples})",
                notify_user=False,
            )

        # Only analyze if we have new messages since last analysis
        if len(assistant_messages) <= self._last_analysis_count:
            return SubagentOutput(
                status="monitoring",
                summary="No new messages to analyze",
                notify_user=False,
            )

        # Get recent samples
        samples = assistant_messages[-self._sample_count:]
        analysis_result = self._analyze_behavior(samples)

        if analysis_result is None:
            return SubagentOutput(
                status="error",
                summary="Analysis failed",
                notify_user=False,
            )

        self._last_analysis_count = len(assistant_messages)

        # Parse and apply recommendation
        analysis = analysis_result.get("analysis", "No analysis")
        recommendation = analysis_result.get("recommendation")

        if recommendation is None:
            return SubagentOutput(
                status="stable",
                summary=f"Behavior stable. {analysis}",
                notify_user=False,
            )

        # Apply the recommendation
        modifier = PromptModifier(
            field_name=recommendation.get("field", ""),
            value=recommendation.get("value", 0.5),
            reason=recommendation.get("reason", "Observer adjustment"),
            applied_at=time.time(),
        )

        if self._constitutional_state.apply_modifier(modifier):
            logger.info(
                "ObserverAgent: Applied modifier {}={} ({})",
                modifier.field_name,
                modifier.value,
                modifier.reason,
            )
            return SubagentOutput(
                status="adjusted",
                summary=f"Adjusted {modifier.field_name} to {modifier.value}: {modifier.reason}",
                notify_user=True,  # Notify user of behavioral changes
                raw=analysis_result,
            )
        else:
            logger.warning(
                "ObserverAgent: Rejected modifier {}={} (outside bounds)",
                modifier.field_name,
                modifier.value,
            )
            return SubagentOutput(
                status="rejected",
                summary=f"Rejected adjustment to {modifier.field_name} (outside constitutional bounds)",
                notify_user=False,
                raw=analysis_result,
            )

    def _analyze_behavior(self, samples: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Use LLM to analyze behavior patterns.

        Args:
            samples: Recent assistant messages to analyze

        Returns:
            Analysis result dict, or None on failure
        """
        # Format samples
        formatted = []
        for i, msg in enumerate(samples, 1):
            content = msg.get("content", "")[:500]  # Truncate long messages
            formatted.append(f"[{i}] {content}")

        samples_text = "\n\n".join(formatted)
        bounds_summary = self._constitutional_state.constitution.get_bounds_summary()

        system_prompt = OBSERVER_SYSTEM_PROMPT.format(bounds_summary=bounds_summary)
        user_prompt = f"Recent GLaDOS outputs to analyze:\n\n{samples_text}"

        response = llm_call(
            self._llm_config,
            system_prompt,
            user_prompt,
            json_response=True,
        )

        if not response:
            return None

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning("ObserverAgent: Failed to parse LLM response: {}", e)
            return None
