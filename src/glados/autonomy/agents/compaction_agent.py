"""
Conversation compaction agent.

Monitors conversation length and compacts older messages when
approaching token limits. Uses LLM to summarize and extract facts.
"""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger

from ..llm_client import LLMConfig
from ..subagent import Subagent, SubagentConfig, SubagentOutput
from ..summarization import estimate_tokens, extract_facts, summarize_messages


class CompactionAgent(Subagent):
    """
    Monitors conversation and compacts when token count gets high.

    Preserves recent messages while summarizing older ones.
    Extracted facts can be stored in long-term memory.
    """

    def __init__(
        self,
        config: SubagentConfig,
        llm_config: LLMConfig | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        conversation_lock: threading.Lock | None = None,
        token_threshold: int = 8000,
        preserve_recent: int = 10,
        **kwargs,
    ) -> None:
        """
        Initialize the compaction agent.

        Args:
            config: Subagent configuration.
            llm_config: LLM configuration for summarization calls.
            conversation_history: Shared conversation history to compact.
            conversation_lock: Lock for thread-safe history access.
            token_threshold: Start compacting when tokens exceed this.
            preserve_recent: Number of recent messages to keep uncompacted.
        """
        super().__init__(config, **kwargs)
        self._llm_config = llm_config
        self._conversation_history = conversation_history or []
        self._conversation_lock = conversation_lock or threading.Lock()
        self._token_threshold = token_threshold
        self._preserve_recent = preserve_recent
        self._last_compaction_size = 0

    def tick(self) -> SubagentOutput | None:
        """Check conversation size and compact if needed."""
        if not self._llm_config:
            return SubagentOutput(
                status="idle",
                summary="No LLM configured",
                notify_user=False,
            )

        with self._conversation_lock:
            messages = list(self._conversation_history)

        token_count = estimate_tokens(messages)

        # Check if we need to compact
        if token_count < self._token_threshold:
            return SubagentOutput(
                status="monitoring",
                summary=f"Context at {token_count} tokens (threshold: {self._token_threshold})",
                notify_user=False,
            )

        # Find compactable messages (exclude system and recent)
        compactable_indices = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            # Skip system messages and preserved recent messages
            if role == "system":
                continue
            if i >= len(messages) - self._preserve_recent:
                continue
            # Skip already-compacted summaries
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("[summary]"):
                continue
            compactable_indices.append(i)

        if len(compactable_indices) < 3:
            # Not enough to compact
            return SubagentOutput(
                status="monitoring",
                summary=f"At {token_count} tokens but not enough compactable messages",
                notify_user=False,
            )

        # Get the messages to compact (oldest half of compactable)
        half = max(3, len(compactable_indices) // 2)
        indices_to_compact = compactable_indices[:half]
        messages_to_compact = [messages[i] for i in indices_to_compact]

        logger.info(
            "CompactionAgent: compacting {} messages ({} tokens)",
            len(messages_to_compact),
            estimate_tokens(messages_to_compact),
        )

        # Summarize and extract facts
        summary = summarize_messages(messages_to_compact, self._llm_config)
        facts = extract_facts(messages_to_compact, self._llm_config)

        if not summary:
            return SubagentOutput(
                status="error",
                summary="Failed to generate summary",
                notify_user=False,
            )

        # Build the replacement summary message
        summary_content = f"[summary] Previous conversation summary: {summary}"

        # Apply changes to conversation history
        with self._conversation_lock:
            # Create new history without compacted messages
            new_history = []
            summary_inserted = False

            for i, msg in enumerate(self._conversation_history):
                if i in indices_to_compact:
                    # Insert summary at first compacted position
                    if not summary_inserted:
                        new_history.append({
                            "role": "system",
                            "content": summary_content,
                        })
                        summary_inserted = True
                    # Skip the compacted message
                    continue
                new_history.append(msg)

            # Replace history in-place
            self._conversation_history.clear()
            self._conversation_history.extend(new_history)

        new_token_count = estimate_tokens(new_history)
        self._last_compaction_size = len(indices_to_compact)

        result = SubagentOutput(
            status="compacted",
            summary=f"Compacted {len(indices_to_compact)} messages: {token_count} -> {new_token_count} tokens",
            notify_user=False,
            raw={
                "compacted_count": len(indices_to_compact),
                "facts_extracted": len(facts),
                "tokens_before": token_count,
                "tokens_after": new_token_count,
            },
        )

        # Store facts if we have them (will be handled by memory system in Stage 5)
        if facts:
            logger.info("CompactionAgent: extracted {} facts", len(facts))
            for fact in facts:
                self.memory.set(f"fact_{hash(fact) % 10000}", fact)

        return result
