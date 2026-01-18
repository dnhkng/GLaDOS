"""
Conversation compaction agent.

Monitors conversation length and compacts older messages when
approaching token limits. Uses LLM to summarize and extract facts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..llm_client import LLMConfig
from ..subagent import Subagent, SubagentConfig, SubagentOutput
from ..summarization import estimate_tokens, extract_facts, summarize_messages
from ...mcp.memory_server import Fact, _save_fact

if TYPE_CHECKING:
    from ...core.conversation_store import ConversationStore


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
        conversation_store: "ConversationStore | None" = None,
        token_threshold: int = 8000,
        preserve_recent: int = 10,
        **kwargs,
    ) -> None:
        """
        Initialize the compaction agent.

        Args:
            config: Subagent configuration.
            llm_config: LLM configuration for summarization calls.
            conversation_store: Thread-safe conversation store to compact.
            token_threshold: Start compacting when tokens exceed this.
            preserve_recent: Number of recent messages to keep uncompacted.
        """
        super().__init__(config, **kwargs)
        self._llm_config = llm_config
        self._conversation_store = conversation_store
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

        if not self._conversation_store:
            return SubagentOutput(
                status="idle",
                summary="No conversation store configured",
                notify_user=False,
            )

        messages = self._conversation_store.snapshot()

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

        # Apply changes to conversation history atomically
        # Get fresh snapshot for building new history to minimize race window
        current_messages = self._conversation_store.snapshot()
        new_history = []
        summary_inserted = False

        for i, msg in enumerate(current_messages):
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

        # Atomically replace entire history
        self._conversation_store.replace_all(new_history)

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

        # Store facts in long-term memory
        if facts:
            logger.info("CompactionAgent: storing {} facts in long-term memory", len(facts))
            for fact_text in facts:
                fact = Fact(
                    content=fact_text,
                    source="conversation",
                    importance=0.6,  # Medium importance for auto-extracted facts
                )
                _save_fact(fact)

        return result
