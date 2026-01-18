"""
Memory context for LLM prompt injection.

Reads facts from long-term memory and formats them for context injection.
Works alongside the memory MCP server - this reads, MCP server writes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# Same paths as memory_server.py
MEMORY_DIR = Path(os.path.expanduser("~/.glados/memory"))
FACTS_FILE = MEMORY_DIR / "facts.jsonl"


@dataclass
class MemoryConfig:
    """Configuration for memory context injection."""

    enabled: bool = True
    min_importance: float = 0.7  # Only inject facts above this importance
    max_facts: int = 10  # Maximum facts to inject
    include_source: bool = False  # Include source in prompt
    include_age: bool = True  # Include how old the fact is


class MemoryContext:
    """
    Reads facts from long-term memory for LLM context injection.

    Usage:
        memory = MemoryContext(config)
        context_builder.register("memory", memory.as_prompt, priority=7)
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig()

    @property
    def config(self) -> MemoryConfig:
        return self._config

    def load_facts(self) -> list[dict[str, Any]]:
        """Load all facts from storage."""
        if not FACTS_FILE.exists():
            return []

        facts = []
        try:
            with FACTS_FILE.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        facts.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"MemoryContext: failed to load facts: {e}")

        return facts

    def get_important_facts(self) -> list[dict[str, Any]]:
        """Get facts filtered by importance and limited by max_facts."""
        facts = self.load_facts()

        # Filter by importance
        filtered = [f for f in facts if f.get("importance", 0) >= self._config.min_importance]

        # Sort by importance (desc), then recency (desc)
        filtered.sort(key=lambda f: (f.get("importance", 0), f.get("created_at", 0)), reverse=True)

        # Limit
        return filtered[: self._config.max_facts]

    def format_fact(self, fact: dict[str, Any]) -> str:
        """Format a single fact for the prompt."""
        content = fact.get("content", "")
        parts = [f"- {content}"]

        if self._config.include_source:
            source = fact.get("source", "unknown")
            parts.append(f" (source: {source})")

        if self._config.include_age:
            created_at = fact.get("created_at")
            if created_at:
                age = self._format_age(created_at)
                parts.append(f" [{age}]")

        return "".join(parts)

    def _format_age(self, timestamp: float) -> str:
        """Format timestamp as human-readable age."""
        import time

        seconds = time.time() - timestamp
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m ago"
        if seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        days = int(seconds / 86400)
        if days == 1:
            return "yesterday"
        if days < 7:
            return f"{days}d ago"
        if days < 30:
            weeks = int(days / 7)
            return f"{weeks}w ago"
        return datetime.fromtimestamp(timestamp).strftime("%b %d")

    def as_prompt(self) -> str | None:
        """
        Format important facts as a prompt string for context injection.

        Returns None if no facts meet the importance threshold.
        """
        if not self._config.enabled:
            return None

        facts = self.get_important_facts()
        if not facts:
            return None

        lines = ["[memory] Important facts I remember:"]
        for fact in facts:
            lines.append(self.format_fact(fact))

        return "\n".join(lines)

    def preload_facts(self, facts: list[dict[str, Any]]) -> int:
        """
        Preload facts into memory storage.

        Useful for initializing memory with known information.

        Args:
            facts: List of dicts with 'content', 'source', 'importance' keys

        Returns:
            Number of facts saved
        """
        import time

        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        count = 0
        with FACTS_FILE.open("a") as f:
            for fact_data in facts:
                fact = {
                    "content": fact_data.get("content", ""),
                    "source": fact_data.get("source", "preload"),
                    "importance": fact_data.get("importance", 0.5),
                    "created_at": fact_data.get("created_at", time.time()),
                    "id": f"fact_{int(time.time() * 1000)}_{count}",
                }
                f.write(json.dumps(fact) + "\n")
                count += 1

        logger.info(f"MemoryContext: preloaded {count} facts")
        return count
