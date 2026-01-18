"""
Long-term memory MCP server for GLaDOS.

Provides tools for storing and retrieving facts and summaries.
Uses LLM for semantic search ranking (LLM-first principle).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.server.fastmcp import FastMCP

logger.remove()
logging.getLogger().setLevel(logging.CRITICAL)

mcp = FastMCP("memory")

# Storage paths
MEMORY_DIR = Path(os.path.expanduser("~/.glados/memory"))
FACTS_FILE = MEMORY_DIR / "facts.jsonl"
SUMMARIES_FILE = MEMORY_DIR / "summaries.jsonl"


@dataclass
class Fact:
    """A stored fact with metadata."""

    content: str
    source: str
    importance: float
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"fact_{int(time.time() * 1000)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        return cls(**data)


@dataclass
class Summary:
    """A stored conversation summary."""

    content: str
    period: str  # "daily", "weekly", "session"
    start_time: str  # ISO format
    end_time: str  # ISO format
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"summary_{int(time.time() * 1000)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Summary":
        return cls(**data)


def _ensure_storage() -> None:
    """Ensure storage directory exists."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _load_facts() -> list[Fact]:
    """Load all facts from storage."""
    if not FACTS_FILE.exists():
        return []
    facts = []
    try:
        with FACTS_FILE.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    facts.append(Fact.from_dict(json.loads(line)))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load facts: {e}")
    return facts


def _save_fact(fact: Fact) -> None:
    """Append a fact to storage."""
    _ensure_storage()
    with FACTS_FILE.open("a") as f:
        f.write(json.dumps(fact.to_dict()) + "\n")


def _load_summaries() -> list[Summary]:
    """Load all summaries from storage."""
    if not SUMMARIES_FILE.exists():
        return []
    summaries = []
    try:
        with SUMMARIES_FILE.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    summaries.append(Summary.from_dict(json.loads(line)))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load summaries: {e}")
    return summaries


def _save_summary(summary: Summary) -> None:
    """Append a summary to storage."""
    _ensure_storage()
    with SUMMARIES_FILE.open("a") as f:
        f.write(json.dumps(summary.to_dict()) + "\n")


@mcp.tool()
def store_fact(fact: str, source: str = "user", importance: float = 0.5) -> str:
    """
    Store a fact in long-term memory.

    Args:
        fact: The fact to store (e.g., "User's name is David")
        source: Where the fact came from ("user", "conversation", "system")
        importance: How important the fact is (0.0 to 1.0)

    Returns:
        Confirmation message with fact ID
    """
    importance = max(0.0, min(1.0, importance))
    new_fact = Fact(content=fact, source=source, importance=importance)
    _save_fact(new_fact)
    return json.dumps({
        "status": "stored",
        "id": new_fact.id,
        "fact": fact,
    })


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """
    Search long-term memory for relevant facts.

    Uses simple keyword matching. For semantic search, the main agent
    should interpret results in context.

    Args:
        query: What to search for
        limit: Maximum number of results (default 5)

    Returns:
        JSON array of matching facts, sorted by relevance
    """
    facts = _load_facts()
    if not facts:
        return json.dumps({"facts": [], "message": "No facts stored yet"})

    # Simple keyword-based scoring (LLM in main agent handles semantics)
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored = []
    for fact in facts:
        content_lower = fact.content.lower()
        # Score: word overlap + importance boost + recency
        word_score = sum(1 for w in query_words if w in content_lower)
        importance_boost = fact.importance * 0.5
        recency_boost = min(0.3, (time.time() - fact.created_at) / (86400 * 30))  # Decay over 30 days
        total_score = word_score + importance_boost - recency_boost

        if word_score > 0 or query_lower in content_lower:
            scored.append((total_score, fact))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, fact in scored[:limit]:
        results.append({
            "id": fact.id,
            "content": fact.content,
            "source": fact.source,
            "importance": fact.importance,
            "created_at": datetime.fromtimestamp(fact.created_at).isoformat(),
        })

    return json.dumps({"facts": results, "total_stored": len(facts)})


@mcp.tool()
def list_facts(limit: int = 20, min_importance: float = 0.0) -> str:
    """
    List stored facts, optionally filtered by importance.

    Args:
        limit: Maximum number of facts to return
        min_importance: Minimum importance threshold (0.0 to 1.0)

    Returns:
        JSON array of facts sorted by importance then recency
    """
    facts = _load_facts()
    filtered = [f for f in facts if f.importance >= min_importance]

    # Sort by importance (desc), then recency (desc)
    filtered.sort(key=lambda f: (f.importance, f.created_at), reverse=True)

    results = []
    for fact in filtered[:limit]:
        results.append({
            "id": fact.id,
            "content": fact.content,
            "source": fact.source,
            "importance": fact.importance,
            "created_at": datetime.fromtimestamp(fact.created_at).isoformat(),
        })

    return json.dumps({"facts": results, "total_stored": len(facts)})


@mcp.tool()
def store_summary(summary: str, period: str, start_time: str, end_time: str) -> str:
    """
    Store a conversation summary.

    Args:
        summary: The summary text
        period: Summary period ("session", "daily", "weekly")
        start_time: Start of summarized period (ISO format)
        end_time: End of summarized period (ISO format)

    Returns:
        Confirmation message with summary ID
    """
    if period not in ("session", "daily", "weekly"):
        return json.dumps({"error": "period must be 'session', 'daily', or 'weekly'"})

    new_summary = Summary(
        content=summary,
        period=period,
        start_time=start_time,
        end_time=end_time,
    )
    _save_summary(new_summary)
    return json.dumps({
        "status": "stored",
        "id": new_summary.id,
        "period": period,
    })


@mcp.tool()
def get_summaries(period: str = "all", limit: int = 5) -> str:
    """
    Retrieve stored conversation summaries.

    Args:
        period: Filter by period ("session", "daily", "weekly", or "all")
        limit: Maximum number of summaries to return

    Returns:
        JSON array of summaries, most recent first
    """
    summaries = _load_summaries()

    if period != "all":
        summaries = [s for s in summaries if s.period == period]

    # Sort by created_at descending (most recent first)
    summaries.sort(key=lambda s: s.created_at, reverse=True)

    results = []
    for summary in summaries[:limit]:
        results.append({
            "id": summary.id,
            "content": summary.content,
            "period": summary.period,
            "start_time": summary.start_time,
            "end_time": summary.end_time,
            "created_at": datetime.fromtimestamp(summary.created_at).isoformat(),
        })

    return json.dumps({"summaries": results, "total_stored": len(_load_summaries())})


@mcp.tool()
def memory_stats() -> str:
    """
    Get statistics about stored memories.

    Returns:
        JSON object with memory statistics
    """
    facts = _load_facts()
    summaries = _load_summaries()

    source_counts: dict[str, int] = {}
    for fact in facts:
        source_counts[fact.source] = source_counts.get(fact.source, 0) + 1

    period_counts: dict[str, int] = {}
    for summary in summaries:
        period_counts[summary.period] = period_counts.get(summary.period, 0) + 1

    avg_importance = sum(f.importance for f in facts) / len(facts) if facts else 0

    return json.dumps({
        "total_facts": len(facts),
        "total_summaries": len(summaries),
        "facts_by_source": source_counts,
        "summaries_by_period": period_counts,
        "average_importance": round(avg_importance, 2),
    })


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
