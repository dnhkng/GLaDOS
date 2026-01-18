"""
Hacker News subagent for GLaDOS autonomy system.

Periodically fetches top stories from Hacker News and updates the slot
with notable changes.
"""

from __future__ import annotations

import httpx
from loguru import logger

from ..subagent import Subagent, SubagentConfig, SubagentOutput


class HackerNewsSubagent(Subagent):
    """
    Subagent that monitors Hacker News top stories.

    Fetches top stories at configured intervals and notifies when
    new high-scoring stories appear.
    """

    def __init__(
        self,
        config: SubagentConfig,
        top_n: int = 5,
        min_score: int = 200,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._top_n = top_n
        self._min_score = min_score
        self._last_ids: list[int] = []

    def tick(self) -> SubagentOutput | None:
        """Fetch and analyze top HN stories."""
        top_ids = self._fetch_top_ids()
        if not top_ids:
            return SubagentOutput(
                status="error",
                summary="HN fetch failed",
                notify_user=False,
            )

        top_ids = top_ids[: self._top_n]
        items = [self._fetch_item(story_id) for story_id in top_ids]
        items = [item for item in items if item]

        if not items:
            return SubagentOutput(
                status="error",
                summary="HN items unavailable",
                notify_user=False,
            )

        new_ids = [story_id for story_id in top_ids if story_id not in self._last_ids]
        new_items = [item for item in items if item["id"] in new_ids]
        eligible_items = [
            item for item in new_items if item.get("score", 0) >= self._min_score
        ]
        top_item = items[0]

        if not self._last_ids:
            summary = f"Top HN: {top_item['title']} ({top_item.get('score', 0)} points)"
            notify_user = False
            importance = 0.3
        elif eligible_items:
            titles = ", ".join(item["title"] for item in eligible_items[:3])
            summary = f"HN update: new in top {self._top_n}: {titles}"
            notify_user = True
            importance = min(1.0, 0.4 + 0.1 * len(eligible_items))
        else:
            summary = f"HN steady: {top_item['title']} stays on top"
            notify_user = False
            importance = 0.2

        self._last_ids = top_ids
        return SubagentOutput(
            status="done",
            summary=summary,
            notify_user=notify_user,
            importance=importance,
            confidence=0.7,
            next_run=self._config.loop_interval_s,
        )

    def _fetch_top_ids(self) -> list[int]:
        """Fetch top story IDs from HN API."""
        try:
            response = httpx.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json",
                timeout=8.0,
            )
            response.raise_for_status()
            return list(response.json())
        except Exception as exc:
            logger.warning("HackerNewsSubagent: failed to fetch top stories: %s", exc)
            return []

    def _fetch_item(self, story_id: int) -> dict[str, object] | None:
        """Fetch a single story item from HN API."""
        try:
            response = httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                timeout=8.0,
            )
            response.raise_for_status()
            data = response.json()
            if not data or "title" not in data:
                return None
            return {
                "id": data.get("id"),
                "title": data.get("title", "Unknown"),
                "score": data.get("score", 0),
            }
        except Exception as exc:
            logger.warning("HackerNewsSubagent: failed to fetch item %s: %s", story_id, exc)
            return None
