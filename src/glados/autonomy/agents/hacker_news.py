"""
Hacker News subagent for GLaDOS autonomy system.

Monitors top stories and reports new ones the user hasn't heard about.
Uses persistent memory to track what's been shown.
"""

from __future__ import annotations

import httpx
from loguru import logger

from ...core.llm_decision import LLMConfig, LLMDecisionError, RelevanceDecision, llm_decide_sync
from ..subagent import Subagent, SubagentConfig, SubagentOutput


class HackerNewsSubagent(Subagent):
    """
    Monitors Hacker News top stories.

    Stores stories in memory, only reports unshown ones above score threshold.
    """

    def __init__(
        self,
        config: SubagentConfig,
        top_n: int = 5,
        min_score: int = 200,
        llm_config: LLMConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._top_n = top_n
        self._min_score = min_score
        self._llm_config = llm_config

    def tick(self) -> SubagentOutput | None:
        """Fetch top stories, store new ones, report unshown."""
        stories = self._fetch_top_stories()
        if not stories:
            return SubagentOutput(
                status="error",
                summary="HN fetch failed",
                notify_user=False,
            )

        # Store all fetched stories in memory
        for story in stories:
            key = f"hn_{story['id']}"
            if key not in self.memory:
                self.memory.set(key, story)

        # Find unshown stories - use LLM to evaluate relevance if available
        unshown = []
        for entry in self.memory.list_unshown():
            story = entry.value
            if not isinstance(story, dict):
                continue

            if self._llm_config:
                try:
                    decision = llm_decide_sync(
                        prompt="Is this Hacker News story relevant and interesting? {story}",
                        context={
                            "story": f"Title: {story.get('title', 'Unknown')}, Score: {story.get('score', 0)}",
                        },
                        schema=RelevanceDecision,
                        config=self._llm_config,
                        system_prompt=(
                            "Evaluate Hacker News stories for a tech enthusiast. "
                            "Consider: technical depth, novelty, broad appeal. "
                            "Set importance 0.0-1.0. Be concise in your summary."
                        ),
                    )
                    if decision.relevant:
                        story["_importance"] = decision.importance
                        story["_summary"] = decision.summary
                        unshown.append(story)
                except LLMDecisionError as e:
                    logger.warning("HN: LLM decision failed, using fallback: %s", e)
                    if story.get("score", 0) >= self._min_score:
                        unshown.append(story)
            else:
                # No LLM config - use score threshold
                if story.get("score", 0) >= self._min_score:
                    unshown.append(story)

        if not unshown:
            top = stories[0]
            return SubagentOutput(
                status="idle",
                summary=f"HN quiet. Top: {top['title']}",
                notify_user=False,
                importance=0.2,
            )

        # Report top unshown stories (sorted by importance if available)
        unshown.sort(key=lambda s: s.get("_importance", 0.5), reverse=True)
        to_report = unshown[: self._top_n]
        titles = [s["title"] for s in to_report]
        summary = f"HN: {', '.join(titles)}"

        # Use average LLM importance or fallback to formula
        if any("_importance" in s for s in to_report):
            avg_importance = sum(s.get("_importance", 0.5) for s in to_report) / len(to_report)
            importance = min(1.0, avg_importance + 0.1 * (len(to_report) - 1))
        else:
            importance = min(1.0, 0.4 + 0.1 * len(to_report))

        return SubagentOutput(
            status="update",
            summary=summary,
            notify_user=True,
            importance=importance,
            confidence=0.8,
            raw={"stories": to_report},
        )

    def _fetch_top_stories(self) -> list[dict]:
        """Fetch top N stories from HN API."""
        try:
            resp = httpx.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json",
                timeout=8.0,
            )
            resp.raise_for_status()
            top_ids = resp.json()[: self._top_n * 2]  # Fetch extra in case some fail
        except Exception as exc:
            logger.warning("HN: failed to fetch top stories: %s", exc)
            return []

        stories = []
        for story_id in top_ids:
            if len(stories) >= self._top_n:
                break
            story = self._fetch_story(story_id)
            if story:
                stories.append(story)

        return stories

    def _fetch_story(self, story_id: int) -> dict | None:
        """Fetch a single story."""
        try:
            resp = httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                timeout=8.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data or "title" not in data:
                return None
            return {
                "id": data["id"],
                "title": data["title"],
                "score": data.get("score", 0),
                "url": data.get("url", ""),
            }
        except Exception as exc:
            logger.debug("HN: failed to fetch story %d: %s", story_id, exc)
            return None
