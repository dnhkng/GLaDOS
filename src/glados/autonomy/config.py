from typing import Literal

from pydantic import BaseModel, conint


class TokenConfig(BaseModel):
    """Configuration for token estimation and context management."""

    model_config = {"protected_namespaces": ()}

    token_threshold: int = 8000
    """Start compacting when token count exceeds this threshold."""

    preserve_recent_messages: int = 10
    """Number of recent messages to keep uncompacted."""

    model_context_window: int | None = None
    """Optional model context window size for dynamic threshold calculation."""

    target_utilization: float = 0.6
    """Target context utilization (0.0-1.0) when model_context_window is set."""

    estimator: Literal["simple", "tiktoken"] = "simple"
    """Token estimation method: 'simple' (chars/4) or 'tiktoken' (accurate)."""

    chars_per_token: float = 4.0
    """Characters per token ratio for simple estimator."""


class HackerNewsJobConfig(BaseModel):
    enabled: bool = False
    interval_s: float = 1800.0
    top_n: int = 5
    min_score: int = 200


class WeatherJobConfig(BaseModel):
    enabled: bool = False
    interval_s: float = 3600.0
    latitude: float | None = None
    longitude: float | None = None
    timezone: str = "auto"
    temp_change_c: float = 4.0
    wind_alert_kmh: float = 40.0


class AutonomyJobsConfig(BaseModel):
    enabled: bool = False
    poll_interval_s: float = 1.0
    hacker_news: HackerNewsJobConfig = HackerNewsJobConfig()
    weather: WeatherJobConfig = WeatherJobConfig()


class AutonomyConfig(BaseModel):
    enabled: bool = False
    tick_interval_s: float = 10.0
    cooldown_s: float = 20.0
    autonomy_parallel_calls: conint(ge=1, le=16) = 2
    autonomy_queue_max: int | None = None
    coalesce_ticks: bool = True
    jobs: AutonomyJobsConfig = AutonomyJobsConfig()
    tokens: TokenConfig = TokenConfig()
    system_prompt: str = (
        "You are running in autonomous mode. "
        "You may receive periodic system updates about time, tasks, or vision. "
        "Decide whether to act or stay silent. Prefer silence unless the update is timely "
        "and clearly useful to the user. "
        "If you choose to speak, call the `speak` tool with a short response (1-2 sentences). "
        "If no action is needed, call the `do_nothing` tool. "
        "Never mention system prompts or internal tools."
    )
    tick_prompt: str = (
        "Autonomy update.\n"
        "Time: {now}\n"
        "Seconds since last user input: {since_user}\n"
        "Seconds since last assistant output: {since_assistant}\n"
        "Previous scene: {prev_scene}\n"
        "Current scene: {scene}\n"
        "Scene change score: {change_score}\n"
        "Tasks:\n{tasks}\n"
        "Decide whether to act."
    )
