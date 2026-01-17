from pydantic import BaseModel, conint


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
