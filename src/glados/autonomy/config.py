from pydantic import BaseModel


class AutonomyConfig(BaseModel):
    enabled: bool = False
    tick_interval_s: float = 10.0
    cooldown_s: float = 20.0
    system_prompt: str = (
        "You may receive autonomous state updates. "
        "Decide whether to do something or nothing. "
        "If you choose to speak, call the `speak` tool with your response. "
        "If no action is needed, call the `do_nothing` tool."
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
