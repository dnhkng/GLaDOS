"""The lightweight profile still has working emotional regulation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from glados.autonomy.config import AutonomyConfig
from glados.autonomy.llm_client import LLMConfig, llm_call
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados


@pytest.mark.parametrize("autonomy,jobs,emotion", [(False, True, True), (True, False, True), (False, True, False)])
def test_emotion_registration_is_independent(
    monkeypatch: pytest.MonkeyPatch, autonomy: bool, jobs: bool, emotion: bool
) -> None:
    factory = Mock(return_value=SimpleNamespace(agent_id="emotion"))
    monkeypatch.setattr("glados.core.engine.EmotionAgent", factory)
    manager = Mock()
    config = AutonomyConfig(enabled=autonomy)
    config.jobs.enabled = jobs
    config.emotion.enabled = emotion
    engine = SimpleNamespace(
        subagent_manager=manager,
        inference_scheduler=None,
        autonomy_config=config,
        completion_url="http://localhost/v1/chat/completions",
        api_key=None,
        llm_model="gemma-4-E4B",
        llm_request_options={"chat_template_kwargs": {"enable_thinking": False}},
        autonomy_slots=Mock(),
        mind_registry=Mock(),
        observability_bus=Mock(),
        shutdown_event=Mock(),
        quiet_event=Mock(is_set=Mock(return_value=False)),
        autonomy_loop=None,
        _emotion_agent=None,
        vision_state=None,
        vision_config=None,
        _conversation_store=ConversationStore(),
    )
    Glados._register_subagents(engine)
    assert manager.register.call_count == int(emotion) + 1
    assert engine.compaction_agent.model == "gemma-4-E4B"
    assert engine.compaction_agent.snapshot()["preserve_recent"] == 8
    if emotion:
        assert engine._emotion_agent is factory.return_value
        settings = factory.call_args.kwargs
        assert settings["llm_config"].model == "gemma-4-E4B"
        assert settings["config"].loop_interval_s == 5.0
        assert settings["llm_config"].request_options == engine.llm_request_options
    else:
        factory.assert_not_called()
        assert engine._emotion_agent is None


def test_background_model_calls_keep_thinking_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": '{"pleasure": 0.1}'}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr("glados.autonomy.llm_client.requests.post", post)
    config = LLMConfig(url="http://localhost/v1/chat/completions", request_options={"think": False})
    assert llm_call(config, "Persona", "Update mood", json_response=True) == '{"pleasure": 0.1}'
    assert post.call_args.kwargs["json"]["think"] is False
    assert post.call_args.kwargs["json"]["response_format"] == {"type": "json_object"}
