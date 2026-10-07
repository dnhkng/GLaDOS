"""The lightweight profile still has working emotional regulation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from glados.autonomy.config import AutonomyConfig
from glados.autonomy.llm_client import LLMConfig, llm_call
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados
from glados.vision.vision_config import VisionConfig


@pytest.mark.parametrize("autonomy,jobs,emotion", [(False, True, True), (True, False, True), (False, True, False)])
@pytest.mark.parametrize("vision_enabled", [None, False, True])
def test_emotion_registration_is_independent(
    monkeypatch: pytest.MonkeyPatch, autonomy: bool, jobs: bool, emotion: bool, vision_enabled: bool | None
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
        api_key="conversation-key",
        llm_model="conversation-model",
        llm_request_options={"chat_template_kwargs": {"enable_thinking": False}},
        autonomy_slots=Mock(),
        mind_registry=Mock(),
        observability_bus=Mock(),
        shutdown_event=Mock(),
        quiet_event=Mock(is_set=Mock(return_value=False)),
        autonomy_loop=None,
        _emotion_agent=None,
        vision_state=None,
        vision_config=VisionConfig(
            enabled=vision_enabled, completion_url="http://vision/v1/chat/completions", api_key="vision-key"
        )
        if vision_enabled is not None
        else None,
        _conversation_store=ConversationStore(),
    )
    Glados._register_subagents(engine)
    assert manager.register.call_count == int(emotion) + 1
    memory_llm = engine.compaction_agent._llm_config
    assert memory_llm.model == ("gemma-4-E4B" if vision_enabled else engine.llm_model)
    assert memory_llm.url == (engine.vision_config.completion_url if vision_enabled else engine.completion_url)
    assert memory_llm.api_key == ("vision-key" if vision_enabled else engine.api_key)
    assert memory_llm.owner == "Compaction"
    assert memory_llm.scheduler is engine.inference_scheduler
    assert memory_llm.shutdown_event is engine.shutdown_event
    assert memory_llm.cancelled() is False
    if not vision_enabled:
        assert memory_llm.request_options == engine.llm_request_options
    assert engine.compaction_agent.snapshot()["preserve_recent"] == 8
    if emotion:
        assert engine._emotion_agent is factory.return_value
        settings = factory.call_args.kwargs
        assert settings["llm_config"].model == engine.llm_model
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
