"""Decision schemas are reusable instructions, separate from changing task data."""

from unittest.mock import MagicMock

import pytest

from glados.autonomy.llm_client import LLMConfig
from glados.core.llm_decision import UrgencyDecision, llm_decide_sync


def test_decision_schema_stays_in_identical_system_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"notify_user":false,"importance":0.1,"reason":"Routine weather"}',
                }
            }
        ]
    }
    monkeypatch.setattr("glados.core.llm_decision.httpx.Client", lambda **kwargs: client)
    for weather in ("Clear", "Heavy rain"):
        result = llm_decide_sync(
            "Evaluate {weather}",
            {"weather": weather},
            UrgencyDecision,
            LLMConfig(url="http://test/v1/chat/completions"),
            "Evaluate weather alerts.",
        )
        assert not result.notify_user
    first, second = [call.kwargs["json"]["messages"] for call in client.post.call_args_list]
    assert first[0] == second[0]
    assert "notify_user" in first[0]["content"]
    assert first[-1]["content"] == "Evaluate Clear"
    assert second[-1]["content"] == "Evaluate Heavy rain"
