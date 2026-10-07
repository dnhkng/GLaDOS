"""Host and port overrides do not require or change the webapp enable flag."""

from pathlib import Path

import pytest
import yaml

from glados.core.engine import GladosConfig


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("GLADOS_WEBAPP_ENABLED", "GLADOS_WEBAPP_HOST", "GLADOS_WEBAPP_PORT"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "webapp.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "Glados": {
                    "llm_model": "test",
                    "completion_url": "http://localhost/v1/chat/completions",
                    "api_key": None,
                    "interruptible": True,
                    "audio_io": "sounddevice",
                    "asr_engine": "tdt",
                    "wake_word": None,
                    "voice": "glados",
                    "announcement": None,
                    "personality_preprompt": [],
                    "webapp": {"enabled": True, "host": "127.0.0.1", "port": 8050, "allowed_hosts": ["console.local"]},
                }
            }
        )
    )
    return path


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"GLADOS_WEBAPP_HOST": "192.168.1.20"},
        {"GLADOS_WEBAPP_PORT": "8088"},
        {"GLADOS_WEBAPP_HOST": "192.168.1.20", "GLADOS_WEBAPP_PORT": "8088"},
    ],
)
def test_host_port_overrides_preserve_yaml_enable_state(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, overrides: dict[str, str]
) -> None:
    data = yaml.safe_load(config_path.read_text())
    data["Glados"]["webapp"]["enabled"] = enabled
    config_path.write_text(yaml.safe_dump(data))
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    webapp = GladosConfig.from_yaml(config_path).webapp
    assert webapp.enabled is enabled
    assert webapp.host == overrides.get("GLADOS_WEBAPP_HOST", "127.0.0.1")
    assert webapp.port == int(overrides.get("GLADOS_WEBAPP_PORT", "8050"))
    assert webapp.allowed_hosts == ["console.local"]


def test_host_only_override_still_enforces_wildcard_allowlist(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = yaml.safe_load(config_path.read_text())
    data["Glados"]["webapp"]["allowed_hosts"] = []
    config_path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("GLADOS_WEBAPP_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="allowed_hosts"):
        GladosConfig.from_yaml(config_path)


@pytest.mark.parametrize("flag, enabled", [("0", False), ("1", True)])
def test_explicit_enable_flag_takes_precedence(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, enabled: bool
) -> None:
    monkeypatch.setenv("GLADOS_WEBAPP_ENABLED", flag)
    monkeypatch.setenv("GLADOS_WEBAPP_PORT", "8088")
    webapp = GladosConfig.from_yaml(config_path).webapp
    assert webapp.enabled is enabled
    assert webapp.port == 8088
