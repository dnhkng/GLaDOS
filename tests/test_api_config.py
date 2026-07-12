from pathlib import Path

import pytest

from glados.api.app import load_api_config
from glados.api.config import ApiConfig


def test_api_config_defaults_to_reuse_tts() -> None:
    config = ApiConfig()
    assert config.reuse_tts is True


def test_api_config_yaml_section(tmp_path: Path) -> None:
    config_file = tmp_path / "api_config.yaml"
    config_file.write_text("Api:\n  reuse_tts: false\n", encoding="utf-8")

    config = load_api_config(str(config_file))

    assert config.reuse_tts is False


def test_api_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLADOS_API_REUSE_TTS", "false")

    config = ApiConfig.model_validate({})

    assert config.reuse_tts is False


def test_load_api_config_uses_defaults_when_file_missing(tmp_path: Path) -> None:
    config = load_api_config(str(tmp_path / "missing.yaml"))

    assert config.reuse_tts is True
