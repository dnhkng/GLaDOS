"""Human-editable YAML overrides and validated migration preserve existing console settings."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.slots import TaskSlotStore
from glados.core.decision_lists import DecisionListStore
from glados.core.operator_state import OperatorState
from glados.core.search_preferences import SearchPreferences, SearchSources
from glados.core.settings_files import read_settings, settings_source, write_settings
from glados.core.store import Store
from glados.vision.vision_config import VisionConfig
from glados.vision.vision_mind import VisionMind
from glados.vision.vision_state import VisionState


def test_search_preferences_migrate_json_and_roundtrip_yaml(tmp_path: Path) -> None:
    path = tmp_path / "search_settings.yaml"
    values = {"news": ["news.ycombinator.com", "huggingnews.com"], "weather": ["dwd.de"], "reddit": [], "general": []}
    legacy = path.with_suffix(".json")
    original = json.dumps(values)
    legacy.write_text(original)
    preferences = SearchPreferences(SearchSources(), path)
    assert preferences.snapshot() == yaml.safe_load(path.read_text()) == values
    assert legacy.read_text() == original
    assert "- huggingnews.com" in path.read_text()
    preferences.update({**values, "reddit": ["reddit.com/r/LocalLLaMA"]})
    assert SearchPreferences(SearchSources(), path).snapshot()["reddit"] == ["reddit.com/r/LocalLLaMA"]
    assert legacy.read_text() == original


def test_hand_edited_yaml_takes_precedence_over_legacy_json(tmp_path: Path) -> None:
    path = tmp_path / "search_settings.yaml"
    path.with_suffix(".json").write_text(SearchSources().model_dump_json())
    path.write_text("# Favorite pages\nnews:\n  - huggingnews.com\nweather: []\nreddit: []\ngeneral: []\n")
    assert settings_source(path) == path
    assert SearchPreferences(SearchSources(), path).snapshot()["news"] == ["huggingnews.com"]


def test_invalid_yaml_is_preserved_for_correction(tmp_path: Path) -> None:
    path = tmp_path / "search_settings.yaml"
    text = "news: [unfinished"
    path.write_text(text)
    with pytest.raises(ValueError, match="YAML"):
        read_settings(path)
    assert SearchPreferences(SearchSources(), path).snapshot() == SearchSources().model_dump()
    assert path.read_text() == text


def test_default_yaml_exists_and_json_compatibility_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "search_settings.yaml"
    SearchPreferences(SearchSources(), path)
    assert yaml.safe_load(path.read_text())["weather"] == ["dwd.de", "meteoblue.com"]
    legacy = tmp_path / "old.json"
    write_settings(legacy, {"enabled": True})
    assert json.loads(legacy.read_text()) == {"enabled": True}


def test_vision_timing_migrates_and_saves_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glados.vision.vision_mind.CameraSampler", Mock())
    path = tmp_path / "vision_settings.yaml"
    path.with_suffix(".json").write_text('{"interval_min_s": 3, "interval_max_s": 7}')
    mind = VisionMind(VisionConfig(), LLMConfig("http://unused"), VisionState(), TaskSlotStore(), settings_path=path)
    assert mind.settings.interval_min_s == 3 and mind.settings.interval_max_s == 7
    mind.set_interval_range(4, 8)
    assert yaml.safe_load(path.read_text()) == {"interval_min_s": 4.0, "interval_max_s": 8.0}
    restored = VisionMind(VisionConfig(), LLMConfig("http://unused"), VisionState(),
                          TaskSlotStore(), settings_path=path)
    assert restored.settings.interval_min_s == 4 and restored.settings.interval_max_s == 8


def test_routing_lists_migrate_and_hand_edits_reload(tmp_path: Path) -> None:
    legacy_path = tmp_path / "decision_lists.json"
    legacy = DecisionListStore(legacy_path, lambda: [])
    legacy._persist(legacy.snapshot())
    original = legacy_path.read_text()
    path = legacy_path.with_suffix(".yaml")
    migrated = DecisionListStore(path, lambda: [])
    assert migrated.snapshot() == yaml.safe_load(path.read_text())
    data = yaml.safe_load(path.read_text())
    data["lists"][0]["name"] = "Hand edited routing list"
    path.write_text(yaml.safe_dump(data))
    assert DecisionListStore(path, lambda: []).snapshot()["lists"][0]["name"] == "Hand edited routing list"
    assert legacy_path.read_text() == original


def test_user_preferences_migrate_and_console_changes_remain_yaml(tmp_path: Path) -> None:
    path = tmp_path / "preferences.yaml"
    original = '{"theme": "dark", "topics": ["AI", "science"]}'
    path.with_suffix(".json").write_text(original)
    store = Store(path)
    assert store.get("theme") == "dark" and store.get("topics") == ["AI", "science"]
    assert yaml.safe_load(path.read_text()) == store.all()
    store.set("language", "English")
    assert Store(path).get("language") == "English"
    assert path.with_suffix(".json").read_text() == original


def test_response_instructions_save_and_reload_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "operator_settings.yaml"
    state = OperatorState(path)
    state.set_instructions("Keep replies brief.\nAddress me as the test subject.")
    assert OperatorState(path).snapshot()["instructions"] == state.snapshot()["instructions"]
    assert yaml.safe_load(path.read_text())["instructions"].startswith("Keep replies brief.")
    assert "instructions: |-" in path.read_text()
    previous = path.read_text()
    with pytest.raises(ValueError):
        state.set_instructions("x" * 4001)
    assert path.read_text() == previous


def test_yaml_dates_remain_usable_in_json_tool_messages(tmp_path: Path) -> None:
    path = tmp_path / "preferences.yaml"
    path.write_text("birthday: 2026-10-06\n")
    assert json.loads(json.dumps(Store(path).all())) == {"birthday": "2026-10-06"}
