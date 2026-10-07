"""Atomic YAML settings, with read-only migration from earlier JSON files."""

import json
from pathlib import Path
import re
from typing import ClassVar

import yaml


class _SettingsLoader(yaml.SafeLoader):
    # Settings are also used in JSON tool/API messages: retain calendar values as strings.
    yaml_implicit_resolvers: ClassVar[dict[str | None, list[tuple[str, re.Pattern[str]]]]] = {
        key: [(tag, pattern) for tag, pattern in values if tag != "tag:yaml.org,2002:timestamp"]
        for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


class _SettingsDumper(yaml.SafeDumper):
    pass


def _string_value(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_SettingsDumper.add_representer(str, _string_value)


def settings_source(path: Path) -> Path:
    if path.suffix in {".yaml", ".yml"} and not path.exists():
        legacy = path.with_suffix(".json")
        if legacy.exists():
            return legacy
    return path


def read_settings(path: Path) -> object:
    source = settings_source(path)
    text = source.read_text(encoding="utf-8")
    try:
        return yaml.load(text, Loader=_SettingsLoader) if source.suffix in {".yaml", ".yml"} else json.loads(text)
    except yaml.YAMLError as exc:
        raise ValueError("Invalid settings YAML") from exc


def write_settings(path: Path, values: dict) -> None:
    if path.suffix in {".yaml", ".yml"}:
        text = "# Edit this file and restart GLaDOS, or save changes through the console.\n"
        text += yaml.dump(values, Dumper=_SettingsDumper, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(values, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
