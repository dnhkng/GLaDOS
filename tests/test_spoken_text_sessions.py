"""Replay real session speech with reviewed expected text, without model downloads."""

from pathlib import Path

import pytest
import yaml

from glados.utils.spoken_text_converter import SpokenTextConverter

_CASES = yaml.safe_load((Path(__file__).parent / "fixtures" / "spoken_text_sessions.yaml").read_text())


@pytest.mark.parametrize("case", _CASES, ids=[case["id"] for case in _CASES])
def test_logged_speech(case: dict[str, str]) -> None:
    assert SpokenTextConverter().text_to_spoken(case["text"]) == case["expected"], case["source"]
