# ruff: noqa: RUF001
"""Negative contractions must keep their pronunciation through text cleaning."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from glados.TTS.phonemizer import Phonemizer
from glados.utils.spoken_text_converter import SpokenTextConverter


@pytest.fixture
def phonemizer(monkeypatch: pytest.MonkeyPatch) -> Phonemizer:
    dictionaries = iter(
        [
            {"you": "juː", "see": "sˈiː", "it": "ɪt", "johns": "dʒˈɑːnz", "coat": "kˈoʊt", "wont": "wˈɔnt"},
            {},
            {},
        ]
    )

    def load(_path: Path) -> dict[str, object]:
        return next(dictionaries)

    monkeypatch.setattr(Phonemizer, "_load_pickle", staticmethod(load))
    monkeypatch.setattr("glados.TTS.phonemizer.ort.get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr("glados.TTS.phonemizer.ort.InferenceSession", Mock(return_value=Mock()))
    return Phonemizer()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Can't you see?", "kˈænt juː sˈiː?"),
        ("Can’t you see?", "kˈænt juː sˈiː?"),
        ("Don't you see?", "dˈoʊnt juː sˈiː?"),
        ("Won't you see?", "wˈoʊnt juː sˈiː?"),
        ("Isn't it?", "ˈɪzənt ɪt?"),
        ("Aren't you?", "ˈɑːɹnt juː?"),
        ("Doesn't it?", "dˈʌzənt ɪt?"),
    ],
)
def test_converter_to_phonemizer_question(phonemizer: Phonemizer, raw: str, expected: str) -> None:
    text = SpokenTextConverter().text_to_spoken(raw)
    assert phonemizer.convert_to_phonemes([text]) == [expected]
    phonemizer.ort_session.run.assert_not_called()


@pytest.mark.parametrize(
    "raw", ["can't", "Can't", "CAN'T", "can’t", "can‘t", "canʼt", "'can't'", "‘can’t’", '"can\'t"']
)
def test_apostrophe_and_case_variants(phonemizer: Phonemizer, raw: str) -> None:
    assert phonemizer.convert_to_phonemes([raw]) == ["kˈænt"]
    phonemizer.ort_session.run.assert_not_called()


def test_contraction_does_not_replace_different_word(phonemizer: Phonemizer) -> None:
    assert phonemizer.convert_to_phonemes(["won't", "wont"]) == ["wˈoʊnt", "wˈɔnt"]


@pytest.mark.parametrize("raw", ["John's coat", "John’s coat", "'Johns' coat"])
def test_possessives_and_quotes_keep_existing_pronunciation(phonemizer: Phonemizer, raw: str) -> None:
    assert phonemizer.convert_to_phonemes([raw]) == ["dʒˈɑːnz kˈoʊt"]
    phonemizer.ort_session.run.assert_not_called()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CPU", "sˈiː pˈiː jˈuː"),
        ("GPU", "dʒˈiː pˈiː jˈuː"),
        ("RAM", "ˈɑːɹ ˈeɪ ˈɛm"),
        ("RTX", "ˈɑːɹ tˈiː ˈɛks"),
        ("NVIDIA", "ˈɛn vˈiː ˈaɪ dˈiː ˈaɪ ˈeɪ"),
        ("C E S T", "sˈiː ˈiː ˈɛs tˈiː"),
        ("IoT", "ˈaɪ ˈoʊ tˈiː"),
    ],
)
def test_initialisms_pronounce_letter_names(phonemizer: Phonemizer, raw: str, expected: str) -> None:
    text = SpokenTextConverter().text_to_spoken(raw)
    assert phonemizer.convert_to_phonemes([text]) == [expected]
    phonemizer.ort_session.run.assert_not_called()


def test_lowercase_article_keeps_word_pronunciation(phonemizer: Phonemizer) -> None:
    phonemizer.phoneme_dict["a"] = "ɐ"
    assert phonemizer.convert_to_phonemes(["a C P U"]) == ["ɐ sˈiː pˈiː jˈuː"]
    phonemizer.ort_session.run.assert_not_called()


def test_glados_pronunciation_coexists_with_contractions(phonemizer: Phonemizer) -> None:
    text = SpokenTextConverter().text_to_spoken("GLaDOS can't")
    assert phonemizer.convert_to_phonemes([text]) == ["ɡlˈædoʊs kˈænt"]
    phonemizer.ort_session.run.assert_not_called()
