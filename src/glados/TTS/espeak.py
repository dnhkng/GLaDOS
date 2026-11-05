"""espeak-ng phonemization for non-English speech with the Piper-trained GLaDOS voice.

The voice's phoneme table is espeak's IPA set, so any language espeak-ng speaks maps onto it.
Punctuation is kept between clauses because the model uses it for pauses and intonation.
"""

from __future__ import annotations

import re
import shutil
import subprocess

_CLAUSE = re.compile(r"([^.,;:!?…]+)([.,;:!?…]*)")
_SENTENCE_END = set(".!?…")
VOICES = {"fr": "fr-fr"}
# espeak spells these out letter by letter.
RESPELLINGS = {
    "fr": {
        r"\bGLaDOS\b": "Gladosse",
        r"\bAperture\b": "Apertchur",
        r"\b(\d{1,2}) ?h ?(\d{2})\b": r"\1 heures \2",
        r"\b(\d{1,2}) ?h\b": r"\1 heures",
    }
}


class EspeakPhonemizer:
    def __init__(self, language: str) -> None:
        binary = shutil.which("espeak-ng")
        if binary is None:
            raise RuntimeError(f"espeak-ng is required for '{language}' speech with the GLaDOS voice")
        self._binary = binary
        self._voice = VOICES.get(language, language)
        self._respell = [
            (re.compile(pattern, re.IGNORECASE), word) for pattern, word in RESPELLINGS.get(language, {}).items()
        ]

    def _ipa(self, text: str) -> str:
        completed = subprocess.run(
            [self._binary, "-q", "--ipa", "-v", self._voice],
            input=text,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return " ".join(completed.stdout.split())

    def __call__(self, text: str) -> list[str]:
        """Return one phoneme string per sentence."""
        for pattern, word in self._respell:
            text = pattern.sub(word, text)
        sentences, current = [], ""
        for words, punctuation in _CLAUSE.findall(text):
            if not words.strip():
                continue
            current += self._ipa(words) + (punctuation[:1] or "") + " "
            if not punctuation or punctuation[-1] in _SENTENCE_END:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        return sentences
