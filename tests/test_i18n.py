"""Every English string the console ships must have a French rendering (glados-i18n.js)."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src/glados/webapp/static"
# Names, sample values and placeholders that read the same in French.
UNTRANSLATED = {"GLaDOS", "GLaDOS · Aperture Science", "APERTURE SCIENCE · CORE CONSOLE", "SF Mono", "Fira Code",
                "Segoe UI", "IBM Plex Mono", "PotatOS", "Terminal", "Audio", "Vision", "Interaction", "Diagnostics",
                "Consultation", "Auto", "Dominance", "Action", "Microphone"}
SKIPPED_TAGS = {"script", "style", "pre", "code", "textarea"}

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required for the console scripts")


class _Strings(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth, self.found = 0, []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.depth += tag in SKIPPED_TAGS
        self.found += [v for k, v in attrs if k in {"placeholder", "title", "aria-label"} and v]

    def handle_endtag(self, tag: str) -> None:
        self.depth -= tag in SKIPPED_TAGS

    def handle_data(self, data: str) -> None:
        if not self.depth:
            self.found.append(data)


def _french(strings: list[str]) -> list[str]:
    result = subprocess.run(["node", str(Path(__file__).with_name("i18n_keys.cjs"))], input=json.dumps(strings),
                            capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def test_every_static_console_string_has_a_french_translation() -> None:
    parser = _Strings()
    parser.feed((STATIC / "index.html").read_text())
    strings = sorted({" ".join(s.split()) for s in parser.found if re.search("[A-Za-z]{2}", s)})
    # Sample source lists (one site per line) are user data, not interface text.
    strings = [s for s in strings if s not in UNTRANSLATED and not re.fullmatch(r"[\w./ -]+\.\w+(/[\w./ -]*)?", s)]
    missing = [s for s, fr in zip(strings, _french(strings), strict=True) if fr == s]
    assert not missing, f"Add French for these console strings in glados-i18n.js: {missing}"


def test_live_values_keep_their_numbers() -> None:
    assert _french(["3 in-flight", "Demo · Idle", "Microphone: off", "Median of last 12", "Unknown text"]) == [
        "3 en cours", "Démo · Inactif", "Micro : désactivé", "Médiane des 12 dernières", "Unknown text"]
