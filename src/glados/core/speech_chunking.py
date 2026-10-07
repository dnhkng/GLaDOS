"""Release natural speech clauses without relying on model token boundaries."""

import re

_BOUNDARY = re.compile(r'[.!?;:]+[\"\u201d\u2019\)]*(?=\s|$)|\n+')


def split_speech_clauses(text: str) -> tuple[list[str], str]:
    """Return complete clauses and the unfinished suffix.

    Keep numeric punctuation pending until another word disambiguates decimals
    and times. Keep URLs intact; the speech normalizer handles them later.
    """
    clauses: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        prefix = text[start:match.start()]
        if re.search(r'https?://\S*$|https?$', prefix):
            continue
        if (match.group()[0] in '.:' and prefix[-1:].isdigit()
                and match.end() == len(text)):
            continue
        clause = text[start:match.end()]
        if clause.strip():
            clauses.append(clause)
        start = match.end()
    return clauses, text[start:]
