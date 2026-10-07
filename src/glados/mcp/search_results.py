"""Bound search evidence while retaining each source's title, URL and date."""

import re

SEARCH_TOOL = "mcp.internet_search.web_search_exa"
EXCERPT_NOTICE = "[Search excerpts shortened to fit model context.]"


def compact_search_results(text: str, max_chars: int = 3500) -> str:
    if len(text) <= max_chars:
        return text
    blocks = re.split(r"\n(?=Title:)", text.strip())
    sources = []
    for block in blocks[:3]:
        lines = block.splitlines()
        metadata = [line for line in lines if line.startswith(("Title:", "URL:", "Published:", "Author:"))]
        evidence = [line for line in lines if line not in metadata and line.strip() != "---"]
        sources.append(("\n".join(metadata), "\n".join(evidence)))
    headers_size = sum(len(header) + 4 for header, _ in sources)
    available = max_chars - headers_size - len(EXCERPT_NOTICE) - 2
    if not sources or available < len(sources) * 100:
        return text[: max_chars - len(EXCERPT_NOTICE) - 1] + "\n" + EXCERPT_NOTICE
    per_source = available // len(sources)
    excerpts = [header + "\n" + evidence[:per_source] for header, evidence in sources]
    return "\n\n".join(excerpts) + "\n" + EXCERPT_NOTICE
