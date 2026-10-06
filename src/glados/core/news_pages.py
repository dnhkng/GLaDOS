"""Read bounded visible HTML from the user's preferred news pages, without executing scripts."""

from collections.abc import Callable
from datetime import UTC, datetime
from html.parser import HTMLParser
import time
from urllib.parse import urljoin, urlsplit

import requests

_MAX_BYTES = 500_000
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "wbr"}
_SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "title"}
_BLOCK = {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "br"}


class _VisiblePage(HTMLParser):
    def __init__(self, url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.url = url
        self.stack: list[tuple[str, bool, str]] = []
        self.parts: list[str] = []
        self.title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        ignored = bool(self.stack and self.stack[-1][1]) or tag in _SKIP or "hidden" in attributes
        ignored |= attributes.get("aria-hidden") == "true" or any(
            name in {"filter-chip", "trending-chip", "story-rank"} for name in classes
        )
        if tag in _BLOCK and not ignored:
            self.parts.append("\n")
        if tag not in _VOID:
            href = urljoin(self.url, attributes["href"]) if tag == "a" and attributes.get("href") else ""
            self.stack.append((tag, ignored, href))

    def handle_endtag(self, tag: str) -> None:
        index = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag), None)
        if index is None:
            return
        _, ignored, href = self.stack[index]
        if tag == "a" and not ignored and urlsplit(href).scheme in {"http", "https"}:
            self.parts.append(" (" + href[:600] + ") ")
        if tag in _BLOCK and not ignored:
            self.parts.append("\n")
        del self.stack[index:]

    def handle_data(self, data: str) -> None:
        if any(tag == "title" for tag, _, _ in self.stack):
            self.title.append(data)
        if not self.stack or not self.stack[-1][1]:
            self.parts.append(data)


def page_source(html: str, url: str, retrieved_at: str) -> dict | None:
    parser = _VisiblePage(url)
    parser.feed(html)
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    text = "\n".join(line for line in lines if line)
    if len(text) < 80:
        return None
    title = " ".join(" ".join(parser.title).split())[:200] or urlsplit(url).hostname
    header = f"Title: {title}\nURL: {url}\nRetrieved: {retrieved_at}\nVisible page:\n"
    update_label = next((line for line in lines if "News as of " in line), "")
    if update_label:
        text = update_label[:200] + "\n" + text
    return {"url": url, "title": title, "published": None, "retrieved_at": retrieved_at,
            "kind": "news_page", "excerpt": header + text[:3500 - len(header)]}


def read_news_page(site: str, timeout: float, cancelled: Callable[[], bool]) -> dict | None:
    """Fetch only configured pages, with a deadline and a strict response-size bound."""
    url = "https://" + site
    deadline = time.monotonic() + timeout
    for _ in range(4):
        if cancelled() or time.monotonic() >= deadline:
            return None
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.port not in {None, 80, 443}):
            raise ValueError("Invalid news page URL")
        budget = max(0.1, deadline - time.monotonic())
        headers = {"User-Agent": "GLaDOS-NewsReader/1.0", "Accept": "text/html,text/plain"}
        with requests.get(url, timeout=(min(3.0, budget), budget), stream=True,
                          allow_redirects=False, headers=headers) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                url = urljoin(url, response.headers.get("Location", ""))
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(kind in content_type for kind in ("text/html", "application/xhtml+xml", "text/plain")):
                raise ValueError("News page did not return readable text")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                if cancelled() or time.monotonic() >= deadline:
                    return None
                if len(body) + len(chunk) > _MAX_BYTES:
                    raise ValueError("News page exceeds the reading size limit")
                body.extend(chunk)
            encoding = response.encoding if response.encoding and "charset=" in content_type else "utf-8"
            html = body.decode(encoding, errors="replace")
            return page_source(html, url, datetime.now(UTC).isoformat(timespec="seconds"))
    raise ValueError("Too many news page redirects")
