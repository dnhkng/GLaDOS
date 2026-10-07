"""Direct page reads retain visible headlines and links within strict network and evidence bounds."""

from unittest.mock import MagicMock, Mock

import pytest

from glados.core.news_pages import page_source, read_news_page

HTML = """<html><head><title>Example News</title><script>Invented secret headline</script></head>
<body><nav>Ignore this navigation and login menu</nav><main><h2>Tuesday, October 6, 2026</h2>
<div><a href="/story/1">Example Lab announces a new model &amp; tools</a></div>
<p>Today the lab released documentation. Details are available in the linked article.</p>
<a href="javascript:bad()">No executable link</a></main></body></html>"""


def response(html: str = HTML) -> MagicMock:
    result = MagicMock(status_code=200, headers={"Content-Type": "text/html; charset=utf-8"}, encoding="utf-8")
    result.__enter__.return_value = result
    result.iter_content.return_value = [html.encode()]
    return result


def test_visible_page_keeps_dates_headlines_and_absolute_links() -> None:
    source = page_source(HTML, "https://news.example.org/", "2026-10-06T16:00:00Z")
    assert source is not None and source["published"] is None
    assert source["retrieved_at"] == "2026-10-06T16:00:00Z" and source["kind"] == "news_page"
    excerpt = source["excerpt"]
    assert "October 6, 2026" in excerpt and "new model & tools" in excerpt
    assert "https://news.example.org/story/1" in excerpt
    assert "Invented secret headline" not in excerpt and "navigation" not in excerpt
    assert "javascript:" not in excerpt


def test_page_excerpt_bound_keeps_update_label_despite_long_frontpage() -> None:
    html = "<h1>News</h1><p>" + "A headline with useful evidence. " * 250 + "</p>"
    html += "<span>News as of Oct 6, 12:25 PM EDT</span>"
    source = page_source(html, "https://huggingnews.com/", "2026-10-06T16:00:00Z")
    assert source and len(source["excerpt"]) <= 3500
    assert "News as of Oct 6, 12:25 PM EDT" in source["excerpt"]
    assert page_source("<script>Lots of hidden text</script>", "https://example.org/", "now") is None


def test_direct_fetch_streams_with_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    get = Mock(return_value=response())
    monkeypatch.setattr("glados.core.news_pages.requests.get", get)
    result = read_news_page("news.example.org/front", 8, lambda: False)
    assert result and result["url"] == "https://news.example.org/front"
    assert get.call_args.args == ("https://news.example.org/front",)
    assert get.call_args.kwargs["stream"] and not get.call_args.kwargs["allow_redirects"]
    assert max(get.call_args.kwargs["timeout"]) <= 8


def test_read_follows_relative_redirect_and_cites_final_page(monkeypatch: pytest.MonkeyPatch) -> None:
    redirect = response()
    redirect.status_code = 302
    redirect.headers = {"Location": "/news"}
    get = Mock(side_effect=[redirect, response()])
    monkeypatch.setattr("glados.core.news_pages.requests.get", get)
    source = read_news_page("news.example.org", 8, lambda: False)
    assert source and source["url"] == "https://news.example.org/news"
    assert get.call_args.args == ("https://news.example.org/news",)


@pytest.mark.parametrize("failure", ["size", "type", "redirect", "unsafe_redirect"])
def test_unreadable_or_unbounded_page_fails_for_search_fallback(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    result = response()
    if failure == "size":
        result.iter_content.return_value = [b"x" * 8192] * 62
    elif failure == "type":
        result.headers = {"Content-Type": "application/pdf"}
    else:
        result.status_code = 302
        result.headers = {"Location": "file:///etc/passwd" if failure == "unsafe_redirect" else "/again"}
    get = Mock(return_value=result)
    monkeypatch.setattr("glados.core.news_pages.requests.get", get)
    with pytest.raises(ValueError):
        read_news_page("news.example.org", 8, lambda: False)
    assert get.call_count <= 4


def test_cancellation_before_and_during_download_publishes_no_page(monkeypatch: pytest.MonkeyPatch) -> None:
    get = Mock(return_value=response())
    monkeypatch.setattr("glados.core.news_pages.requests.get", get)
    assert read_news_page("news.example.org", 8, lambda: True) is None
    get.assert_not_called()
    assert read_news_page("news.example.org", 8, Mock(side_effect=[False, True])) is None


def test_fetch_deadline_also_bounds_a_slow_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glados.core.news_pages.requests.get", Mock(return_value=response()))
    monkeypatch.setattr("glados.core.news_pages.time.monotonic", Mock(side_effect=[0, 0, 0, 9]))
    assert read_news_page("news.example.org", 8, lambda: False) is None
