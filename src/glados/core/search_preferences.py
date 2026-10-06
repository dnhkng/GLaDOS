"""Bounded, persistent source preferences for requested internet research."""

from pathlib import Path
import re
import threading
from urllib.parse import urlsplit

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .settings_files import read_settings, settings_source, write_settings


class SearchSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weather: list[str] = Field(default_factory=lambda: ["dwd.de", "meteoblue.com"], max_length=8)
    news: list[str] = Field(default_factory=lambda: ["reuters.com", "bbc.com/news"], max_length=8)
    reddit: list[str] = Field(default_factory=list, max_length=8)
    general: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("weather", "news", "reddit", "general")
    @classmethod
    def normalize_sites(cls, sites: list[str]) -> list[str]:
        normalized = []
        for site in sites:
            site = site.strip()
            if not site or len(site) > 200 or re.search(r"\s", site):
                raise ValueError("Enter a domain or URL path, at most 200 characters, without spaces")
            parsed = urlsplit(site if "://" in site else "https://" + site)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.username
                or parsed.password
                or parsed.port
                or parsed.query
                or parsed.fragment
                or not parsed.hostname
                or not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", parsed.hostname)
                or "." not in parsed.hostname
                or re.search(r"[^a-zA-Z0-9/_.~%-]", parsed.path)
            ):
                raise ValueError("Use a public domain or URL path without credentials, ports, queries or fragments")
            value = parsed.hostname.lower() + parsed.path.rstrip("/")
            if value not in normalized:
                normalized.append(value)
        return normalized


class SearchPreferences:
    def __init__(self, defaults: SearchSources, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._path = path
        self._sources = defaults.model_copy(deep=True)
        if path and settings_source(path).exists():
            try:
                self._sources = SearchSources.model_validate(read_settings(path))
            except (OSError, ValueError) as exc:
                logger.warning("Could not load search source preferences; using configured values: {}", exc)
        if path and path.suffix in {".yaml", ".yml"} and not path.exists():
            write_settings(path, self._sources.model_dump())

    def snapshot(self) -> dict[str, list[str]]:
        with self._lock:
            return self._sources.model_dump()

    def update(self, values: dict) -> dict[str, list[str]]:
        if set(values) != set(SearchSources.model_fields):
            raise ValueError("Provide weather, news, reddit and general lists")
        validated = SearchSources.model_validate(values)
        with self._lock:
            if self._path:
                write_settings(self._path, validated.model_dump())
            self._sources = validated
            return validated.model_dump()

    def relevant(self, request: str) -> dict[str, list[str]]:
        """Take one settings snapshot per research request; unrelated lists stay out of context."""
        sources = self.snapshot()
        selected = {"general": sources["general"]} if sources["general"] else {}
        terms = {
            "weather": r"\b(?:weather|wetter|meteorological|precipitation|vorhersage)\b",
            "news": r"\b(?:news|headlines|nachrichten|current events)\b",
            "reddit": r"\b(?:reddit|subreddits?)\b|reddit\.com/r/",
        }
        for category, pattern in terms.items():
            if sources[category] and re.search(pattern, request, re.I):
                selected[category] = sources[category]
        return selected
