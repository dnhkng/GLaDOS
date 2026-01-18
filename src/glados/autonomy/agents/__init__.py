"""
Concrete subagent implementations for GLaDOS autonomy system.
"""

from .hacker_news import HackerNewsSubagent
from .weather import WeatherSubagent

__all__ = [
    "HackerNewsSubagent",
    "WeatherSubagent",
]
