"""Configuration for the Glados webapp observability console."""
from __future__ import annotations

from pydantic import BaseModel, Field


class WebappConfig(BaseModel):
    """Webapp observability console server configuration."""

    enabled: bool = Field(
        default=False,
        description="Serve the webapp observability console (default off).",
    )
    host: str = Field(default="127.0.0.1", description="Listen address.")
    port: int = Field(default=8050, description="Listen port.")
