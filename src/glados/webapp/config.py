"""Configuration for the Glados webapp observability console."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WebappConfig(BaseModel):
    """Webapp observability console server configuration."""

    enabled: bool = Field(
        default=False,
        description="Serve the webapp observability console (default off).",
    )
    host: Literal["127.0.0.1", "localhost"] = Field(
        default="127.0.0.1",
        description="Loopback listen address; remote access is intentionally disabled.",
    )
    port: int = Field(default=8050, ge=0, le=65535, description="Listen port.")
