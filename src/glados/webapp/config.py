"""Configuration for the Glados webapp observability console."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


def console_hosts(host: str, allowed_hosts: list[str]) -> frozenset[str]:
    """Build an explicit Host allowlist; a wildcard bind is never an allowed Host."""
    bind = host.strip("[]").lower()
    wildcards = {"", "0.0.0.0", "::"}
    extra = {value.strip("[]").lower() for value in allowed_hosts}
    if any(not value or value in wildcards or any(c.isspace() or c in "/@?#" for c in value) for value in extra):
        raise ValueError("allowed_hosts must contain explicit hostnames or IP addresses")
    if bind in wildcards and not extra:
        raise ValueError("A wildcard webapp host requires an explicit allowed_hosts list")
    return frozenset({"127.0.0.1", "localhost", "::1"} | extra | ({bind} if bind not in wildcards else set()))


class WebappConfig(BaseModel):
    """Webapp observability console server configuration."""

    enabled: bool = Field(
        default=False,
        description="Serve the webapp observability console (default off).",
    )
    host: str = Field(default="127.0.0.1", description="Listen address.")
    port: int = Field(default=8050, description="Listen port.")
    allowed_hosts: list[str] = Field(default_factory=list, description="Additional HTTP Host names, without ports.")

    @model_validator(mode="after")
    def validate_hosts(self) -> WebappConfig:
        if self.enabled:
            console_hosts(self.host, self.allowed_hosts)
        return self
