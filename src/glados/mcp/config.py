from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    url: HttpUrl | None = None
    headers: dict[str, str] | None = None
    token: str | None = None
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] | None = None
    context_resources: list[str] = Field(default_factory=list)
    resource_ttl_s: float = Field(default=300.0, ge=0.0)
