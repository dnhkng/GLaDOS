"""Camera observation always uses a dedicated Gemma 4 E4B endpoint."""

import math
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import AliasChoices, BaseModel, Field, NonNegativeInt, StrictInt, StrictStr, model_validator


class VisionConfig(BaseModel):
    enabled: bool = True
    # Do not inherit the speaking agent's endpoint, model, or credentials.
    completion_url: str = "http://127.0.0.1:18080/v1/chat/completions"
    model: str = "gemma-4-E4B"  # E4B's server alias, including on a remote E4B server.
    api_key: str | None = None
    timeout_s: float = Field(default=10, gt=0, le=120)
    camera_spec: NonNegativeInt | StrictStr = Field(
        default=0,
        validation_alias=AliasChoices("camera_spec", "camera_index"),
        union_mode="left_to_right",
    )
    interval_min_s: StrictInt = Field(default=2, ge=1, le=60)
    interval_max_s: StrictInt = Field(default=5, ge=1, le=60)
    frame_window_s: float = Field(default=0.25, ge=0.05, le=1)
    image_max_side: int = Field(default=512, ge=128, le=1024)
    max_tokens: int = Field(default=256, ge=48, le=512)
    question_image_max_side: int = Field(default=1024, ge=128, le=2048)
    question_max_tokens: int = Field(default=192, ge=48, le=512)
    face_tracking: bool = True
    face_backend: Literal["yunet", "e4b", "haar"] = "yunet"
    face_interval_s: float = Field(default=0.03, ge=1 / 60, le=2)
    sleep_after_s: float = Field(default=5.0, ge=2, le=60)
    greeting_absence_s: float = Field(default=60, ge=10, le=3600)
    morning_greeting_enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def migrate_fixed_interval(cls, data: object) -> object:
        if isinstance(data, dict) and "interval_s" in data:
            data = dict(data)
            old = data.pop("interval_s")
            if "interval_min_s" not in data and "interval_max_s" not in data:
                if type(old) not in (int, float) or not 0.2 <= old <= 60:
                    raise ValueError("Legacy Vision interval must be between 0.2 and 60 seconds")
                data.update(interval_min_s=math.ceil(old), interval_max_s=math.ceil(old))
        return data

    @model_validator(mode="after")
    def validate_interval_range(self) -> "VisionConfig":
        if self.interval_min_s > self.interval_max_s:
            raise ValueError("Minimum Vision interval must not exceed maximum")
        return self

    @property
    def interval_s(self) -> float:
        """Mean cadence for rate displays; actual waits are sampled per observation."""
        return (self.interval_min_s + self.interval_max_s) / 2

    def redacted_camera_spec_for_log(self) -> str:
        if isinstance(self.camera_spec, int):
            return str(self.camera_spec)
        parts = urlsplit(self.camera_spec)
        if parts.username or parts.password:
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))
        return self.camera_spec
