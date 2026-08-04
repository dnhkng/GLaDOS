from __future__ import annotations

from pathlib import Path
from typing import Union
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, AliasChoices, NonNegativeInt, StrictStr


class VisionConfig(BaseModel):
    """Configuration for ONNX-based FastVLM vision module."""

    model_dir: Path | None = Field(
        default=None,
        description="Path to FastVLM ONNX model directory. Uses default if None.",
    )
    camera_spec: Union[NonNegativeInt, StrictStr] = Field(
        default=0,
        description="The specification of the camera to use for capturing images. Integers will be interpreted as a camera index, strings will be interpreted as a URI/filename. Use 0 if only one camera is connected.",
        validation_alias=AliasChoices("camera_spec", "camera_index"),
        union_mode='left_to_right',
    )
    capture_interval_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Interval in seconds between image captures. Tune this to your own system.",
    )
    resolution: int = Field(
        default=384,
        gt=0,
        description="Resolution (in pixels) used for scene-change detection. FastVLM handles its own resize.",
    )
    scene_change_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum normalized difference between frames to trigger VLM inference. 0=always process, 1=never process.",
    )
    max_tokens: int = Field(
        default=64,
        gt=0,
        le=512,
        description="Maximum tokens to generate in the background vision description.",
    )

    def redacted_camera_spec_for_log(self) -> str:
        """
        Redact the camera spec for logging: censor credentials in a possible URL
        Returns:
            The redacted camera spec.
        """
        if isinstance(self.camera_spec, int):
            return str(self.camera_spec)
        elif isinstance(self.camera_spec, str):
            # works for files as well
            parts = urlsplit(self.camera_spec)
            if parts.username or parts.password:
                host = parts.hostname or ""
                if parts.port:
                    host = f"{host}:{parts.port}"
                redacted_netloc = f"***:***@{host}"
                return '"' + urlunsplit((parts.scheme, redacted_netloc, parts.path, parts.query, parts.fragment)) + '"'
        return '"' + self.camera_spec + '"'
