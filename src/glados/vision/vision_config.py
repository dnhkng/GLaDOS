from __future__ import annotations

from pathlib import Path
from typing import Union

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
        validation_alias=AliasChoices("camera_index", "camera_spec"),
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
