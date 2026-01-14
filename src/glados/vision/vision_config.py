from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class VisionConfig(BaseModel):
    """Configuration for ONNX-based FastVLM vision module."""

    model_dir: Path | None = Field(
        default=None,
        description="Path to FastVLM ONNX model directory. Uses default if None.",
    )
    camera_index: int = Field(
        default=0,
        ge=0,
        description="The index of the camera to use for capturing images. Use 0 if only one camera is connected.",
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
