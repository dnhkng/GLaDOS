from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl

class VisionConfig(BaseModel):
    """Configuration values for the vision module."""
    vlm_model: str = Field(description="The model name for the vision-language model to generate scene descriptions.", examples=["qwen3-vl:2b-instruct-q4_K_M"])
    completion_url: HttpUrl = Field(description="The URL for the completion API endpoint.", examples=["http://localhost:11434/api/chat"])
    api_key: str | None = Field(default=None, description="API key for authentication with the completion service.")
    camera_index: int = Field(default=0, ge=0, description="The index of the camera to use for capturing images. Use 0 if only one camera is connected.")
    capture_interval_seconds: float = Field(default=10.0, gt=0.0, description="Interval in seconds between image captures. Tune this to your own system.")
    resolution: int = Field(default=500, gt=0, description="The resolution (in pixels) to which captured images are resized before processing. Smaller sizes reduce computational load but may affect description quality.")
