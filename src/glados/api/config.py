import os

from pydantic import BaseModel, Field, model_validator


class ApiConfig(BaseModel):
    """Configuration for the OpenAI-compatible TTS API server."""

    reuse_tts: bool = Field(
        default=True,
        description="Reuse a single SpeechSynthesizer instance across requests instead of reloading ONNX each call.",
    )

    @model_validator(mode="after")
    def _apply_env_overrides(self) -> "ApiConfig":
        env_reuse_tts = os.environ.get("GLADOS_API_REUSE_TTS")
        if env_reuse_tts is not None:
            self.reuse_tts = env_reuse_tts.lower() in ("1", "true", "yes")
        return self
