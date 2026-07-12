from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Literal

from litestar import Litestar, post
from litestar.response import Stream
import yaml

from glados.utils.resources import resource_path

from .config import ApiConfig
from .log import structlog_plugin
from .tts import configure_tts, write_glados_audio_file

Voice = Literal["glados"]
ResponseFormat = Literal["mp3", "wav", "ogg"]

DEFAULT_API_CONFIG = resource_path("configs/api_config.yaml")


@dataclass
class RequestData:
    input: str
    model: str = "glados"
    voice: Voice = "glados"
    response_format: ResponseFormat = "mp3"
    speed: float = 1.0


CONTENT_TYPES: dict[ResponseFormat, str] = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}


def load_api_config(path: str | Path | None = None) -> ApiConfig:
    """Load API settings from configs/api_config.yaml (Api: section)."""
    config_path = Path(path) if path is not None else DEFAULT_API_CONFIG
    data: dict[str, object] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        data = raw.get("Api", {}) or {}
    return ApiConfig.model_validate(data)


@post("/v1/audio/speech")
async def create_speech(data: RequestData) -> Stream:
    """
    Generate speech audio from input text.

    Parameters:
        data: The request data containing input text and speech parameters

    Returns:
        Stream: Stream of bytes data containing the generated speech
    """
    # TODO: Handle other voices
    # TODO: Handle speed
    buffer = io.BytesIO()
    write_glados_audio_file(buffer, data.input, format=data.response_format)
    buffer.seek(0)
    return Stream(
        buffer,
        headers={
            "content-type": CONTENT_TYPES[data.response_format],
            "content-disposition": f'attachment; filename="speech.{data.response_format}"',
        },
    )


def create_app() -> Litestar:
    """Create the Litestar application for the TTS API server."""

    @asynccontextmanager
    async def lifespan(app: Litestar) -> AsyncIterator[None]:
        configure_tts(load_api_config())
        yield

    return Litestar([create_speech], plugins=[structlog_plugin], lifespan=[lifespan])


app = create_app()
