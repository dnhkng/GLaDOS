from dataclasses import dataclass
import io
from typing import Literal

from litestar import Litestar, post
from litestar.response import Stream

from .log import structlog_plugin
from .tts import write_glados_audio_file

Voice = Literal["glados"]
ResponseFormat = Literal["mp3", "wav", "ogg"]


@dataclass
class RequestData:
    input: str
    model: str = "glados"
    voice: Voice = "glados"
    response_format: ResponseFormat = "mp3"
    speed: float = 1.0


CONTENT_TYPES: dict[ResponseFormat, str] = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}


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


app = Litestar([create_speech], plugins=[structlog_plugin])
