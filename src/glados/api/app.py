from dataclasses import dataclass
import tempfile
from typing import Literal

from litestar import Litestar, post
from litestar.response import File

from .log import structlog_plugin
from .tts import write_glados_audio_file

Voice = Literal["glados"]
ResponseFormat = Literal["mp3"]


@dataclass
class RequestData:
    input: str
    model: str = "glados"
    voice: Voice = "glados"
    response_format: ResponseFormat = "mp3"
    speed: float = 1.0


@post("/v1/audio/speech")
async def create_speech(data: RequestData) -> File:
    """
    Generate speech audio from input text.

    Parameters:
        data: The request data containing input text and speech parameters

    Returns:
        File: Audio file containing the generated speech
    """
    # TODO: Handle other voices
    # TODO: Handle other formats
    # TODO: Handle speed
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_output_file:
        write_glados_audio_file(temp_output_file, data.input)
        return File(
            temp_output_file.name,
            content_disposition_type="attachment",
            filename="speech.mp3",
        )


app = Litestar([create_speech], plugins=[structlog_plugin])
