"""Transient audio input for a local multimodal chat server, without a resident ASR model."""

import base64
from collections.abc import Callable
import io
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
import requests
import soundfile as sf

from .inference import InferenceCancelledError


class NativeAudioConfig(BaseModel):
    enabled: bool = False
    user_transcripts: bool = False
    language: str = "English"
    max_duration_s: float = Field(default=30.0, ge=1, le=30)


class NativeAudioInput:
    """Encode one VAD-delimited turn. PCM data never enters text history or telemetry."""

    SAMPLE_RATE = 16000

    def __init__(self, config: NativeAudioConfig) -> None:
        self.config = config

    def message(self, samples: list[NDArray[np.float32]]) -> dict[str, Any] | None:
        if not samples:
            return None
        audio = np.concatenate(samples)
        if not audio.size or np.max(np.abs(audio)) < 1e-10:
            return None
        if len(audio) > int(self.config.max_duration_s * self.SAMPLE_RATE):
            raise ValueError("Native audio turn exceeds the configured clip limit")
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return {
            "role": "user",
            "content": ("[User spoke via audio; optional transcript unavailable.]" if self.config.user_transcripts
                        else "[User spoke via audio; transcript disabled.]"),
            "_native_audio": [
                {
                    "type": "text",
                    "text": f"Listen and respond to the user's speech. Default language: {self.config.language}.",
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                        "format": "wav",
                    },
                },
            ],
        }

    def transcribe(
        self, content: list[dict[str, Any]], url: str, model: str, headers: dict[str, str],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> str:
        """Optional transcript from the same Gemma server; no additional model is loaded."""
        prompt = (
            f"Transcribe this {self.config.language} speech exactly. "
            "Return only the transcript, without commentary. If there is no speech, return an empty string."
        )
        with requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, *content[1:]]}],
                "stream": True,
                "temperature": 0,
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30,
            stream=True,
        ) as response:
            response.raise_for_status()
            parts = []
            for line in response.iter_lines(chunk_size=1):
                if cancelled():
                    raise InferenceCancelledError()
                if not line or not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    break
                data = json.loads(line[6:])
                if data.get("choices"):
                    text = data["choices"][0].get("delta", {}).get("content")
                    if text is not None:
                        if not isinstance(text, str):
                            raise ValueError("Audio transcript was not text")
                        parts.append(text)
            if cancelled():
                raise InferenceCancelledError()
            return "".join(parts).strip()
