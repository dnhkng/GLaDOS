"""Local, synthetic multimodal smoke benchmark; requires espeak-ng and a running llama-server.

Run each backend separately to avoid GPU contention. Results are not a natural-speech evaluation.
"""

import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import cv2
from Levenshtein import distance
import numpy as np
import requests
from scipy.signal import resample_poly
import soundfile as sf


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gemma", "parakeet"], default="gemma")
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1/chat/completions")
    parser.add_argument("--model", default="gemma-4-E4B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    transcriber = None
    if args.backend == "parakeet":
        from loguru import logger

        from glados.ASR import get_audio_transcriber

        logger.remove()
        started = time.perf_counter()
        transcriber = get_audio_transcriber("tdt")
        print(f"Parakeet loaded in {time.perf_counter() - started:.2f}s", flush=True)

    def request(content: object, system: str | None = None) -> tuple[str, float, float]:
        messages = [{"role": "user", "content": content}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        started = time.perf_counter()
        first = None
        text = ""
        with requests.post(
            args.url,
            json={
                "model": args.model,
                "messages": messages,
                "stream": True,
                "temperature": 0,
                "max_tokens": 256,
                "seed": 42,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
            timeout=180,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                chunk = event.get("choices", [{}])[0].get("delta", {}).get("content")
                if chunk:
                    if first is None:
                        first = time.perf_counter() - started
                    text += chunk
        return text, first, time.perf_counter() - started

    def record(row: dict) -> None:
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "backend": args.backend,
                    "model": args.model,
                    "fixtures": "espeak-ng synthetic speech / OpenCV generated image",
                    "results": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

    with tempfile.TemporaryDirectory(prefix="glados-gemma-") as folder:
        for name, voice, reference in [
            ("english", "en-us", "Please turn off the kitchen lights and set a timer for seven minutes."),
            ("german", "de", "Bitte schalte das Licht in der Küche aus und stelle einen Timer auf sieben Minuten."),
        ]:
            wav = Path(folder) / f"{name}.wav"
            subprocess.run(["espeak-ng", "-v", voice, "-s", "155", "-w", str(wav), reference], check=True)
            samples, rate = sf.read(wav)
            assert rate == 22050
            samples = resample_poly(samples, 320, 441)
            sf.write(wav, samples, 16000)
            samples, rate = sf.read(wav)
            duration = len(samples) / rate
            for repeat in range(3):
                if transcriber:
                    started = time.perf_counter()
                    text = transcriber.transcribe_file(wav)
                    total = time.perf_counter() - started
                    first = None
                else:
                    text, first, total = request(
                        [
                            {
                                "type": "text",
                                "text": "Transcribe the audio exactly in its original language. "
                                "Output only the transcript.",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": base64.b64encode(wav.read_bytes()).decode(), "format": "wav"},
                            },
                        ]
                    )
                record(
                    {
                        "case": name,
                        "repeat": repeat,
                        "reference": reference,
                        "response": text,
                        "audio_s": duration,
                        "first_text_s": first,
                        "total_s": total,
                        "real_time_factor": total / duration,
                        "word_error_rate": distance(words(reference), words(text)) / len(words(reference)),
                    }
                )

        if not transcriber:
            image = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (65, 160), (235, 330), (0, 0, 255), -1)
            cv2.circle(image, (445, 245), 90, (255, 0, 0), -1)
            cv2.putText(image, "TEST 42", (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
            encoded = base64.b64encode(cv2.imencode(".png", image)[1]).decode()
            for repeat in range(3):
                text, first, total = request(
                    [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                        {
                            "type": "text",
                            "text": "Read the text. Describe the two shapes, their colours, "
                            "and which is on the left. Be brief.",
                        },
                    ]
                )
                record({"case": "image", "repeat": repeat, "response": text, "first_text_s": first, "total_s": total})
            from glados.core.speech_markup import SPEECH_DIRECTION_PROMPT

            text, first, total = request(
                "Congratulate me for passing the test, then express disappointment that I broke the equipment. "
                "Use two emotion markers and two short sentences.",
                SPEECH_DIRECTION_PROMPT,
            )
            record({"case": "emotion_markup", "response": text, "first_text_s": first, "total_s": total})


if __name__ == "__main__":
    main()
