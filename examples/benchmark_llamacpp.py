"""Measure routing cache reuse and concurrent decoding on an already running server.

Uses current decision settings from the webapp without executing tools. Audio is
synthetic and different for every request. Pause background minds externally for
isolated comparisons; this script does not change application settings.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import requests
from scipy.signal import resample_poly
import soundfile as sf

from glados.core.decision_lists import DecisionListStore
from glados.core.inference import InferenceScheduler
from glados.core.native_audio import NativeAudioConfig, NativeAudioInput
from glados.core.routing import DecisionRouter, RoutingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1/chat/completions")
    parser.add_argument("--app-url", default="http://127.0.0.1:8050")
    parser.add_argument("--model", default="gemma-4-E4B")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    response = requests.get(args.app_url + "/api/snapshot", timeout=5)
    response.raise_for_status()
    snapshot = response.json()
    rows = []
    real_post = requests.post
    phase, cache = "", None

    def measured_post(url: str, **kwargs: Any) -> requests.Response:  # noqa: ANN401 - forwards requests.post kwargs
        if url != args.url:
            return real_post(url, **kwargs)
        if cache is not None:
            kwargs["json"]["cache_prompt"] = cache
        start = time.perf_counter()
        result = real_post(url, **kwargs)
        result.raise_for_status()
        body = result.json()
        rows.append(
            {
                "phase": phase,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "timings": body.get("timings", {}),
                "usage": body.get("usage", {}),
            }
        )
        return result

    with tempfile.TemporaryDirectory(prefix="glados-cache-") as folder:
        path = Path(folder) / "decisions.json"
        path.write_text(json.dumps(snapshot["decisions"]))
        definitions = [{"type": "function", "function": t} for t in snapshot["tools"]]
        store = DecisionListStore(path, lambda: definitions, backend_key=snapshot["decisions"]["backend_key"])
        router = DecisionRouter(store, InferenceScheduler(), args.url, args.model, {}, RoutingConfig(enabled=True))
        tree = router.tree(store.get())
        root = tree.nodes["area"].decision
        router.token_ids([chr(65 + i) for i in range(len(root.options))])
        native = NativeAudioInput(NativeAudioConfig())

        def route(question: str, media: list | None = None, full: bool = False) -> dict:
            result = (
                router.score(store.get(), question, media, dry_run=True)
                if full
                else router._score_step(root, question, media, dry_run=True, record=False)
            )
            rows[-1].update(
                action=result["action"],
                option_id=result["option_id"],
                tool=result.get("tool"),
                accepted=result["accepted"],
            )
            return result

        with patch("glados.core.routing.requests.post", measured_post):
            for repeat in range(args.repeats):
                phase, cache = "uncached_text", False
                route("What time is it?")
                phase, cache = "identical_text", None
                route("What time is it?")
                phase = "changed_text"
                route("What is the CPU load?")
                # A changing history suffix also occurs in real router prompts.
                phase = "changing_history"
                result = router._score_step(
                    root,
                    "What time is it?",
                    context=[{"role": "assistant", "content": f"We have discussed {repeat + 1} tests."}],
                    dry_run=True,
                    record=False,
                )
                rows[-1].update(action=result["action"], option_id=result["option_id"], accepted=result["accepted"])
                for label, question in [("fresh_audio", "What time is it?"), ("fresh_audio", "What is the CPU load?")]:
                    wav = Path(folder) / "question.wav"
                    subprocess.run(
                        ["espeak-ng", "-v", "en-us", "-s", str(145 + repeat * 3), "-w", str(wav), question], check=True
                    )
                    samples, rate = sf.read(wav)
                    samples = resample_poly(samples, 16000, rate).astype(np.float32)
                    media = native.message([samples])["_native_audio"]
                    phase = label
                    route(question, media)
                phase = "hierarchical_text"
                start = time.perf_counter()
                result = route("What time is it?", full=True)
                rows.append(
                    {
                        "phase": "hierarchical_total",
                        "elapsed_ms": (time.perf_counter() - start) * 1000,
                        "accepted": result["accepted"],
                        "tool": result.get("tool"),
                        "stages": len(result.get("stages", [])),
                    }
                )

    # Long enough to measure decoding and concurrency, with a stable prefix.
    system = "Answer briefly in English. " + "Use clear language and explain each step accurately. " * 90

    def reply(index: int) -> dict:
        start = time.perf_counter()
        first, sentence, output, timings = None, None, "", {}
        with real_post(
            args.url,
            json={
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Explain why the sky is blue in two sentences. Request {index}."},
                ],
                "stream": True,
                "temperature": 0,
                "seed": 42,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
            timeout=30,
        ) as result:
            result.raise_for_status()
            for line in result.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                timings = event.get("timings", timings)
                for choice in event.get("choices", []):
                    token = choice.get("delta", {}).get("content") or ""
                    if token:
                        first = first if first is not None else (time.perf_counter() - start) * 1000
                        output += token
                        if sentence is None and any(c in token for c in ".!?"):
                            sentence = (time.perf_counter() - start) * 1000
        if not output or first is None:
            raise RuntimeError("Server returned no spoken response")
        return {
            "elapsed_ms": (time.perf_counter() - start) * 1000,
            "first_token_ms": first,
            "first_sentence_ms": sentence,
            "characters": len(output),
            "timings": timings,
        }

    reply(0)  # Warm the stable prefix before recording varying questions.
    for repeat in range(args.repeats):
        rows.append({"phase": "reply", **reply(repeat + 1)})
    with ThreadPoolExecutor(max_workers=2) as pool:
        for repeat in range(3):
            start = time.perf_counter()
            results = list(pool.map(reply, [100 + repeat * 2, 101 + repeat * 2]))
            rows.append(
                {"phase": "concurrent_pair", "elapsed_ms": (time.perf_counter() - start) * 1000, "requests": results}
            )

    summary = []
    for key in dict.fromkeys(row["phase"] for row in rows):
        group = [row for row in rows if row["phase"] == key]
        stats = {"phase": key, "median_ms": round(statistics.median(r["elapsed_ms"] for r in group), 2)}
        for field in ("first_token_ms", "first_sentence_ms"):
            values = [row[field] for row in group if row.get(field) is not None]
            if values:
                stats[field] = round(statistics.median(values), 2)
        stats["cached_tokens"] = [row["timings"].get("cache_n") for row in group if "timings" in row]
        stats["processed_tokens"] = [row["timings"].get("prompt_n") for row in group if "timings" in row]
        summary.append(stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "note": "Synthetic English audio; real router prompt and current local tools. "
                "No tools executed. No conversation history except the synthetic changing-history condition. "
                "Concurrent requests use two worker threads. Background load controlled externally.",
                "summary": summary,
                "samples": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
