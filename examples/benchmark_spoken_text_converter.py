"""Measure normalization latency without loading ASR, TTS, or LLM models.

Run with the project Python environment. To compare versions, save the original
converter to a file and pass --baseline /path/to/original.py. Warm-up and timed
iterations use identical inputs in one process; timings are microseconds per call.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
import timeit
from typing import Protocol

CORPUS = {
    "plain": "That was a remarkably predictable result.",
    "contractions": "I'm sure you won't enjoy this test.",
    "weather": "Tomorrow's forecast is -5°C to 12°C, with a 30% chance of rain.",
    "currency": "In 2026, the test cost $1,234.56 and took 3.14 seconds.",
    "time_and_date": "The meeting at 3:00pm on 1/1/2024 will cost $50.00.",
    "paragraph": "The test has 128 samples, costs $12.50 and starts at 8:05 pm. " * 20,
}


class Converter(Protocol):
    def text_to_spoken(self, text: str) -> str: ...


def load_converter(path: Path) -> Converter:
    spec = importlib.util.spec_from_file_location("bench_spoken_" + path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpokenTextConverter()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    current_path = Path(__file__).resolve().parents[1] / "src/glados/utils/spoken_text_converter.py"
    converters = {"current": load_converter(current_path)}
    if args.baseline:
        converters["baseline"] = load_converter(args.baseline)
    findings = {}
    for label, text in CORPUS.items():
        calls = {name: lambda c=c, t=text: c.text_to_spoken(t) for name, c in converters.items()}
        for call in calls.values():
            for _ in range(50):
                call()
        timings = {name: [] for name in calls}
        # Interleave measurements so each version sees comparable system load.
        for repeat in range(args.repeats):
            names = list(calls) if repeat % 2 else list(reversed(calls))
            for name in names:
                timings[name].append(timeit.timeit(calls[name], number=args.iterations) * 1e6 / args.iterations)
        row = {name + "_median_us": round(median(values), 3) for name, values in timings.items()}
        if "baseline" in timings:
            row["speedup"] = round(median(timings["baseline"]) / median(timings["current"]), 3)
        row["characters"] = len(text)
        findings[label] = row
    print(
        json.dumps(
            {
                "iterations": args.iterations,
                "repeats": args.repeats,
                "scope": "warm text normalization only; excludes model inference",
                "cases": findings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
