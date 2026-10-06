"""Compare isolated and four-sequence native logits using the saved routing corpus.

No endpoint requests, sampled tokens or production configuration changes.
Build probe_parallel_logits.cpp against the matching libllama first.
"""

import argparse
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time


def best(result: dict) -> str:
    return max(result["candidates"], key=lambda c: c["conditional_probability"])["label"]


def accepted(result: dict, threshold: float, margin: float) -> bool:
    scores = sorted((c["conditional_probability"] for c in result["candidates"]), reverse=True)
    return scores[0] >= threshold and scores[0] - scores[1] >= margin


def compare(reference: dict, candidate: dict) -> dict:
    a = {c["label"]: c["conditional_probability"] for c in reference["candidates"]}
    b = {c["label"]: c["conditional_probability"] for c in candidate["candidates"]}
    return {
        "same_choice": best(reference) == best(candidate),
        "max_probability_difference": max(abs(a[label] - b[label]) for label in a),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, default=Path("models/gemma4-benchmark/gemma-4-E4B-it-Q4_0.gguf"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("docs/benchmarks/gemma4-prompt-positions-2026-10-06.details.json.gz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.corpus, "rt") as f:
        corpus = json.load(f)
    rows = corpus["rows"]
    if len(rows) % 4:
        raise ValueError("This four-slot experiment requires a corpus divisible by four")
    inputs = [{k: row[k] for k in ["id", "prompt", "labels"]} for row in rows]
    groups = [{"id": "serial-" + row["id"], "requests": [row]} for row in inputs]
    for i in range(0, len(inputs), 4):
        groups.append({"id": f"parallel-{i // 4}", "requests": inputs[i : i + 4]})
        groups.append({"id": f"reordered-{i // 4}", "requests": inputs[i : i + 4][::-1]})
    # Equal lengths force four output rows in the same decode, with different label-array orders.
    copies = [
        {**inputs[0], "id": f"copy-{i}", "labels": inputs[0]["labels"][i:] + inputs[0]["labels"][:i]} for i in range(4)
    ]
    groups.append({"id": "simultaneous-outputs", "requests": copies})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_path = args.output.with_suffix(".inputs.jsonl")
    raw_path = args.output.with_suffix(".native.jsonl")
    log_path = args.output.with_suffix(".native.log")
    input_path.write_text("".join(json.dumps(group) + "\n" for group in groups))
    command = [str(args.probe.resolve()), str(args.gguf.resolve()), "99", "8", "4"]
    print(f"Evaluating {len(groups)} groups with one model and one four-sequence context", flush=True)
    with input_path.open() as source, raw_path.open("w") as output, log_path.open("w") as log:
        process = subprocess.Popen(command, stdin=source, stdout=output, stderr=log)
        try:
            while process.poll() is None:
                time.sleep(5)
                with raw_path.open() as progress:
                    print(f"Completed {sum(1 for _ in progress)}/{len(groups)} groups", flush=True)
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise
    results = [json.loads(line) for line in raw_path.read_text().splitlines()]
    if process.returncode or len(results) != len(groups) or any("error" in r for r in results):
        raise RuntimeError(f"Native probe failed; inspect {raw_path} and {log_path}")
    isolated = {r["results"][0]["id"]: r["results"][0] for r in results if r["id"].startswith("serial-")}
    comparisons = []
    accuracy = {}
    for mode in ["parallel", "reordered"]:
        candidate_results = {
            r["id"]: r for group in results if group["id"].startswith(mode + "-") for r in group["results"]
        }
        for row in rows:
            serial_pass = accepted(isolated[row["id"]], row["threshold"], row["margin"])
            batched_pass = accepted(candidate_results[row["id"]], row["threshold"], row["margin"])
            comparisons.append(
                {
                    "id": row["id"],
                    "mode": mode,
                    **compare(isolated[row["id"]], candidate_results[row["id"]]),
                    "serial_accepted": serial_pass,
                    "batched_accepted": batched_pass,
                    "confidence_gate_changed": serial_pass != batched_pass,
                }
            )
        accuracy[mode] = {
            split: {
                "n": sum(row["split"] == split for row in rows),
                "correct": sum(
                    row["label_options"][best(candidate_results[row["id"]])] == row["expected"]
                    for row in rows
                    if row["split"] == split
                ),
            }
            for split in ["development", "held_out"]
        }
    simultaneous = next(r for r in results if r["id"] == "simultaneous-outputs")
    simultaneous_comparisons = [compare(isolated[inputs[0]["id"]], result) for result in simultaneous["results"]]
    serial_accuracy = {
        split: {
            "n": sum(row["split"] == split for row in rows),
            "correct": sum(
                row["label_options"][best(isolated[row["id"]])] == row["expected"]
                for row in rows
                if row["split"] == split
            ),
        }
        for split in ["development", "held_out"]
    }
    report = {
        "date": datetime.now(UTC).isoformat(),
        "model": str(args.gguf),
        "backend": corpus["backend"],
        "corpus": str(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "probe_binary_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "command": command,
        "slots": 4,
        "context_tokens_per_slot": 2048,
        "batch_capacity": 512,
        "prefill_chunk_per_sequence": 32,
        "group_evaluations": len(groups),
        "request_evaluations": sum(len(group["requests"]) for group in groups),
        "generated_tokens": sum(group["generated_tokens"] for group in results),
        "model_loads": 1,
        "context_instances": 1,
        "max_sequences_per_call": max(group["max_sequences_per_call"] for group in results),
        "max_logit_rows_per_call": max(group["max_logit_rows_per_call"] for group in results),
        "serial_accuracy": serial_accuracy,
        "batched_accuracy": accuracy,
        "comparisons": comparisons,
        "choice_disagreements": sum(not result["same_choice"] for result in comparisons),
        "confidence_gate_changes": sum(result["confidence_gate_changed"] for result in comparisons),
        "max_probability_difference": max(result["max_probability_difference"] for result in comparisons),
        "simultaneous_output_comparisons": simultaneous_comparisons,
        "groups": results,
        "limitations": [
            "Text-only prefill and final-logit extraction; mixed generation and multimodal input remain untested.",
            "Static groups validate isolation and output mapping; continuous batching remains unimplemented.",
            "One owner calls decode and copies outputs; no concurrent calls on the same context.",
            "Batch shape can change floating-point results. Exact probability equality is not required.",
            "Times include vocabulary statistics and shared-GPU activity; no production speedup is established.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "request_evaluations",
                    "generated_tokens",
                    "max_sequences_per_call",
                    "max_logit_rows_per_call",
                    "choice_disagreements",
                    "max_probability_difference",
                    "serial_accuracy",
                    "batched_accuracy",
                ]
            },
            indent=2,
        )
    )
    print("Saved", args.output)


if __name__ == "__main__":
    main()
