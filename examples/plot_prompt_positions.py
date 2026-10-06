"""Plot a prompt-position report: uv run --no-project --with matplotlib FILE REPORT PNG."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    development = report["summaries"]["development"]
    held_out = report["summaries"]["held_out"]
    thresholds = report.get("decision_thresholds", {"threshold": 0.8, "margin": 0.2})
    offsets = [row["offset"] for row in held_out]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    fig.suptitle("Gemma-4 E4B: answer-choice logits before generation", fontsize=16)
    axes[0].plot(offsets, [100 * s["accuracy"] for s in development], "--", color="#8493a2", label="Development")
    axes[0].plot(offsets, [100 * s["accuracy"] for s in held_out], "o-", color="#087f8c", label="Held-out")
    axes[0].set(ylabel="Correct choices (%)", ylim=(0, 105))
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].annotate(
        "After user newline",
        (-3, held_out[-4]["accuracy"] * 100),
        xytext=(-8, 96),
        arrowprops={"arrowstyle": "->", "color": "#3b4650"},
    )
    axes[1].semilogy(offsets, [max(s["median_label_mass"], 1e-12) for s in held_out], "o-", color="#5e4b8b")
    axes[1].set(ylabel="Median vocabulary mass\non answer-choice letters", ylim=(1e-12, 2))
    axes[1].axhline(1, color="#b7bdc4", linestyle="--", linewidth=0.7)
    accepted_correct = [100 * s["threshold_correct"] / s["n"] for s in held_out]
    accepted_wrong = [100 * s["threshold_wrong"] / s["n"] for s in held_out]
    unaccepted = [100 * (s["n"] - s["threshold_passes"]) / s["n"] for s in held_out]
    axes[2].bar(offsets, accepted_correct, color="#2a9d8f", label="Threshold passes, correct")
    axes[2].bar(offsets, accepted_wrong, bottom=accepted_correct, color="#d85a4e", label="Threshold passes, wrong")
    axes[2].bar(
        offsets,
        unaccepted,
        bottom=[a + b for a, b in zip(accepted_correct, accepted_wrong, strict=True)],
        color="#e4e7ea",
        label="Below threshold",
    )
    axes[2].set(
        ylabel="Held-out prompts (%)",
        ylim=(0, 105),
        xlabel="Position relative to final prompt token (0 predicts first output token)",
    )
    axes[2].legend(loc="upper left", fontsize=8, frameon=False)
    for ax in axes:
        ax.axvline(0, color="#1b2838", linestyle=":", linewidth=1)
        ax.grid(axis="y", color="#d6dade", alpha=0.5, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[2].set_xticks(offsets)
    fig.supxlabel(
        f"Q4_0 GGUF; {report['base_cases']} text prompts x 2 option orders; "
        f"{held_out[0]['n'] // 2} held-out prompts x 2 orders. "
        f"Threshold {thresholds['threshold']}, margin {thresholds['margin']}.\n"
        "Small hand-labelled study; prompts and model evaluated without native output generation.",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".svg"))
    print(args.output)


if __name__ == "__main__":
    main()
