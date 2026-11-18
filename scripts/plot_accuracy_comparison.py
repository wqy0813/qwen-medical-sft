import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "results" / "final_accuracy_results_20261117.csv"
OUT = ROOT / "reports" / "figures" / "accuracy"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["qwen25", "qwen35"]
MODEL_LABELS = {
    "qwen25": "Qwen2.5-3B",
    "qwen35": "Qwen3.5-4B",
}
DATASETS = {
    "medmcqa": {
        "title": "MedMCQA Accuracy Comparison",
        "subtitle": "Official Validation · 4,183 questions",
        "filename": "medmcqa_accuracy_comparison.png",
        "note": "",
    },
    "medexpqa": {
        "title": "MedExpQA Accuracy Comparison",
        "subtitle": "Full independent blind test · 618 questions",
        "filename": "medexpqa_accuracy_comparison.png",
        "note": "",
    },
}


def load_results():
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_dataset(rows, dataset, config):
    selected = {
        row["model"]: row
        for row in rows
        if row["dataset"] == dataset
    }

    base_values = [
        float(selected[f"{model}_base"]["report_accuracy"]) * 100
        for model in MODEL_ORDER
    ]
    sft_values = [
        float(selected[f"{model}_sft"]["report_accuracy"]) * 100
        for model in MODEL_ORDER
    ]

    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    base_bars = ax.bar(
        x - width / 2,
        base_values,
        width,
        label="Base",
    )
    sft_bars = ax.bar(
        x + width / 2,
        sft_values,
        width,
        label="SFT",
    )

    fig.suptitle(config["title"], fontsize=16, y=0.98)
    ax.set_title(
        config["subtitle"] + " · Normalized Accuracy",
        fontsize=11,
        pad=10,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True)

    for bars in (base_bars, sft_bars):
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.2f}%",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    if config["note"]:
        fig.text(
            0.5,
            0.025,
            config["note"],
            ha="center",
            va="bottom",
            fontsize=8,
            wrap=True,
        )
        fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.17)
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 0.92))

    destination = OUT / config["filename"]
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"{dataset}: {destination}")
    print("  Base:", [f"{value:.2f}%" for value in base_values])
    print("  SFT: ", [f"{value:.2f}%" for value in sft_values])


def main():
    rows = load_results()
    for dataset, config in DATASETS.items():
        draw_dataset(rows, dataset, config)

    print("Accuracy figures generated:", len(DATASETS))
    print("Output directory:", OUT)


if __name__ == "__main__":
    main()
