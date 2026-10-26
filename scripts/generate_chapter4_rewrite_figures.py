from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


OUTPUT_DIR = Path("chapter4_rewrite_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

BLUE = "#377ba8"
GRAY = "#7b8495"
GREEN = "#3a9638"
EDGE = "#37474f"


def save(name):
    plt.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def add_box(ax, x, y, width, text, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.72,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=EDGE,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + 0.36,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=8.2,
        fontweight="bold",
    )


def add_arrow(ax, start, end, y):
    ax.annotate(
        "",
        xy=(end, y),
        xytext=(start, y),
        arrowprops={"arrowstyle": "->", "color": EDGE, "lw": 1.4},
    )


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 4.1))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(0.1, 2.95, "Baseline", fontsize=11, fontweight="bold", va="center")
    ax.text(0.1, 1.05, "Improved", fontsize=11, fontweight="bold", va="center")

    base_x = [1.6, 4.2, 6.8, 9.4]
    base_labels = [
        "Raw audio",
        "Whisper",
        "Transcript",
        "Text classifier",
    ]
    improved_x = [1.3, 3.35, 5.4, 7.45, 9.5]
    improved_labels = [
        "Raw audio",
        "Audio\npreprocessing",
        "Whisper",
        "Transcript",
        "Text classifier",
    ]

    for row_y, xs, labels in [
        (2.58, base_x, base_labels),
        (0.68, improved_x, improved_labels),
    ]:
        width = 1.45
        for index, (x, label) in enumerate(zip(xs, labels)):
            color = GREEN if row_y < 2 and index == 1 else BLUE
            add_box(ax, x, row_y, width, label, color)
            if index < len(labels) - 1:
                add_arrow(
                    ax,
                    x + width + 0.08,
                    xs[index + 1] - 0.08,
                    row_y + 0.36,
                )

    ax.text(
        6,
        3.76,
        "Baseline and Improved Speech-Processing Pipelines",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save("figure_4_1_pipeline.png")


def grouped_chart(name, title, baseline, improved, limit):
    labels = ["WER", "CER"]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    baseline_bars = ax.bar(
        x - width / 2,
        baseline,
        width,
        label="Baseline",
        color=GRAY,
    )
    improved_bars = ax.bar(
        x + width / 2,
        improved,
        width,
        label="Improved",
        color=BLUE,
    )

    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Error rate (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, limit)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.bar_label(baseline_bars, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(improved_bars, fmt="%.2f", padding=3, fontsize=9)
    plt.tight_layout()
    save(name)


pipeline_figure()
grouped_chart(
    "figure_4_2_real_audio.png",
    "Real Recording Transcription Error Rates",
    [11.11, 9.52],
    [0.00, 0.00],
    14,
)
grouped_chart(
    "figure_4_3_synthetic_noise.png",
    "Mean Error Rates Across Synthetic-Noise Conditions",
    [8.33, 8.93],
    [5.56, 4.76],
    11,
)
