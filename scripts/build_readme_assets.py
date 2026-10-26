"""Collate retained classifier reports and draw the README comparison figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "3 Models After Training" / "outputs"
ASSETS = ROOT / "docs" / "assets"
MODELS = (
    ("TF-IDF + Logistic Regression", "tfidf_logistic_regression_classification_report.csv"),
    ("Fine-tuned BERT", "fine_tuned_bert_classification_report.csv"),
    ("BioBERT", "biobert_classification_report.csv"),
)
METRICS = (
    "Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
    "Weighted Precision", "Weighted Recall", "Weighted F1",
)
DISPLAY = ("TF-IDF + LR", "Fine-tuned BERT", "BioBERT")
COLOURS = {"Accuracy": "#087F8C", "Weighted F1": "#254B73", "Macro F1": "#A86619"}


def read_expanded_reports() -> list[dict[str, float | str]]:
    rows = []
    for name, filename in MODELS:
        with (EXPANDED / filename).open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        label_key = next(iter(records[0]))
        report = {item[label_key]: item for item in records}
        weighted, macro = report["weighted avg"], report["macro avg"]
        if int(float(weighted["support"])) != 24695:
            raise ValueError(f"Unexpected test support in {filename}")
        row = {"Model": name, "Accuracy": float(report["accuracy"]["f1-score"]) * 100}
        for prefix, source in (("Macro", macro), ("Weighted", weighted)):
            for title, key in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1-score")):
                row[f"{prefix} {title}"] = float(source[key]) * 100
        rows.append(row)
    return rows


def read_preliminary() -> list[dict[str, float | str]]:
    with (ROOT / "three_model_comparison.csv").open(newline="", encoding="utf-8") as handle:
        lookup = {row["Model"]: row for row in csv.DictReader(handle)}
    return [
        {"Model": name, **{key: float(lookup[name][key]) for key in METRICS}}
        for name, _ in MODELS
    ]


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    expanded = read_expanded_reports()
    preliminary = read_preliminary()
    summary = EXPANDED / "three_model_comparison.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Model", *METRICS])
        writer.writeheader()
        for row in expanded:
            writer.writerow({key: value if key == "Model" else f"{value:.8f}" for key, value in row.items()})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.6), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    chart_metrics = ("Accuracy", "Weighted F1", "Macro F1")
    for axis, rows, title, note in (
        (axes[0], preliminary, "Preliminary: Symptom2Disease", "24 classes | 240 test descriptions"),
        (axes[1], expanded, "Expanded: disease-diagnosis dataset", "773 labels overall | 24,695 test rows"),
    ):
        base = np.arange(3, dtype=float)
        for offset, metric in zip((-0.23, 0.0, 0.23), chart_metrics):
            values = [float(row[metric]) for row in rows]
            bars = axis.barh(base + offset, values, height=0.20,
                             color=COLOURS[metric], label=metric, zorder=3)
            for bar, value in zip(bars, values):
                axis.text(value + 0.9, bar.get_y() + bar.get_height() / 2,
                          f"{value:.2f}", ha="left", va="center", fontsize=11.5, color="#17383E")
        axis.set_yticks(base, DISPLAY)
        axis.invert_yaxis()
        axis.set_xlim(0, 109)
        axis.set_xticks(range(0, 101, 20))
        axis.grid(axis="x", color="#E3EAED", linewidth=0.8, zorder=0)
        axis.tick_params(axis="both", length=0, labelcolor="#355057")
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(title, loc="left", fontsize=17, weight="bold", color="#123B40", pad=32)
        axis.text(0, 1.04, note, transform=axis.transAxes, color="#50656B", fontsize=12)
        axis.set_xlabel("Score (%)", color="#355057", labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.18, 0.94),
               ncol=3, frameon=False, fontsize=12)
    fig.suptitle("Reported classifier comparison", x=0.18, ha="left", y=0.989,
                 fontsize=23, weight="bold", color="#123B40")
    fig.text(0.18, 0.015, "Retained split results, not independent patient-level or clinical validation.",
             ha="left", fontsize=11, color="#50656B")
    fig.subplots_adjust(left=0.18, right=0.97, top=0.82, bottom=0.09, hspace=0.65)
    destination = ASSETS / "classifier-comparison.png"
    fig.savefig(destination, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated {destination.relative_to(ROOT)} and {summary.relative_to(ROOT)} from retained reports.")
    for row in expanded:
        print(row)


if __name__ == "__main__":
    main()
