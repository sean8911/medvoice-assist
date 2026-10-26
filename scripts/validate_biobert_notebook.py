import json
import sys
from pathlib import Path


NOTEBOOK_PATH = Path("BioBERT.ipynb")

REQUIRED_SNIPPETS = {
    "BioBERT checkpoint": "dmis-lab/biobert-base-cased-v1.2",
    "reproducibility seed": "SEED = 42",
    "outer test size": "test_size=0.2",
    "outer split seed": "random_state=SEED",
    "internal validation split": "validation_df",
    "validation macro F1 selection": "metric_for_best_model=\"macro_f1\"",
    "accuracy metric": "\"Accuracy\"",
    "macro precision metric": "\"Macro Precision\"",
    "macro recall metric": "\"Macro Recall\"",
    "macro F1 metric": "\"Macro F1\"",
    "weighted precision metric": "\"Weighted Precision\"",
    "weighted recall metric": "\"Weighted Recall\"",
    "weighted F1 metric": "\"Weighted F1\"",
    "comparison export": "biobert_model_comparison.csv",
    "difference export": "biobert_vs_tfidf_difference.csv",
    "classification report export": "biobert_classification_report.csv",
    "confusion matrix export": "biobert_confusion_matrix.png",
    "comparison chart export": "biobert_vs_tfidf_comparison.png",
    "saved model folder": "biobert_symptom_model",
    "same test length check": "len(tfidf_predictions) == len(biobert_predictions)",
    "outer/test row separation": "outer_train_ids.isdisjoint(test_ids)",
    "24-class validation": "num_labels != 24",
    "960/240 split validation": "len(outer_train_df) != 960 or len(test_df) != 240",
    "dataset-specific conclusion": "for this dataset and split",
}


def fail(message):
    print(f"FAIL: {message}")
    raise SystemExit(1)


if not NOTEBOOK_PATH.exists():
    fail(f"{NOTEBOOK_PATH} does not exist")

try:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    fail(f"invalid notebook JSON: {exc}")

cells = notebook.get("cells", [])
if not cells:
    fail("notebook has no cells")
if any(not cell.get("id") for cell in cells):
    fail("one or more notebook cells are missing stable IDs")

all_source = "\n".join(
    "".join(cell.get("source", []))
    for cell in cells
)
code_source = "\n".join(
    "".join(cell.get("source", []))
    for cell in cells
    if cell.get("cell_type") == "code"
)

missing = [
    label
    for label, snippet in REQUIRED_SNIPPETS.items()
    if snippet not in all_source
]
if missing:
    fail("missing required content: " + ", ".join(missing))

if "LinearRegression" in code_source:
    fail("notebook incorrectly uses LinearRegression")

if "LogisticRegression" not in code_source:
    fail("notebook does not use LogisticRegression")

if "eval_dataset=test" in code_source.replace(" ", ""):
    fail("held-out test set appears to be used as the Trainer evaluation set")

if "load_best_model_at_end=True" not in code_source:
    fail("BioBERT does not load the best validation checkpoint")

if "greater_is_better=True" not in code_source:
    fail("macro F1 selection direction is not explicitly configured")

if notebook.get("nbformat") != 4:
    fail("notebook must use nbformat 4")

print(
    "PASS: BioBERT.ipynb contains the required fair-comparison structure, "
    "metrics, safeguards, and exports."
)
