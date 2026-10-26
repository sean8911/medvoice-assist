import json
from pathlib import Path


NOTEBOOK_PATH = Path("Three Model Comparison.ipynb")

REQUIRED_SNIPPETS = {
    "TF-IDF model": "TF-IDF + Logistic Regression",
    "fine-tuned BERT model": "Fine-tuned BERT",
    "BioBERT model": "BioBERT",
    "seed": "SEED = 42",
    "outer split": "test_size=0.2",
    "split seed": "random_state=SEED",
    "TF-IDF training": "tfidf_logistic_model.fit",
    "BERT path": "bert_symptom_model",
    "BioBERT path": "biobert_symptom_model",
    "fresh prediction function": "predict_transformer_in_batches",
    "prediction count check": "len(tfidf_predictions)",
    "accuracy metric": "\"Accuracy\"",
    "macro precision": "\"Macro Precision\"",
    "macro recall": "\"Macro Recall\"",
    "macro F1": "\"Macro F1\"",
    "weighted precision": "\"Weighted Precision\"",
    "weighted recall": "\"Weighted Recall\"",
    "weighted F1": "\"Weighted F1\"",
    "ranking": "ranked_comparison_df",
    "comparison export": "three_model_comparison.csv",
    "difference export": "three_model_differences.csv",
    "TF-IDF report": "tfidf_three_model_classification_report.csv",
    "BERT report": "bert_three_model_classification_report.csv",
    "BioBERT report": "biobert_three_model_classification_report.csv",
    "chart export": "three_model_comparison.png",
    "dataset limitation": "for this dataset and split",
    "larger dataset caveat": "larger and more diverse datasets",
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
if "pd.read_csv(\"biobert_model_comparison.csv\")" in code_source:
    fail("notebook copies BioBERT summary metrics instead of predicting")
if "pd.read_csv(\"tfidf_vs_finetuned_bert.csv\")" in code_source:
    fail("notebook copies BERT summary metrics instead of predicting")
if notebook.get("nbformat") != 4:
    fail("notebook must use nbformat 4")

print(
    "PASS: Three Model Comparison.ipynb contains the required shared-split "
    "prediction, evaluation, export, and interpretation structure."
)
