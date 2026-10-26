import json
from pathlib import Path


OUTPUT_PATH = Path("Three Model Comparison.ipynb")


def source_lines(text):
    return text.strip("\n").splitlines(keepends=True)


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


cells = [
    markdown(
        """
# Comparison of Three Text Classification Models

This notebook compares three approaches for **single-label, 24-class disease
classification**:

1. TF-IDF + Logistic Regression
2. Fine-tuned BERT
3. BioBERT

All three models are evaluated using fresh predictions on the same 240
held-out samples produced by the existing seed-42 train/test split.

> The traditional model is **Logistic Regression**, not Linear Regression.

The comparison determines which configuration performs highest **for this
dataset and split**. It does not establish that one model is universally
better for medical text classification.
"""
    ),
    markdown(
        """
## 1. Dependencies and configuration

The saved BERT and BioBERT folders must already exist. The notebook uses a
GPU when a CUDA-enabled PyTorch installation is available and otherwise runs
inference on the CPU.
"""
    ),
    code(
        """
# Uncomment and run only if dependencies are missing.
# %pip install -U pandas numpy scikit-learn matplotlib seaborn torch transformers

from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
"""
    ),
    markdown(
        """
## 2. Load the dataset and reproduce the shared test split

This matches the earlier experiments: `test_size=0.2`,
`random_state=42`, and no stratification.
"""
    ),
    code(
        """
DATA_PATH = Path("Symptom2Disease.csv")
BERT_MODEL_PATH = Path("bert_symptom_model")
BIOBERT_MODEL_PATH = Path("biobert_symptom_model")

for required_path in [DATA_PATH, BERT_MODEL_PATH, BIOBERT_MODEL_PATH]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required path not found: {required_path.resolve()}")

df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
df["label"] = df["label"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()
df = df[(df["label"] != "") & (df["text"] != "")].reset_index(drop=True)
df["source_row_id"] = np.arange(len(df))

label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])
class_names = label_encoder.classes_.tolist()
num_labels = len(class_names)

if len(df) != 1200:
    raise ValueError(f"Expected 1,200 usable rows, found {len(df)}")
if num_labels != 24:
    raise ValueError(f"Expected 24 disease classes, found {num_labels}")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

if len(train_df) != 960 or len(test_df) != 240:
    raise AssertionError("Expected a 960/240 training/test split.")
if not set(train_df["source_row_id"]).isdisjoint(
    set(test_df["source_row_id"])
):
    raise AssertionError("Training and held-out test rows overlap.")

test_texts = test_df["text"].tolist()
y_true = test_df["label_id"].to_numpy()

print(f"Dataset rows: {len(df)}")
print(f"Disease classes: {num_labels}")
print(f"Training rows: {len(train_df)}")
print(f"Held-out test rows: {len(test_df)}")
"""
    ),
    markdown(
        """
## 3. Train TF-IDF + Logistic Regression

TF-IDF is fitted only on the 960 training samples.
"""
    ),
    code(
        """
tfidf_logistic_model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=20_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2_000,
                random_state=SEED,
            ),
        ),
    ]
)

tfidf_start = time.perf_counter()
tfidf_logistic_model.fit(train_df["text"], train_df["label_id"])
tfidf_training_seconds = time.perf_counter() - tfidf_start
tfidf_predictions = tfidf_logistic_model.predict(test_df["text"])

print(f"TF-IDF training time: {tfidf_training_seconds:.2f} seconds")
"""
    ),
    markdown(
        """
## 4. Load fine-tuned BERT and BioBERT

Both models must have 24 output labels. BioBERT's stored disease-name mapping
is checked against the dataset mapping. The older BERT model uses
`LABEL_0`–`LABEL_23`, which correspond to the same encoded class IDs used
during its original training.
"""
    ),
    code(
        """
def load_saved_transformer(model_path, expected_class_names, generic_labels_ok):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    if model.config.num_labels != len(expected_class_names):
        raise ValueError(
            f"{model_path} has {model.config.num_labels} labels; "
            f"expected {len(expected_class_names)}."
        )

    stored_labels = [
        model.config.id2label[index]
        if index in model.config.id2label
        else model.config.id2label[str(index)]
        for index in range(model.config.num_labels)
    ]
    generic_labels = [f"LABEL_{index}" for index in range(model.config.num_labels)]

    if stored_labels != expected_class_names:
        if not (generic_labels_ok and stored_labels == generic_labels):
            raise ValueError(
                f"{model_path} label mapping does not match the dataset."
            )

    return tokenizer, model


bert_tokenizer, bert_model = load_saved_transformer(
    BERT_MODEL_PATH,
    class_names,
    generic_labels_ok=True,
)
biobert_tokenizer, biobert_model = load_saved_transformer(
    BIOBERT_MODEL_PATH,
    class_names,
    generic_labels_ok=False,
)

print("Fine-tuned BERT and BioBERT loaded successfully.")
"""
    ),
    markdown(
        """
## 5. Generate fresh predictions on the same 240 samples
"""
    ),
    code(
        """
def predict_transformer_in_batches(
    model,
    tokenizer,
    texts,
    batch_size=16,
    max_length=128,
):
    predictions = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {
            name: value.to(device)
            for name, value in encoded.items()
        }

        with torch.inference_mode():
            logits = model(**encoded).logits

        predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return np.asarray(predictions)


bert_predictions = predict_transformer_in_batches(
    bert_model,
    bert_tokenizer,
    test_texts,
)
biobert_predictions = predict_transformer_in_batches(
    biobert_model,
    biobert_tokenizer,
    test_texts,
)

prediction_sets = {
    "TF-IDF + Logistic Regression": tfidf_predictions,
    "Fine-tuned BERT": bert_predictions,
    "BioBERT": biobert_predictions,
}

if not len(tfidf_predictions) == len(bert_predictions) == len(
    biobert_predictions
) == len(test_df):
    raise AssertionError("All three models must predict the same test rows.")

if not all(
    len(predictions) == len(test_df)
    for predictions in prediction_sets.values()
):
    raise AssertionError("Every model must predict the same 240 test samples.")

for model_name, predictions in prediction_sets.items():
    if not np.isin(predictions, np.arange(num_labels)).all():
        raise AssertionError(f"{model_name} produced an invalid class ID.")

print("Fresh predictions completed for all three models.")
"""
    ),
    markdown(
        """
## 6. Calculate identical metrics for all models
"""
    ),
    code(
        """
METRIC_COLUMNS = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted Precision",
    "Weighted Recall",
    "Weighted F1",
]


def calculate_metrics(model_name, labels, predictions):
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(labels, predictions) * 100,
        "Macro Precision": macro_precision * 100,
        "Macro Recall": macro_recall * 100,
        "Macro F1": macro_f1 * 100,
        "Weighted Precision": weighted_precision * 100,
        "Weighted Recall": weighted_recall * 100,
        "Weighted F1": weighted_f1 * 100,
    }


comparison_df = pd.DataFrame(
    [
        calculate_metrics(model_name, y_true, predictions)
        for model_name, predictions in prediction_sets.items()
    ]
)
comparison_df[METRIC_COLUMNS] = comparison_df[METRIC_COLUMNS].round(2)

if not comparison_df[METRIC_COLUMNS].apply(
    lambda column: column.between(0, 100).all()
).all():
    raise AssertionError("All metric values must be between 0 and 100.")

ranked_comparison_df = comparison_df.sort_values(
    "Weighted F1",
    ascending=False,
).reset_index(drop=True)
ranked_comparison_df.insert(
    0,
    "Rank",
    np.arange(1, len(ranked_comparison_df) + 1),
)

best_by_metric = comparison_df[METRIC_COLUMNS].max()
difference_df = comparison_df[["Model"]].copy()
for metric in METRIC_COLUMNS:
    difference_df[f"{metric} gap from best"] = (
        comparison_df[metric] - best_by_metric[metric]
    ).round(2)

display(ranked_comparison_df)
display(difference_df)
"""
    ),
    markdown(
        """
## 7. Export tables and class-level reports
"""
    ),
    code(
        """
ranked_comparison_df.to_csv("three_model_comparison.csv", index=False)
difference_df.to_csv("three_model_differences.csv", index=False)

report_outputs = {
    "TF-IDF + Logistic Regression": (
        tfidf_predictions,
        "tfidf_three_model_classification_report.csv",
    ),
    "Fine-tuned BERT": (
        bert_predictions,
        "bert_three_model_classification_report.csv",
    ),
    "BioBERT": (
        biobert_predictions,
        "biobert_three_model_classification_report.csv",
    ),
}

for model_name, (predictions, output_path) in report_outputs.items():
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            predictions,
            labels=np.arange(num_labels),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(output_path)
    print(f"Saved {model_name} report to {output_path}")
"""
    ),
    markdown(
        """
## 8. Visual comparison
"""
    ),
    code(
        """
plot_df = comparison_df.melt(
    id_vars="Model",
    value_vars=METRIC_COLUMNS,
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=plot_df,
    x="Metric",
    y="Score",
    hue="Model",
    palette=["#1f77b4", "#697386", "#2ca02c"],
)
ax.set_title(
    "Comparison of Three Disease-Classification Models "
    "on the Same Test Set"
)
ax.set_xlabel("")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 105)
for bar_container in ax.containers:
    ax.bar_label(
        bar_container,
        fmt="%.2f",
        padding=2,
        fontsize=8,
    )
plt.xticks(rotation=25, ha="right")
plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3)
plt.tight_layout()
plt.savefig("three_model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
## 9. Thesis-ready interpretation

The following wording reports the observed result without treating one model
as universally superior.
"""
    ),
    code(
        """
best_model = ranked_comparison_df.iloc[0]
second_model = ranked_comparison_df.iloc[1]
weighted_f1_gap = best_model["Weighted F1"] - second_model["Weighted F1"]

print(
    f"{best_model['Model']} achieved the highest performance for this dataset and split, "
    f"with {best_model['Accuracy']:.2f}% accuracy and "
    f"{best_model['Weighted F1']:.2f}% weighted F1. Its weighted F1 was "
    f"{weighted_f1_gap:.2f} percentage points above the second-ranked model."
)
print(
    "This finding is specific to the 1,200-sample Symptom2Disease dataset, "
    "the seed-42 split, and the configurations evaluated here."
)
print(
    "Transformer-based models such as fine-tuned BERT and BioBERT may benefit "
    "from larger and more diverse datasets, further hyperparameter tuning, "
    "and medical text that better matches their pretrained representations. "
    "However, improved performance under those conditions must be confirmed "
    "through future experiments rather than assumed."
)
"""
    ),
    markdown(
        """
## Files produced

- `three_model_comparison.csv`
- `three_model_differences.csv`
- `tfidf_three_model_classification_report.csv`
- `bert_three_model_classification_report.csv`
- `biobert_three_model_classification_report.csv`
- `three_model_comparison.png`
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"three-model-cell-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT_PATH.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=True) + "\n",
    encoding="utf-8",
)
print(f"Created {OUTPUT_PATH.resolve()}")
