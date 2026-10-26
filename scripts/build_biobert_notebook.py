import json
from pathlib import Path


OUTPUT_PATH = Path("BioBERT.ipynb")


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
# BioBERT vs TF-IDF + Logistic Regression

This notebook fine-tunes **BioBERT** for single-label, 24-class disease
classification and compares it with **TF-IDF + Logistic Regression**.

> The traditional model is Logistic Regression, not Linear Regression.

## Fair-comparison rules

- Both final models use the same 960 outer-training rows.
- Both models are evaluated on the same unchanged 240-row held-out test set.
- BioBERT selects its epoch count using a validation subset created only from
  the outer-training rows.
- The held-out test set is evaluated once after training and is not used for
  model selection.
- The result only establishes which model performs better **for this dataset
  and split**. It does not prove universal superiority for medical chatbots.

BioBERT checkpoint:
[`dmis-lab/biobert-base-cased-v1.2`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
"""
    ),
    markdown(
        """
## 1. Dependencies

If the import cell reports missing packages, run the installation cell once,
restart the kernel if requested, and continue from the top.

BioBERT training is much faster with an NVIDIA GPU. CPU execution is
supported but may take substantially longer.
"""
    ),
    code(
        """
# Uncomment and run only if your environment is missing dependencies.
# %pip install -U pandas numpy scikit-learn matplotlib seaborn torch transformers accelerate

# Optional NVIDIA GPU install used for this project's RTX 5060 environment:
# %pip install --upgrade --force-reinstall torch==2.12.0+cu130 --index-url https://download.pytorch.org/whl/cu130
"""
    ),
    code(
        """
from pathlib import Path
import gc
import inspect
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
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError as exc:
    raise ImportError(
        "Missing Hugging Face dependencies. Run the installation cell above, "
        "restart the kernel if needed, and run the notebook again."
    ) from exc
"""
    ),
    markdown(
        """
## 2. Reproducibility and experiment configuration

The outer split deliberately matches the existing TF-IDF and fine-tuned BERT
comparison: `test_size=0.2` and `random_state=42`, without stratification.
"""
    ),
    code(
        """
SEED = 42
CHECKPOINT = "dmis-lab/biobert-base-cased-v1.2"
MAX_LENGTH = 128
MAX_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
VALIDATION_SIZE = 0.10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 8 if torch.cuda.is_available() else 4
EVAL_BATCH_SIZE = 8 if torch.cuda.is_available() else 4
GRADIENT_ACCUMULATION_STEPS = 1 if torch.cuda.is_available() else 2

print("Device:", device)
print("BioBERT checkpoint:", CHECKPOINT)
print("Train batch size:", TRAIN_BATCH_SIZE)
print("Gradient accumulation steps:", GRADIENT_ACCUMULATION_STEPS)
if device.type == "cpu":
    print(
        "Warning: BioBERT will run on CPU. Training can take substantially "
        "longer than GPU training."
    )
"""
    ),
    markdown(
        """
## 3. Load the Symptom2Disease dataset

The dataset must contain `label` and `text` columns. The unnamed CSV index
column is removed before modelling.
"""
    ),
    code(
        """
DATA_PATH = Path("Symptom2Disease.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()

required_columns = {"label", "text"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
df["label"] = df["label"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()
df = df[(df["label"] != "") & (df["text"] != "")].reset_index(drop=True)
df["source_row_id"] = np.arange(len(df))

label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])
class_names = label_encoder.classes_.tolist()
num_labels = len(class_names)

if num_labels != 24:
    raise ValueError(f"Expected 24 disease classes, found {num_labels}")
if len(df) != 1200:
    raise ValueError(f"Expected 1,200 usable rows, found {len(df)}")

print(f"Usable rows: {len(df)}")
print(f"Disease classes: {num_labels}")
display(df.head())
"""
    ),
    markdown(
        """
## 4. Reproduce the outer train/test split

The 240-row test set is held aside here and is not passed to the BioBERT
`Trainer` during model selection.
"""
    ),
    code(
        """
outer_train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
)
outer_train_df = outer_train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

if len(outer_train_df) != 960 or len(test_df) != 240:
    raise AssertionError(
        "The outer split must contain 960 training and 240 test rows."
    )

# BioBERT uses only this internal split to choose the best epoch.
biobert_train_df, validation_df = train_test_split(
    outer_train_df,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=outer_train_df["label_id"],
)
biobert_train_df = biobert_train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

outer_train_ids = set(outer_train_df["source_row_id"])
test_ids = set(test_df["source_row_id"])
if not outer_train_ids.isdisjoint(test_ids):
    raise AssertionError("Outer training and held-out test rows overlap.")
if outer_train_ids.union(test_ids) != set(df["source_row_id"]):
    raise AssertionError("Outer split lost or duplicated dataset rows.")

print(f"Outer training rows: {len(outer_train_df)}")
print(f"Held-out test rows: {len(test_df)}")
print(f"BioBERT selection-training rows: {len(biobert_train_df)}")
print(f"BioBERT validation rows: {len(validation_df)}")
"""
    ),
    markdown(
        """
## 5. Shared metric functions

Macro metrics give every disease class equal importance. Weighted metrics
weight class scores by the number of test samples in each class.
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


def calculate_metrics(y_true, y_pred):
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Macro Precision": macro_precision * 100,
        "Macro Recall": macro_recall * 100,
        "Macro F1": macro_f1 * 100,
        "Weighted Precision": weighted_precision * 100,
        "Weighted Recall": weighted_recall * 100,
        "Weighted F1": weighted_f1 * 100,
    }


def compute_trainer_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    metrics = calculate_metrics(labels, predictions)
    return {
        "accuracy": metrics["Accuracy"] / 100,
        "macro_precision": metrics["Macro Precision"] / 100,
        "macro_recall": metrics["Macro Recall"] / 100,
        "macro_f1": metrics["Macro F1"] / 100,
        "weighted_precision": metrics["Weighted Precision"] / 100,
        "weighted_recall": metrics["Weighted Recall"] / 100,
        "weighted_f1": metrics["Weighted F1"] / 100,
    }
"""
    ),
    markdown(
        """
## 6. Train TF-IDF + Logistic Regression

This configuration matches the existing baseline notebook and is fitted on
all 960 outer-training rows.
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
tfidf_logistic_model.fit(
    outer_train_df["text"],
    outer_train_df["label_id"],
)
tfidf_training_seconds = time.perf_counter() - tfidf_start
tfidf_predictions = tfidf_logistic_model.predict(test_df["text"])
tfidf_metrics = calculate_metrics(test_df["label_id"], tfidf_predictions)

print(f"TF-IDF training time: {tfidf_training_seconds:.2f} seconds")
display(pd.DataFrame([{"Model": "TF-IDF + Logistic Regression", **tfidf_metrics}]))
"""
    ),
    markdown(
        """
## 7. Tokenize text for BioBERT

The checkpoint is downloaded from Hugging Face on the first run. A network
connection is therefore required unless it is already cached.
"""
    ),
    code(
        """
try:
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
except Exception as exc:
    raise RuntimeError(
        "BioBERT could not be downloaded. Check your internet connection or "
        "Hugging Face access, then rerun this cell."
    ) from exc


class EncodedTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        self.labels = labels.astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


selection_train_dataset = EncodedTextDataset(
    biobert_train_df["text"],
    biobert_train_df["label_id"],
)
validation_dataset = EncodedTextDataset(
    validation_df["text"],
    validation_df["label_id"],
)
full_train_dataset = EncodedTextDataset(
    outer_train_df["text"],
    outer_train_df["label_id"],
)
test_dataset = EncodedTextDataset(
    test_df["text"],
    test_df["label_id"],
)

print("BioBERT datasets tokenized.")
"""
    ),
    markdown(
        """
## 8. Select the BioBERT epoch using validation macro F1

This stage does **not** use `test_df`. The best validation epoch is selected,
then a fresh BioBERT model is trained on all 960 outer-training rows for that
number of epochs.
"""
    ),
    code(
        """
id2label = {index: label for index, label in enumerate(class_names)}
label2id = {label: index for index, label in id2label.items()}


def make_training_arguments(**overrides):
    base_arguments = {
        "output_dir": "results/biobert_selection",
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": MAX_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "logging_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "fp16": torch.cuda.is_available(),
    }
    base_arguments.update(overrides)

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        base_arguments.setdefault("eval_strategy", "epoch")
    else:
        base_arguments.setdefault("evaluation_strategy", "epoch")
    return TrainingArguments(**base_arguments)


selection_model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

selection_arguments = make_training_arguments(
    output_dir="results/biobert_selection",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)

selection_trainer_kwargs = {
    "model": selection_model,
    "args": selection_arguments,
    "train_dataset": selection_train_dataset,
    "eval_dataset": validation_dataset,
    "compute_metrics": compute_trainer_metrics,
}
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    selection_trainer_kwargs["processing_class"] = tokenizer
else:
    selection_trainer_kwargs["tokenizer"] = tokenizer

selection_trainer = Trainer(**selection_trainer_kwargs)

selection_start = time.perf_counter()
selection_trainer.train()
selection_training_seconds = time.perf_counter() - selection_start

validation_records = [
    record
    for record in selection_trainer.state.log_history
    if "eval_macro_f1" in record and "epoch" in record
]
if not validation_records:
    raise RuntimeError("No validation macro F1 records were produced.")

best_validation_record = max(
    validation_records,
    key=lambda record: record["eval_macro_f1"],
)
selected_epochs = max(1, int(round(float(best_validation_record["epoch"]))))

print("Best validation checkpoint:", selection_trainer.state.best_model_checkpoint)
print("Best validation macro F1:", selection_trainer.state.best_metric)
print("Selected final-training epochs:", selected_epochs)
"""
    ),
    markdown(
        """
## 9. Train a fresh BioBERT model on all 960 training rows

The selected epoch count comes from the internal validation split. This final
model then receives the same 960 outer-training rows used by TF-IDF.
"""
    ),
    code(
        """
del selection_trainer
del selection_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

final_model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

final_arguments_kwargs = {
    "output_dir": "results/biobert_final",
    "learning_rate": LEARNING_RATE,
    "per_device_train_batch_size": TRAIN_BATCH_SIZE,
    "per_device_eval_batch_size": EVAL_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_train_epochs": selected_epochs,
    "weight_decay": WEIGHT_DECAY,
    "logging_strategy": "epoch",
    "save_strategy": "no",
    "report_to": "none",
    "seed": SEED,
    "data_seed": SEED,
    "fp16": torch.cuda.is_available(),
}
final_training_arguments = TrainingArguments(**final_arguments_kwargs)

final_trainer_kwargs = {
    "model": final_model,
    "args": final_training_arguments,
    "train_dataset": full_train_dataset,
}
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    final_trainer_kwargs["processing_class"] = tokenizer
else:
    final_trainer_kwargs["tokenizer"] = tokenizer

final_trainer = Trainer(**final_trainer_kwargs)

final_start = time.perf_counter()
final_trainer.train()
final_training_seconds = time.perf_counter() - final_start
biobert_training_seconds = selection_training_seconds + final_training_seconds

test_prediction_output = final_trainer.predict(test_dataset)
biobert_predictions = np.argmax(test_prediction_output.predictions, axis=-1)
biobert_metrics = calculate_metrics(
    test_df["label_id"].to_numpy(),
    biobert_predictions,
)

if not len(tfidf_predictions) == len(biobert_predictions) == len(test_df):
    raise AssertionError("Both models must predict the same 240 test samples.")

MODEL_OUTPUT_DIR = Path("biobert_symptom_model")
final_trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print(f"Total BioBERT training time: {biobert_training_seconds:.2f} seconds")
print(f"Saved BioBERT model to: {MODEL_OUTPUT_DIR.resolve()}")
"""
    ),
    markdown(
        """
## 10. Compare test-set performance

Weighted F1 is the primary model-selection metric because it summarizes
precision and recall while accounting for the number of test samples in each
disease class.
"""
    ),
    code(
        """
comparison_df = pd.DataFrame(
    [
        {
            "Model": "TF-IDF + Logistic Regression",
            **tfidf_metrics,
            "Training Time (seconds)": tfidf_training_seconds,
        },
        {
            "Model": "BioBERT",
            **biobert_metrics,
            "Training Time (seconds)": biobert_training_seconds,
        },
    ]
)

numeric_columns = METRIC_COLUMNS + ["Training Time (seconds)"]
comparison_df[numeric_columns] = comparison_df[numeric_columns].round(2)

if not comparison_df[METRIC_COLUMNS].apply(
    lambda column: column.between(0, 100).all()
).all():
    raise AssertionError("All classification metrics must be between 0 and 100.")

difference_df = pd.DataFrame(
    {
        "Metric": METRIC_COLUMNS,
        "BioBERT difference (percentage points)": [
            biobert_metrics[metric] - tfidf_metrics[metric]
            for metric in METRIC_COLUMNS
        ],
    }
)
difference_df["BioBERT difference (percentage points)"] = difference_df[
    "BioBERT difference (percentage points)"
].round(2)

comparison_df.to_csv("biobert_model_comparison.csv", index=False)
difference_df.to_csv("biobert_vs_tfidf_difference.csv", index=False)

biobert_report_df = pd.DataFrame(
    classification_report(
        test_df["label_id"],
        biobert_predictions,
        labels=list(range(num_labels)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
).transpose()
biobert_report_df.to_csv("biobert_classification_report.csv")

display(comparison_df)
display(difference_df)
"""
    ),
    markdown(
        """
## 11. Visualize the comparison
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

plt.figure(figsize=(13, 6))
ax = sns.barplot(
    data=plot_df,
    x="Metric",
    y="Score",
    hue="Model",
    palette=["#1f77b4", "#2ca02c"],
)
ax.set_title("BioBERT vs TF-IDF + Logistic Regression on the Same Test Set")
ax.set_ylabel("Score (%)")
ax.set_xlabel("")
ax.set_ylim(0, 105)
plt.xticks(rotation=25, ha="right")
plt.legend(title="")
plt.tight_layout()
plt.savefig("biobert_vs_tfidf_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """
biobert_cm = confusion_matrix(
    test_df["label_id"],
    biobert_predictions,
    labels=list(range(num_labels)),
)

plt.figure(figsize=(15, 13))
sns.heatmap(
    biobert_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("BioBERT Confusion Matrix")
plt.xlabel("Predicted disease class")
plt.ylabel("True disease class")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("biobert_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
## 12. Dataset-specific conclusion

Run the following cell after training. It identifies the better-performing
model **for this dataset and split** using weighted F1. Do not generalize this
result to all medical chatbot datasets.
"""
    ),
    code(
        """
tfidf_weighted_f1 = tfidf_metrics["Weighted F1"]
biobert_weighted_f1 = biobert_metrics["Weighted F1"]

if biobert_weighted_f1 > tfidf_weighted_f1:
    winner = "BioBERT"
    difference = biobert_weighted_f1 - tfidf_weighted_f1
elif tfidf_weighted_f1 > biobert_weighted_f1:
    winner = "TF-IDF + Logistic Regression"
    difference = tfidf_weighted_f1 - biobert_weighted_f1
else:
    winner = "Tie"
    difference = 0.0

if winner == "Tie":
    print(
        "Both models achieved the same weighted F1 for this dataset and split."
    )
else:
    print(
        f"{winner} achieved the higher weighted F1 for this dataset and split "
        f"by {difference:.2f} percentage points."
    )

print(
    "This result is specific to the Symptom2Disease dataset, the seed-42 "
    "split, and the configurations used in this notebook."
)
"""
    ),
    markdown(
        """
## Files produced after a complete run

- `biobert_model_comparison.csv`
- `biobert_vs_tfidf_difference.csv`
- `biobert_classification_report.csv`
- `biobert_confusion_matrix.png`
- `biobert_vs_tfidf_comparison.png`
- `biobert_symptom_model/`

The notebook intentionally contains no pre-filled BioBERT score. The winner
is not known until BioBERT training and evaluation finish successfully.
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"biobert-cell-{index:02d}"

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
