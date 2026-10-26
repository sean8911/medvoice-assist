#!/usr/bin/env python
"""Train and compare TF-IDF, BERT, and BioBERT disease classifiers."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


MODEL_DISPLAY_NAMES = {
    "tfidf": "TF-IDF + Logistic Regression",
    "bert": "Fine-tuned BERT",
    "biobert": "BioBERT",
}

MODEL_REPORT_FILES = {
    "tfidf": "tfidf_logistic_regression_classification_report.csv",
    "bert": "fine_tuned_bert_classification_report.csv",
    "biobert": "biobert_classification_report.csv",
}

METRIC_COLUMNS = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted Precision",
    "Weighted Recall",
    "Weighted F1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TF-IDF + Logistic Regression, fine-tuned BERT, and BioBERT "
            "on the disease diagnosis dataset, then export comparison metrics."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/disease-diagnosis-dataset",
        help="Folder containing the cleaned disease diagnosis CSV files.",
    )
    parser.add_argument(
        "--train-file",
        default="disease_diagnosis_train_cleaned.csv",
        help="Training CSV file name inside --dataset-dir.",
    )
    parser.add_argument(
        "--test-file",
        default="disease_diagnosis_test_cleaned.csv",
        help="Test CSV file name inside --dataset-dir.",
    )
    parser.add_argument(
        "--text-column",
        default="text_cleaned",
        help="Column containing cleaned symptom text.",
    )
    parser.add_argument(
        "--label-column",
        default="diagnosis",
        help="Column containing the disease diagnosis label.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder where models, reports, metrics, and plots are saved.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "tfidf", "bert", "biobert"],
        help="Models to train. Use 'all' for all three models.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional row limit for quick server smoke tests.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional test row limit for quick server smoke tests.",
    )

    parser.add_argument(
        "--bert-model-name",
        default="bert-base-uncased",
        help="Hugging Face checkpoint used for the fine-tuned BERT model.",
    )
    parser.add_argument(
        "--biobert-model-name",
        default="dmis-lab/biobert-base-cased-v1.2",
        help="Hugging Face checkpoint used for the BioBERT model.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--fp16",
        choices=["auto", "true", "false"],
        default="auto",
        help="Use fp16 transformer training. 'auto' enables it only on CUDA.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=2,
        help="Worker count used by the Hugging Face Trainer dataloaders.",
    )

    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--tfidf-max-iter", type=int, default=1000)

    return parser.parse_args()


def normalize_model_selection(model_args: list[str]) -> list[str]:
    if "all" in model_args:
        return ["tfidf", "bert", "biobert"]
    selected = []
    for model_name in model_args:
        if model_name not in selected:
            selected.append(model_name)
    return selected


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clean_text_label_frame(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    split_name: str,
) -> pd.DataFrame:
    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{split_name} data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    cleaned = frame[[label_column, text_column]].copy()
    cleaned = cleaned.dropna(subset=[label_column, text_column])
    cleaned[label_column] = cleaned[label_column].astype(str).str.strip()
    cleaned[text_column] = cleaned[text_column].astype(str).str.strip()
    cleaned = cleaned[
        (cleaned[label_column] != "") & (cleaned[text_column] != "")
    ].reset_index(drop=True)
    return cleaned


def apply_sample_limit(
    frame: pd.DataFrame,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_samples is None or max_samples >= len(frame):
        return frame.reset_index(drop=True)
    if max_samples <= 0:
        raise ValueError("Sample limits must be positive integers.")
    return (
        frame.sample(n=max_samples, random_state=seed)
        .sort_index()
        .reset_index(drop=True)
    )


def load_dataset(
    dataset_dir: Path,
    train_file: str,
    test_file: str,
    text_column: str,
    label_column: str,
    seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    train_path = dataset_dir / train_file
    test_path = dataset_dir / test_file

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = clean_text_label_frame(
        pd.read_csv(train_path),
        text_column=text_column,
        label_column=label_column,
        split_name="train",
    )
    test_df = clean_text_label_frame(
        pd.read_csv(test_path),
        text_column=text_column,
        label_column=label_column,
        split_name="test",
    )

    label_encoder = LabelEncoder()
    all_labels = pd.concat(
        [train_df[label_column], test_df[label_column]],
        ignore_index=True,
    )
    label_encoder.fit(all_labels)

    train_df["label_id"] = label_encoder.transform(train_df[label_column])
    test_df["label_id"] = label_encoder.transform(test_df[label_column])

    train_df = apply_sample_limit(train_df, max_train_samples, seed)
    test_df = apply_sample_limit(test_df, max_test_samples, seed)

    print(f"Loaded training rows: {len(train_df):,}")
    print(f"Loaded test rows: {len(test_df):,}")
    print(f"Diagnosis labels: {len(label_encoder.classes_):,}")
    return train_df, test_df, label_encoder


def save_label_mapping(label_encoder: LabelEncoder, output_dir: Path) -> None:
    mapping_df = pd.DataFrame(
        {
            "label_id": np.arange(len(label_encoder.classes_)),
            "diagnosis": label_encoder.classes_,
        }
    )
    mapping_df.to_csv(output_dir / "label_mapping.csv", index=False)


def train_tfidf_logistic_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    print("\nTraining TF-IDF + Logistic Regression...")
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, args.tfidf_ngram_max),
                    max_features=args.tfidf_max_features,
                    min_df=args.tfidf_min_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    max_iter=args.tfidf_max_iter,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    start_time = time.perf_counter()
    model.fit(train_df[text_column], train_df["label_id"])
    training_seconds = time.perf_counter() - start_time
    predictions = model.predict(test_df[text_column])

    model_path = output_dir / "models" / "tfidf_logistic_regression.joblib"
    joblib.dump(model, model_path)
    print(f"Saved TF-IDF model: {model_path}")
    print(f"TF-IDF training seconds: {training_seconds:.2f}")
    return np.asarray(predictions), training_seconds


def resolve_fp16(fp16_arg: str) -> bool:
    import torch

    if fp16_arg == "true":
        return True
    if fp16_arg == "false":
        return False
    return torch.cuda.is_available()


def build_tokenized_dataset(
    frame: pd.DataFrame,
    tokenizer: Any,
    text_column: str,
    max_length: int,
) -> Any:
    from datasets import Dataset

    dataset = Dataset.from_pandas(
        pd.DataFrame(
            {
                "text": frame[text_column].tolist(),
                "labels": frame["label_id"].astype(int).tolist(),
            }
        ),
        preserve_index=False,
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    tokenized.set_format("torch")
    return tokenized


def make_training_arguments(
    output_dir: Path,
    args: argparse.Namespace,
) -> Any:
    from transformers import TrainingArguments

    base_arguments: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_strategy": "steps",
        "logging_steps": 100,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": resolve_fp16(args.fp16),
        "dataloader_num_workers": args.dataloader_num_workers,
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        base_arguments["eval_strategy"] = "no"
    else:
        base_arguments["evaluation_strategy"] = "no"

    return TrainingArguments(**base_arguments)


def make_trainer(
    model: Any,
    training_args: Any,
    train_dataset: Any,
    tokenizer: Any,
) -> Any:
    from transformers import DataCollatorWithPadding, Trainer

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def train_transformer_model(
    model_key: str,
    pretrained_model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    label_names: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    display_name = MODEL_DISPLAY_NAMES[model_key]
    print(f"\nTraining {display_name} from {pretrained_model_name}...")

    model_output_dir = output_dir / "models" / model_key
    checkpoint_output_dir = output_dir / "checkpoints" / model_key
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    id2label = {index: label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_dataset = build_tokenized_dataset(
        train_df,
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=args.max_length,
    )
    test_dataset = build_tokenized_dataset(
        test_df,
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=args.max_length,
    )

    training_args = make_training_arguments(checkpoint_output_dir, args)
    trainer = make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    start_time = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start_time

    prediction_output = trainer.predict(test_dataset)
    predictions = np.argmax(prediction_output.predictions, axis=-1)

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    log_history_path = output_dir / f"{model_key}_trainer_log_history.json"
    log_history_path.write_text(
        json.dumps(trainer.state.log_history, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {display_name} model: {model_output_dir}")
    print(f"{display_name} training seconds: {training_seconds:.2f}")

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.asarray(predictions), training_seconds


def compute_metric_row(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_seconds: float,
) -> dict[str, float | str]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
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
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Macro Precision": macro_precision * 100,
        "Macro Recall": macro_recall * 100,
        "Macro F1": macro_f1 * 100,
        "Weighted Precision": weighted_precision * 100,
        "Weighted Recall": weighted_recall * 100,
        "Weighted F1": weighted_f1 * 100,
        "Training Time (seconds)": training_seconds,
    }


def save_classification_report(
    model_key: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    output_dir: Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    output_path = output_dir / MODEL_REPORT_FILES[model_key]
    report_df.to_csv(output_path)
    print(f"Saved report: {output_path}")


def save_prediction_table(
    test_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    label_names: list[str],
    predictions_by_model: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    prediction_df = pd.DataFrame(
        {
            "text": test_df[text_column],
            "true_label_id": test_df["label_id"],
            "true_diagnosis": test_df[label_column],
        }
    )
    for model_key, predictions in predictions_by_model.items():
        prediction_df[f"{model_key}_predicted_label_id"] = predictions
        prediction_df[f"{model_key}_predicted_diagnosis"] = [
            label_names[index] for index in predictions
        ]
    prediction_df.to_csv(output_dir / "model_predictions.csv", index=False)


def save_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = comparison_df.sort_values("Rank").reset_index(drop=True)
    x_positions = np.arange(len(METRIC_COLUMNS))
    bar_width = 0.8 / max(1, len(plot_df))

    fig, ax = plt.subplots(figsize=(15, 7))
    for model_index, row in plot_df.iterrows():
        offset = (model_index - (len(plot_df) - 1) / 2) * bar_width
        values = [row[metric] for metric in METRIC_COLUMNS]
        bars = ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=row["Model"],
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_title("Three-Model Disease Diagnosis Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(METRIC_COLUMNS, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "three_model_comparison.png", dpi=300)
    plt.close(fig)


def write_comparison_outputs(
    metric_rows: list[dict[str, float | str]],
    test_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    label_names: list[str],
    predictions_by_model: dict[str, np.ndarray],
    output_dir: Path,
) -> pd.DataFrame:
    comparison_df = pd.DataFrame(metric_rows)
    numeric_columns = METRIC_COLUMNS + ["Training Time (seconds)"]
    comparison_df[numeric_columns] = comparison_df[numeric_columns].round(4)
    comparison_df = comparison_df.sort_values(
        "Weighted F1",
        ascending=False,
    ).reset_index(drop=True)
    comparison_df.insert(0, "Rank", np.arange(1, len(comparison_df) + 1))

    comparison_df.to_csv(output_dir / "three_model_comparison.csv", index=False)

    best_by_metric = comparison_df[METRIC_COLUMNS].max()
    difference_df = comparison_df[["Model"]].copy()
    for metric in METRIC_COLUMNS:
        difference_df[f"{metric} gap from best"] = (
            comparison_df[metric] - best_by_metric[metric]
        ).round(4)
    difference_df.to_csv(output_dir / "three_model_differences.csv", index=False)

    save_prediction_table(
        test_df=test_df,
        text_column=text_column,
        label_column=label_column,
        label_names=label_names,
        predictions_by_model=predictions_by_model,
        output_dir=output_dir,
    )
    save_comparison_plot(comparison_df, output_dir)
    return comparison_df


def prepare_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = vars(args).copy()
    config["models"] = normalize_model_selection(args.models)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def validate_completed_model_set(
    selected_models: list[str],
    predictions_by_model: dict[str, np.ndarray],
) -> None:
    missing_models = [
        model_key
        for model_key in selected_models
        if model_key not in predictions_by_model
    ]
    if missing_models:
        missing_names = [
            MODEL_DISPLAY_NAMES[model_key]
            for model_key in missing_models
        ]
        raise RuntimeError(
            "Training finished without predictions for: "
            + ", ".join(missing_names)
        )


def main() -> None:
    args = parse_args()
    selected_models = normalize_model_selection(args.models)
    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)

    selected_model_names = [
        MODEL_DISPLAY_NAMES[model_key]
        for model_key in selected_models
    ]
    print("Selected models: " + ", ".join(selected_model_names))

    prepare_output_dirs(output_dir)
    save_run_config(args, output_dir)
    set_global_seed(args.seed)

    train_df, test_df, label_encoder = load_dataset(
        dataset_dir=dataset_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        text_column=args.text_column,
        label_column=args.label_column,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    save_label_mapping(label_encoder, output_dir)
    label_names = label_encoder.classes_.tolist()
    y_true = test_df["label_id"].to_numpy()

    metric_rows: list[dict[str, float | str]] = []
    predictions_by_model: dict[str, np.ndarray] = {}

    for model_number, model_key in enumerate(selected_models, start=1):
        print(
            f"\nStarting {MODEL_DISPLAY_NAMES[model_key]} "
            f"({model_number}/{len(selected_models)})"
        )
        if model_key == "tfidf":
            predictions, training_seconds = train_tfidf_logistic_regression(
                train_df=train_df,
                test_df=test_df,
                text_column=args.text_column,
                output_dir=output_dir,
                args=args,
            )
        elif model_key == "bert":
            predictions, training_seconds = train_transformer_model(
                model_key="bert",
                pretrained_model_name=args.bert_model_name,
                train_df=train_df,
                test_df=test_df,
                text_column=args.text_column,
                label_names=label_names,
                output_dir=output_dir,
                args=args,
            )
        elif model_key == "biobert":
            predictions, training_seconds = train_transformer_model(
                model_key="biobert",
                pretrained_model_name=args.biobert_model_name,
                train_df=train_df,
                test_df=test_df,
                text_column=args.text_column,
                label_names=label_names,
                output_dir=output_dir,
                args=args,
            )
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

        if len(predictions) != len(y_true):
            raise AssertionError(
                f"{MODEL_DISPLAY_NAMES[model_key]} produced {len(predictions)} "
                f"predictions for {len(y_true)} test rows."
            )

        predictions_by_model[model_key] = predictions
        save_classification_report(
            model_key=model_key,
            y_true=y_true,
            y_pred=predictions,
            label_names=label_names,
            output_dir=output_dir,
        )
        metric_rows.append(
            compute_metric_row(
                model_name=MODEL_DISPLAY_NAMES[model_key],
                y_true=y_true,
                y_pred=predictions,
                training_seconds=training_seconds,
            )
        )

    validate_completed_model_set(selected_models, predictions_by_model)
    comparison_df = write_comparison_outputs(
        metric_rows=metric_rows,
        test_df=test_df,
        text_column=args.text_column,
        label_column=args.label_column,
        label_names=label_names,
        predictions_by_model=predictions_by_model,
        output_dir=output_dir,
    )

    print("\nFinal model comparison ranked by Weighted F1:")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
