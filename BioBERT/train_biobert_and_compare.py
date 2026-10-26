#!/usr/bin/env python
"""Train BioBERT only, then build the three-model comparison table."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


BIOBERT_DISPLAY_NAME = "BioBERT"
BIOBERT_REPORT_FILE = "biobert_classification_report.csv"
BASELINE_REPORTS = {
    "TF-IDF + Logistic Regression": "tfidf_logistic_regression_classification_report.csv",
    "Fine-tuned BERT": "fine_tuned_bert_classification_report.csv",
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
            "Train only BioBERT, then combine its metrics with existing "
            "TF-IDF and fine-tuned BERT reports."
        )
    )
    parser.add_argument("--dataset-dir", default="data/disease-diagnosis-dataset")
    parser.add_argument("--baseline-dir", default="baseline_outputs")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--train-file", default="disease_diagnosis_train_cleaned.csv")
    parser.add_argument("--test-file", default="disease_diagnosis_test_cleaned.csv")
    parser.add_argument("--text-column", default="text_cleaned")
    parser.add_argument("--label-column", default="diagnosis")
    parser.add_argument(
        "--biobert-model-name",
        default="dmis-lab/biobert-base-cased-v1.2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save resumable checkpoints every N optimizer steps.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help=(
            "Use 'auto' or 'true' to resume from the latest checkpoint, "
            "'false' to start fresh, or pass a checkpoint path."
        ),
    )
    parser.add_argument(
        "--fp16",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional row limit for quick smoke testing.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional row limit for quick smoke testing.",
    )
    return parser.parse_args()


def require_torch_26_or_newer(starting_from_base_model: bool) -> None:
    import torch

    version_text = torch.__version__.split("+")[0]
    if Version(version_text) < Version("2.6.0"):
        if not starting_from_base_model:
            print(
                "Torch is below 2.6, but a saved BioBERT checkpoint was found. "
                "Continuing from checkpoint instead of loading the upstream "
                "PyTorch .bin weights."
            )
            return
        raise RuntimeError(
            "BioBERT requires torch>=2.6 in this environment because the "
            "upstream BioBERT checkpoint is loaded from PyTorch .bin weights. "
            f"Current torch version: {torch.__version__}. Install Torch 2.6 "
            "or newer before running this SLURM job, or resume from an existing "
            "outputs/checkpoints/biobert/checkpoint-* folder."
        )
    print(f"Torch version OK for BioBERT loading: {torch.__version__}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text_label_frame(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    split_name: str,
) -> pd.DataFrame:
    missing_columns = {text_column, label_column}.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{split_name} data is missing columns: {sorted(missing_columns)}"
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
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    train_path = dataset_dir / train_file
    test_path = dataset_dir / test_file
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = clean_text_label_frame(
        pd.read_csv(train_path),
        text_column,
        label_column,
        "train",
    )
    test_df = clean_text_label_frame(
        pd.read_csv(test_path),
        text_column,
        label_column,
        "test",
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(
        pd.concat([train_df[label_column], test_df[label_column]], ignore_index=True)
    )
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


def resolve_fp16(fp16_arg: str) -> bool:
    import torch

    if fp16_arg == "true":
        return True
    if fp16_arg == "false":
        return False
    return torch.cuda.is_available()


def make_training_arguments(output_dir: Path, args: argparse.Namespace) -> Any:
    from transformers import TrainingArguments

    training_args: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_strategy": "steps",
        "logging_steps": 100,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": resolve_fp16(args.fp16),
        "dataloader_num_workers": args.dataloader_num_workers,
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        training_args["eval_strategy"] = "no"
    else:
        training_args["evaluation_strategy"] = "no"
    return TrainingArguments(**training_args)


def resolve_resume_checkpoint(
    checkpoint_dir: Path,
    resume_from_checkpoint: str,
) -> str | None:
    normalized = resume_from_checkpoint.strip().lower()
    if normalized in {"false", "no", "none", "0"}:
        print("Checkpoint resume disabled; starting BioBERT from the base model.")
        return None

    if normalized in {"auto", "true", "yes", "1"}:
        if not checkpoint_dir.exists():
            print("No existing BioBERT checkpoint found; starting from the base model.")
            return None
        checkpoints = [
            path
            for path in checkpoint_dir.glob("checkpoint-*")
            if path.is_dir()
        ]
        if not checkpoints:
            print("No existing BioBERT checkpoint found; starting from the base model.")
            return None

        def checkpoint_sort_key(path: Path) -> tuple[int, float]:
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                step = -1
            return step, path.stat().st_mtime

        checkpoint_path = max(checkpoints, key=checkpoint_sort_key)
        print(f"Resuming BioBERT from checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Requested checkpoint does not exist: {checkpoint_path}"
        )
    print(f"Resuming BioBERT from checkpoint: {checkpoint_path}")
    return str(checkpoint_path)


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


def train_biobert_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    label_names: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    resume_checkpoint: str | None,
) -> tuple[np.ndarray, float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_source = resume_checkpoint or args.biobert_model_name
    tokenizer_source = (
        model_source
        if resume_checkpoint is not None
        and (Path(resume_checkpoint) / "tokenizer_config.json").exists()
        else args.biobert_model_name
    )
    print(f"\nTraining BioBERT from {model_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    id2label = {index: label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
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

    checkpoint_dir = output_dir / "checkpoints" / "biobert"
    trainer = make_trainer(
        model=model,
        training_args=make_training_arguments(checkpoint_dir, args),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    start_time = time.perf_counter()
    trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )
    training_seconds = time.perf_counter() - start_time

    prediction_output = trainer.predict(test_dataset)
    predictions = np.argmax(prediction_output.predictions, axis=-1)

    model_output_dir = output_dir / "models" / "biobert"
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    (output_dir / "biobert_trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2),
        encoding="utf-8",
    )

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved BioBERT model: {model_output_dir}")
    print(f"BioBERT training seconds: {training_seconds:.2f}")
    return predictions, training_seconds


def save_biobert_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    output_dir: Path,
) -> Path:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_path = output_dir / BIOBERT_REPORT_FILE
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Saved BioBERT classification report: {report_path}")
    return report_path


def report_to_metric_row(
    model_name: str,
    report_path: Path,
    training_seconds: float | None = None,
) -> dict[str, float | str]:
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found for {model_name}: {report_path}")

    report_df = pd.read_csv(report_path, index_col=0)
    accuracy = float(report_df.loc["accuracy", "precision"])
    macro = report_df.loc["macro avg"]
    weighted = report_df.loc["weighted avg"]

    row: dict[str, float | str] = {
        "Model": model_name,
        "Accuracy": accuracy * 100,
        "Macro Precision": float(macro["precision"]) * 100,
        "Macro Recall": float(macro["recall"]) * 100,
        "Macro F1": float(macro["f1-score"]) * 100,
        "Weighted Precision": float(weighted["precision"]) * 100,
        "Weighted Recall": float(weighted["recall"]) * 100,
        "Weighted F1": float(weighted["f1-score"]) * 100,
    }
    if training_seconds is not None:
        row["Training Time (seconds)"] = training_seconds
    else:
        row["Training Time (seconds)"] = ""
    return row


def save_predictions(
    test_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    label_names: list[str],
    predictions: np.ndarray,
    output_dir: Path,
) -> None:
    prediction_df = pd.DataFrame(
        {
            "text": test_df[text_column],
            "true_label_id": test_df["label_id"],
            "true_diagnosis": test_df[label_column],
            "biobert_predicted_label_id": predictions,
            "biobert_predicted_diagnosis": [
                label_names[index] for index in predictions
            ],
        }
    )
    prediction_df.to_csv(output_dir / "biobert_predictions.csv", index=False)


def save_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = comparison_df.sort_values("Rank").reset_index(drop=True)
    x_positions = np.arange(len(METRIC_COLUMNS))
    bar_width = 0.8 / len(plot_df)

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


def write_three_model_comparison(
    baseline_dir: Path,
    biobert_report_path: Path,
    biobert_training_seconds: float,
    output_dir: Path,
) -> pd.DataFrame:
    rows = [
        report_to_metric_row(
            model_name,
            baseline_dir / report_file,
        )
        for model_name, report_file in BASELINE_REPORTS.items()
    ]
    rows.append(
        report_to_metric_row(
            BIOBERT_DISPLAY_NAME,
            biobert_report_path,
            training_seconds=biobert_training_seconds,
        )
    )

    comparison_df = pd.DataFrame(rows)
    comparison_df[METRIC_COLUMNS] = comparison_df[METRIC_COLUMNS].round(4)
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

    save_comparison_plot(comparison_df, output_dir)
    return comparison_df


def prepare_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    baseline_dir = Path(args.baseline_dir)
    checkpoint_dir = output_dir / "checkpoints" / "biobert"
    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_dir,
        args.resume_from_checkpoint,
    )

    print("Selected model: BioBERT")
    require_torch_26_or_newer(starting_from_base_model=resume_checkpoint is None)
    prepare_output_dirs(output_dir)
    save_run_config(args, output_dir)
    set_global_seed(args.seed)

    train_df, test_df, label_encoder = load_dataset(
        dataset_dir=Path(args.dataset_dir),
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

    predictions, training_seconds = train_biobert_model(
        train_df=train_df,
        test_df=test_df,
        text_column=args.text_column,
        label_names=label_names,
        output_dir=output_dir,
        args=args,
        resume_checkpoint=resume_checkpoint,
    )
    y_true = test_df["label_id"].to_numpy()
    if len(predictions) != len(y_true):
        raise AssertionError(
            f"BioBERT produced {len(predictions)} predictions for "
            f"{len(y_true)} test rows."
        )

    save_predictions(
        test_df=test_df,
        text_column=args.text_column,
        label_column=args.label_column,
        label_names=label_names,
        predictions=predictions,
        output_dir=output_dir,
    )
    biobert_report_path = save_biobert_report(
        y_true=y_true,
        y_pred=predictions,
        label_names=label_names,
        output_dir=output_dir,
    )
    comparison_df = write_three_model_comparison(
        baseline_dir=baseline_dir,
        biobert_report_path=biobert_report_path,
        biobert_training_seconds=training_seconds,
        output_dir=output_dir,
    )

    print("\nFinal three-model comparison ranked by Weighted F1:")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
