#!/usr/bin/env python
"""Evaluate Whisper preprocessing and BioBERT stability on medical speech."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
DATASET_ROOT = SCRIPT_DIR / "datasets"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_BIOBERT_DIR = (
    PROJECT_ROOT
    / "3 Models After Training"
    / "outputs"
    / "models"
    / "biobert"
)
DEFAULT_WHISPER_MODEL = "openai/whisper-small"
TARGET_SAMPLE_RATE = 16_000

sys.path.insert(0, str(PIPELINE_DIR))
from voice_to_disease import preprocess_audio, preprocess_text, validate_biobert_model_dir


PRIMOCK_CASES = [
    # Interval positions were manually checked against each patient transcript.
    # Administrative turns and non-symptom replies are deliberately excluded.
    ("day1_consultation03", "migraine", [0, 1, 2]),
    ("day1_consultation05", "urinary tract infection", [0, 1, 3, 7, 8]),
    ("day5_consultation08", "asthma", [1, 2, 6, 8]),
]


@dataclass
class AudioSample:
    dataset: str
    sample_id: str
    speaker_id: str
    reference_text: str
    audio: np.ndarray
    source_path: str
    expected_disease: str | None = None
    source_category: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw and preprocessed Whisper transcription, then measure "
            "the stability of the final trained BioBERT classifier."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["medical_speech_intent", "primock57"],
        default=["medical_speech_intent", "primock57"],
    )
    parser.add_argument("--samples-per-dataset", type=int, default=3)
    parser.add_argument(
        "--snr-levels",
        nargs="*",
        type=float,
        default=[20.0, 10.0, 5.0, 0.0],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--biobert-dir", type=Path, default=DEFAULT_BIOBERT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-seconds", type=float, default=25.0)
    parser.add_argument("--minimum-score", type=float, default=0.50)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Verify and summarize local datasets without loading either model.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this environment cannot access CUDA.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def normalize_for_error_rate(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def clean_textgrid_text(text: str) -> str:
    text = re.sub(r"</?[^>]+>", "", text)
    return " ".join(text.split()).strip()


def load_audio(path: Path) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-6:
        raise ValueError(f"Silent or empty audio: {path}")
    return audio.astype(np.float32)


def locate_medical_speech_root() -> Path:
    root = DATASET_ROOT / "medical_speech_intent"
    candidates = [
        path.parent
        for path in root.rglob("overview-of-recordings.csv")
        if (path.parent / "recordings").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Medical Speech metadata and recordings were not found under "
            f"{root}"
        )
    return sorted(candidates, key=lambda path: (len(path.parts), str(path).lower()))[0]


def load_medical_speech_samples(limit: int) -> list[AudioSample]:
    dataset_dir = locate_medical_speech_root()
    metadata = pd.read_csv(dataset_dir / "overview-of-recordings.csv")
    audio_index = {
        path.name: path
        for path in (dataset_dir / "recordings").rglob("*.wav")
    }
    metadata = metadata[metadata["file_name"].isin(audio_index)].copy()
    metadata["word_count"] = metadata["phrase"].fillna("").str.split().str.len()
    eligible = metadata[
        (metadata["overall_quality_of_the_audio"] >= 4.0)
        & (metadata["audio_clipping"] == "no_clipping")
        & (metadata["quiet_speaker"] == "audible_speaker")
        & metadata["word_count"].between(6, 35)
    ].copy()
    eligible = eligible.sort_values(
        ["overall_quality_of_the_audio", "speaker_id", "file_name"],
        ascending=[False, True, True],
    )

    selected_rows: list[pd.Series] = []
    used_speakers: set[str] = set()
    used_prompts: set[str] = set()
    for _, row in eligible.iterrows():
        speaker = str(row["speaker_id"])
        prompt = str(row["prompt"])
        if speaker in used_speakers or prompt in used_prompts:
            continue
        selected_rows.append(row)
        used_speakers.add(speaker)
        used_prompts.add(prompt)
        if len(selected_rows) >= limit:
            break

    if len(selected_rows) < limit:
        raise RuntimeError(
            f"Only {len(selected_rows)} speaker-balanced Medical Speech samples "
            f"were available; requested {limit}."
        )

    samples = []
    for row in selected_rows:
        audio_path = audio_index[str(row["file_name"])]
        samples.append(
            AudioSample(
                dataset="medical_speech_intent",
                sample_id=audio_path.stem,
                speaker_id=str(row["speaker_id"]),
                reference_text=str(row["phrase"]),
                audio=load_audio(audio_path),
                source_path=str(audio_path.resolve()),
                source_category=str(row["prompt"]),
            )
        )
    return samples


def load_primock_sample(
    case_id: str,
    expected_disease: str,
    symptom_interval_positions: list[int],
) -> AudioSample:
    import textgrids

    root = DATASET_ROOT / "primock57"
    audio_path = root / "audio" / f"{case_id}_patient.wav"
    transcript_path = root / "transcripts" / f"{case_id}_patient.TextGrid"
    if not audio_path.is_file() or not transcript_path.is_file():
        raise FileNotFoundError(f"Missing PriMock57 patient audio/transcript for {case_id}")

    audio = load_audio(audio_path)
    grid = textgrids.TextGrid(str(transcript_path))
    tier = next(iter(grid.values()))
    substantive: list[tuple[int, int, str]] = []
    for interval in tier:
        text = clean_textgrid_text(str(interval.text))
        if len(normalize_for_error_rate(text).split()) < 5:
            continue
        start = max(0, int(float(interval.xmin) * TARGET_SAMPLE_RATE))
        end = min(audio.size, int(float(interval.xmax) * TARGET_SAMPLE_RATE))
        if end > start:
            substantive.append((start, end, text))
    if not substantive:
        raise RuntimeError(f"No usable patient utterances found for {case_id}")
    if max(symptom_interval_positions) >= len(substantive):
        raise IndexError(
            f"Configured symptom interval is outside the transcript for {case_id}."
        )
    selected = [substantive[position] for position in symptom_interval_positions]
    if not selected:
        raise RuntimeError(f"No usable patient symptom utterances found for {case_id}")

    silence = np.zeros(int(0.20 * TARGET_SAMPLE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, (start, end, _) in enumerate(selected):
        if index:
            pieces.append(silence)
        pieces.append(audio[start:end])
    return AudioSample(
        dataset="primock57",
        sample_id=case_id,
        speaker_id=f"{case_id}_patient",
        reference_text=" ".join(item[2] for item in selected),
        audio=np.concatenate(pieces).astype(np.float32),
        source_path=str(audio_path.resolve()),
        expected_disease=expected_disease,
        source_category="manually_curated_patient_symptom_utterances",
    )


def load_primock_samples(limit: int) -> list[AudioSample]:
    if limit > len(PRIMOCK_CASES):
        raise ValueError(
            f"Only {len(PRIMOCK_CASES)} manually checked PriMock57 cases are "
            "configured for disease evaluation."
        )
    return [load_primock_sample(*case) for case in PRIMOCK_CASES[:limit]]


def load_noise_file() -> tuple[np.ndarray, Path]:
    noise_path = (
        DATASET_ROOT
        / "rirs_noises"
        / "RIRS_NOISES"
        / "pointsource_noises"
        / "noise-free-sound-0232.wav"
    )
    if not noise_path.is_file():
        raise FileNotFoundError(f"RIRS background-noise file not found: {noise_path}")
    return load_audio(noise_path), noise_path


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    seed: int,
) -> np.ndarray:
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if noise.size < speech.size:
        repeats = int(np.ceil(speech.size / noise.size))
        noise_segment = np.tile(noise, repeats)[: speech.size]
    else:
        start = int(rng.integers(0, noise.size - speech.size + 1))
        noise_segment = noise[start : start + speech.size]
    noise_segment = noise_segment - np.mean(noise_segment)
    speech_power = float(np.mean(np.square(speech)))
    noise_power = float(np.mean(np.square(noise_segment)))
    if speech_power <= 0 or noise_power <= 0:
        raise ValueError("Speech or noise power is zero; SNR mixing is undefined.")
    target_noise_power = speech_power / (10 ** (snr_db / 10.0))
    mixed = speech + noise_segment * np.sqrt(target_noise_power / noise_power)
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


class WhisperEngine:
    def __init__(self, model_name: str, device: str, chunk_seconds: float) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = device
        self.chunk_samples = int(chunk_seconds * TARGET_SAMPLE_RATE)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        if device == "cuda":
            self.model.half()
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        transcripts = []
        for start in range(0, audio.size, self.chunk_samples):
            chunk = audio[start : start + self.chunk_samples]
            inputs = self.processor(
                chunk,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            generation_args: dict[str, Any] = {
                "input_features": inputs.input_features.to(self.device),
                "language": "en",
                "task": "transcribe",
                "max_new_tokens": 128,
            }
            if getattr(inputs, "attention_mask", None) is not None:
                generation_args["attention_mask"] = inputs.attention_mask.to(self.device)
            with self.torch.inference_mode():
                ids = self.model.generate(**generation_args)
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if text:
                transcripts.append(text)
        return " ".join(transcripts).strip()


class BioBertEngine:
    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        model_dir = validate_biobert_model_dir(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model.to(device)
        if device == "cuda":
            self.model.half()
        self.model.eval()

    def predict(self, text: str) -> dict[str, Any]:
        text = preprocess_text(text)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(
                self.model(**encoded).logits[0].float(),
                dim=-1,
            )
        scores, ids = self.torch.topk(probabilities, k=3)
        predictions = [
            {
                "disease": str(self.model.config.id2label[int(label_id)]),
                "score": float(score),
            }
            for label_id, score in zip(ids.cpu().tolist(), scores.cpu().tolist())
        ]
        return {"best": predictions[0], "top3": predictions}


def inventory() -> dict[str, Any]:
    medical_root = locate_medical_speech_root()
    medical_metadata = pd.read_csv(medical_root / "overview-of-recordings.csv")
    primock_root = DATASET_ROOT / "primock57"
    noise_root = DATASET_ROOT / "rirs_noises" / "RIRS_NOISES"
    point_noise = list((noise_root / "pointsource_noises").glob("*.wav"))
    result = {
        "medical_speech_intent": {
            "metadata_rows": int(len(medical_metadata)),
            "speakers": int(medical_metadata["speaker_id"].nunique()),
            "symptom_prompts": int(medical_metadata["prompt"].nunique()),
            "wav_files": len(list((medical_root / "recordings").rglob("*.wav"))),
        },
        "primock57": {
            "patient_audio_files": len(list((primock_root / "audio").glob("*_patient.wav"))),
            "doctor_audio_files": len(list((primock_root / "audio").glob("*_doctor.wav"))),
            "patient_transcripts": len(
                list((primock_root / "transcripts").glob("*_patient.TextGrid"))
            ),
            "manually_mapped_test_cases": len(PRIMOCK_CASES),
        },
        "rirs_noises": {
            "all_wav_files": len(list(noise_root.rglob("*.wav"))),
            "point_source_noise_files": len(point_noise),
            "selected_noise_file": "noise-free-sound-0232.wav",
        },
    }
    return result


def save_summary(results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    group_columns = ["dataset", "snr_label", "audio_pipeline"]
    summary = (
        results.groupby(group_columns, dropna=False)
        .agg(
            samples=("sample_id", "count"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            biobert_stability_rate=("matches_reference_prediction", "mean"),
            low_score_rate=("low_score", "mean"),
            mapped_disease_accuracy=("expected_disease_match", "mean"),
        )
        .reset_index()
    )
    numeric = summary.select_dtypes(include=["number"]).columns
    summary[numeric] = summary[numeric].round(4)
    summary.to_csv(output_dir / "dataset_test_summary.csv", index=False)
    save_comparison_plot(summary, output_dir)
    return summary


def save_comparison_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    condition_order = ["clean", "20_dB", "10_dB", "5_dB", "0_dB"]
    condition_labels = ["Clean", "20 dB", "10 dB", "5 dB", "0 dB"]
    datasets = ["medical_speech_intent", "primock57"]
    dataset_titles = {
        "medical_speech_intent": "Medical Speech and Intent",
        "primock57": "PriMock57",
    }
    metrics = [("mean_wer", "Mean WER"), ("mean_cer", "Mean CER")]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row_index, dataset in enumerate(datasets):
        subset = summary[summary["dataset"] == dataset]
        for column_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for pipeline in ["raw", "preprocessed"]:
                values = []
                for condition in condition_order:
                    match = subset[
                        (subset["snr_label"] == condition)
                        & (subset["audio_pipeline"] == pipeline)
                    ]
                    values.append(float(match.iloc[0][metric]) if len(match) else np.nan)
                axis.plot(condition_labels, values, marker="o", label=pipeline)
            axis.set_title(f"{dataset_titles[dataset]}: {title}")
            axis.set_ylabel(title)
            axis.grid(alpha=0.3)
            axis.legend()
    figure.suptitle("Pilot Medical Speech Evaluation: Raw vs Preprocessed Audio")
    figure.tight_layout()
    figure.savefig(output_dir / "dataset_test_comparison.png", dpi=200)
    plt.close(figure)


def run_evaluation(args: argparse.Namespace) -> None:
    from jiwer import cer, wer

    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_data = inventory()
    (output_dir / "dataset_inventory.json").write_text(
        json.dumps(inventory_data, indent=2), encoding="utf-8"
    )
    print(json.dumps(inventory_data, indent=2))
    if args.inventory_only:
        print(f"Saved inventory: {output_dir / 'dataset_inventory.json'}")
        return

    samples: list[AudioSample] = []
    if "medical_speech_intent" in args.datasets:
        samples.extend(load_medical_speech_samples(args.samples_per_dataset))
    if "primock57" in args.datasets:
        samples.extend(load_primock_samples(args.samples_per_dataset))

    device = resolve_device(args.device)
    print(f"Inference device: {device}")
    print(f"Evaluation samples: {len(samples)}")
    noise, noise_path = load_noise_file()
    print(f"RIRS noise: {noise_path}")
    print(f"Loading Whisper once: {args.whisper_model}")
    whisper = WhisperEngine(args.whisper_model, device, args.chunk_seconds)
    print(f"Loading final trained BioBERT once: {args.biobert_dir}")
    classifier = BioBertEngine(args.biobert_dir, device)

    conditions: list[tuple[str, float | None]] = [("clean", None)]
    conditions.extend((f"{snr:g}_dB", snr) for snr in args.snr_levels)
    rows: list[dict[str, Any]] = []
    result_path = output_dir / "dataset_test_results.csv"

    for sample_index, sample in enumerate(samples, start=1):
        reference_prediction = classifier.predict(sample.reference_text)
        print(
            f"\nSample {sample_index}/{len(samples)}: {sample.dataset}/"
            f"{sample.sample_id} | reference BioBERT="
            f"{reference_prediction['best']['disease']}"
        )
        for snr_label, snr_db in conditions:
            if snr_db is None:
                condition_audio = sample.audio.copy()
            else:
                condition_audio = mix_at_snr(
                    sample.audio,
                    noise,
                    snr_db,
                    # Keep the noise segment fixed for this sample. Only its
                    # amplitude changes across SNR levels.
                    stable_seed(sample.sample_id, args.seed),
                )

            for pipeline_name in ["raw", "preprocessed"]:
                if pipeline_name == "preprocessed":
                    asr_audio, _ = preprocess_audio(condition_audio)
                else:
                    asr_audio = condition_audio
                hypothesis = whisper.transcribe(asr_audio)
                normalized_reference = normalize_for_error_rate(sample.reference_text)
                normalized_hypothesis = normalize_for_error_rate(hypothesis)
                if normalized_hypothesis:
                    word_error = float(wer(normalized_reference, normalized_hypothesis))
                    character_error = float(cer(normalized_reference, normalized_hypothesis))
                    prediction = classifier.predict(hypothesis)
                else:
                    word_error = 1.0
                    character_error = 1.0
                    prediction = {
                        "best": {"disease": "", "score": 0.0},
                        "top3": [],
                    }

                predicted_disease = prediction["best"]["disease"]
                score = float(prediction["best"]["score"])
                expected_match: bool | None = None
                if sample.expected_disease is not None:
                    expected_match = predicted_disease == sample.expected_disease
                row = {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "speaker_id": sample.speaker_id,
                    "source_category": sample.source_category,
                    "source_path": sample.source_path,
                    "expected_disease": sample.expected_disease,
                    "reference_text": sample.reference_text,
                    "reference_biobert_disease": reference_prediction["best"]["disease"],
                    "reference_biobert_score": reference_prediction["best"]["score"],
                    "snr_label": snr_label,
                    "snr_db": snr_db,
                    "audio_pipeline": pipeline_name,
                    "whisper_transcript": hypothesis,
                    "wer": word_error,
                    "cer": character_error,
                    "predicted_disease": predicted_disease,
                    "biobert_score": score,
                    "matches_reference_prediction": (
                        predicted_disease == reference_prediction["best"]["disease"]
                    ),
                    "expected_disease_match": expected_match,
                    "low_score": score < args.minimum_score,
                    "top3_json": json.dumps(prediction["top3"]),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(result_path, index=False)
                print(
                    f"  {snr_label:>6} | {pipeline_name:<12} | "
                    f"WER={word_error:.3f} CER={character_error:.3f} | "
                    f"{predicted_disease} ({score:.1%})"
                )

    results = pd.DataFrame(rows)
    results.to_csv(result_path, index=False)
    summary = save_summary(results, output_dir)
    print("\n=== Evaluation summary ===")
    print(summary.to_string(index=False))
    print(f"\nDetailed results: {result_path}")
    print(f"Summary: {output_dir / 'dataset_test_summary.csv'}")

    del whisper
    del classifier
    gc.collect()


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
