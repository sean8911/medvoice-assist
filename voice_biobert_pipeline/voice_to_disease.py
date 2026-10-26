#!/usr/bin/env python
"""Voice/audio to Whisper transcript to BioBERT disease-class prediction."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from clarification_questions import generate_clarification
from rule_based_response import generate_rule_based_response
from symptom_extractor import DEFAULT_VOCABULARY_PATH, SymptomExtractor
from urgency_detector import detect_urgency


# This pipeline is PyTorch-only. Ignore unrelated TensorFlow/Flax installations,
# which may otherwise be imported by Transformers and cause version conflicts.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


TARGET_SAMPLE_RATE = 16_000
DEFAULT_WHISPER_MODEL = "openai/whisper-small"
DEFAULT_MAX_LENGTH = 128
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "3 Models After Training"
    / "outputs"
    / "models"
    / "biobert"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record speech or load an audio file, preprocess it, transcribe it "
            "with Whisper, and classify the transcript with the final BioBERT model."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--audio-file",
        type=Path,
        help="Path to an existing WAV, MP3, M4A, or other readable audio file.",
    )
    source.add_argument(
        "--record-seconds",
        type=float,
        help="Record this many seconds from the default microphone.",
    )
    source.add_argument(
        "--text",
        help="Skip audio and Whisper; classify text directly for a quick model test.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Final trained BioBERT model directory.",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="Hugging Face Whisper model name or a local model directory.",
    )
    parser.add_argument("--language", default="en", help="Whisper language code.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. Auto uses CUDA when it is available.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of highest-scoring disease classes to return.",
    )
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=0.50,
        help=(
            "Heuristic score threshold for marking a result as low confidence. "
            "This is not a clinical probability."
        ),
    )
    parser.add_argument(
        "--minimum-margin",
        type=float,
        default=0.10,
        help=(
            "Minimum top-1 minus top-2 score margin before displaying a class. "
            "Close competing classes trigger clarification."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="BioBERT token limit used during training.",
    )
    parser.add_argument(
        "--disable-symptom-extraction",
        action="store_true",
        help=(
            "Send the normalized full transcript directly to BioBERT. This is "
            "provided for the before/after research comparison."
        ),
    )
    parser.add_argument(
        "--symptom-vocabulary",
        type=Path,
        default=DEFAULT_VOCABULARY_PATH,
        help="Canonical symptom vocabulary derived from the training source.",
    )
    parser.add_argument(
        "--minimum-extracted-symptoms",
        type=int,
        default=2,
        help="Request clarification when fewer current symptoms are extracted.",
    )
    parser.add_argument(
        "--max-clarification-questions",
        type=int,
        default=3,
        help="Maximum number of follow-up questions returned in one turn.",
    )
    parser.add_argument(
        "--emergency-number",
        default=os.environ.get("MEDICAL_CHATBOT_EMERGENCY_NUMBER", "999"),
        help="Emergency number shown by the urgency gate. Defaults to Malaysia's 999.",
    )
    parser.add_argument(
        "--urgent-service",
        default="a same-day medical service",
        help="Local urgent-care service wording shown for non-emergency red flags.",
    )
    parser.add_argument(
        "--disable-urgency-detection",
        action="store_true",
        help="Research-only switch that disables the pre-classification safety gate.",
    )
    parser.add_argument(
        "--disable-rule-based-response",
        action="store_true",
        help="Research-only switch that omits the final response-generation stage.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=25.0,
        help="Split long audio into chunks before Whisper transcription.",
    )
    parser.add_argument(
        "--skip-audio-preprocessing",
        action="store_true",
        help="Transcribe resampled raw audio without denoising or normalization.",
    )
    parser.add_argument(
        "--save-processed-audio",
        type=Path,
        help="Optional WAV path for the preprocessed audio.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for saving the complete inference result as JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.record_seconds is not None and args.record_seconds <= 0:
        raise ValueError("--record-seconds must be greater than zero.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")
    if not 0.0 <= args.minimum_score <= 1.0:
        raise ValueError("--minimum-score must be between 0 and 1.")
    if not 0.0 <= args.minimum_margin <= 1.0:
        raise ValueError("--minimum-margin must be between 0 and 1.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than zero.")
    if args.minimum_extracted_symptoms <= 0:
        raise ValueError("--minimum-extracted-symptoms must be greater than zero.")
    if args.max_clarification_questions <= 0:
        raise ValueError("--max-clarification-questions must be greater than zero.")
    if not args.emergency_number.strip():
        raise ValueError("--emergency-number cannot be empty.")
    if not args.urgent_service.strip():
        raise ValueError("--urgent-service cannot be empty.")
    if not 1.0 <= args.chunk_seconds <= 29.0:
        raise ValueError("--chunk-seconds must be between 1 and 29 seconds.")


def resolve_device(requested_device: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment cannot access CUDA."
        )
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def load_audio_file(path: Path) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "librosa is not installed. Run: pip install -r requirements.txt"
        ) from exc

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, _ = librosa.load(
        path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        dtype=np.float32,
    )
    return validate_audio(audio, source=str(path))


def record_microphone(seconds: float) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed. Run: pip install -r requirements.txt"
        ) from exc

    frame_count = int(round(seconds * TARGET_SAMPLE_RATE))
    print(f"Recording for {seconds:g} seconds. Start speaking now...")
    recording = sd.rec(
        frame_count,
        samplerate=TARGET_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording complete.")
    return validate_audio(recording.reshape(-1), source="microphone")


def validate_audio(audio: np.ndarray, source: str) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError(f"No audio samples were read from {source}.")
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.max(np.abs(audio))) < 1e-5:
        raise ValueError(f"The audio from {source} is silent or too quiet.")
    return audio


def preprocess_audio(audio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the same improved preprocessing used in the thesis experiment."""
    try:
        import librosa
        import noisereduce as nr
    except ImportError as exc:
        raise RuntimeError(
            "Audio preprocessing packages are missing. "
            "Run: pip install -r requirements.txt"
        ) from exc

    audio = validate_audio(audio, source="input")
    original_samples = int(audio.size)
    dc_offset = float(np.mean(audio))
    centered = audio - dc_offset

    trimmed, trim_indexes = librosa.effects.trim(centered, top_db=30)
    if trimmed.size == 0:
        trimmed = centered
        trim_indexes = np.array([0, centered.size])

    denoised = nr.reduce_noise(
        y=trimmed,
        sr=TARGET_SAMPLE_RATE,
        stationary=True,
        prop_decrease=0.5,
    )
    denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
    peak_before_normalization = float(np.max(np.abs(denoised)))
    if peak_before_normalization > 0:
        denoised = 0.95 * denoised / peak_before_normalization

    processed = validate_audio(denoised.astype(np.float32), source="processed audio")
    metadata = {
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "original_duration_seconds": round(original_samples / TARGET_SAMPLE_RATE, 3),
        "processed_duration_seconds": round(processed.size / TARGET_SAMPLE_RATE, 3),
        "dc_offset_removed": dc_offset,
        "trim_start_seconds": round(int(trim_indexes[0]) / TARGET_SAMPLE_RATE, 3),
        "trim_end_seconds": round(int(trim_indexes[1]) / TARGET_SAMPLE_RATE, 3),
        "noise_reduction": "stationary",
        "noise_reduction_prop_decrease": 0.5,
        "peak_normalization": 0.95,
    }
    return processed, metadata


def save_audio(path: Path, audio: np.ndarray) -> Path:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is not installed. Run: pip install -r requirements.txt"
        ) from exc

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return path


def transcribe_audio(
    audio: np.ndarray,
    model_name: str,
    language: str,
    device: str,
    chunk_seconds: float,
) -> str:
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Whisper dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    print(f"Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    if device == "cuda":
        model.half()
    model.eval()

    chunk_size = max(1, int(chunk_seconds * TARGET_SAMPLE_RATE))
    chunks = [audio[start : start + chunk_size] for start in range(0, audio.size, chunk_size)]
    transcripts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        inputs = processor(
            chunk,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        generation_inputs: dict[str, Any] = {
            "input_features": input_features,
            "language": language,
            "task": "transcribe",
        }
        if attention_mask is not None:
            generation_inputs["attention_mask"] = attention_mask

        with torch.inference_mode():
            predicted_ids = model.generate(**generation_inputs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if text:
            transcripts.append(text)
        print(f"Transcribed audio chunk {index}/{len(chunks)}")

    del model
    del processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    transcript = " ".join(transcripts).strip()
    if not transcript:
        raise RuntimeError("Whisper returned an empty transcript.")
    return transcript


def preprocess_text(text: str) -> str:
    """Normalize a transcript while preserving symptom words and separators."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = re.sub(r"[.;:!?]+", ", ", normalized)
    normalized = re.sub(r"[^a-z0-9,'/\-\s]", " ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r",(?:\s*,)+", ",", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" ,")


def validate_biobert_model_dir(model_dir: Path) -> Path:
    model_dir = model_dir.expanduser().resolve()
    required_files = {
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    missing = sorted(name for name in required_files if not (model_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"BioBERT model directory is incomplete: {model_dir}. "
            f"Missing: {', '.join(missing)}"
        )

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = config.get("id2label", {})
    if len(id2label) != 773:
        raise ValueError(
            f"Expected the final 773-class BioBERT model, but config.json has "
            f"{len(id2label)} labels."
        )

    try:
        from safetensors import safe_open

        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as model_file:
            tensor_names = set(model_file.keys())
    except Exception as exc:
        raise RuntimeError(
            "The final BioBERT model.safetensors file is incomplete or corrupted. "
            "Restore it from the completed training checkpoint before inference."
        ) from exc
    if not {"classifier.weight", "classifier.bias"}.issubset(tensor_names):
        raise ValueError("The BioBERT model file does not contain its classification head.")
    return model_dir


def classify_text(
    text: str,
    model_dir: Path,
    device: str,
    top_k: int,
    max_length: int,
    minimum_score: float,
    minimum_margin: float,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "BioBERT dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    model_dir = validate_biobert_model_dir(model_dir)
    print(f"Loading final trained BioBERT model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(device)
    if device == "cuda":
        model.half()
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits.float(), dim=-1)

    number_of_labels = int(probabilities.shape[0])
    requested_top_k = min(top_k, number_of_labels)
    scores, indexes = torch.topk(probabilities, k=requested_top_k)
    id2label = model.config.id2label
    top_predictions = [
        {
            "rank": rank,
            "label_id": int(label_id),
            "disease": str(id2label[int(label_id)]),
            "model_score": round(float(score), 6),
        }
        for rank, (label_id, score) in enumerate(
            zip(indexes.cpu().tolist(), scores.cpu().tolist()),
            start=1,
        )
    ]
    best = top_predictions[0]
    second_score = top_predictions[1]["model_score"] if len(top_predictions) > 1 else 0.0
    score_margin = round(best["model_score"] - second_score, 6)
    if best["model_score"] < minimum_score:
        status = "low_score_review_symptoms"
    elif score_margin < minimum_margin:
        status = "ambiguous_prediction_review_symptoms"
    else:
        status = "prediction_available"
    return {
        "predicted_disease": best["disease"],
        "predicted_label_id": best["label_id"],
        "model_score": best["model_score"],
        "status": status,
        "minimum_score_threshold": minimum_score,
        "top_two_score_margin": score_margin,
        "minimum_margin_threshold": minimum_margin,
        "top_predictions": top_predictions,
        "number_of_disease_classes": number_of_labels,
        "display_prediction": status == "prediction_available",
    }


def skipped_classification(status: str, minimum_score: float) -> dict[str, Any]:
    """Return one consistent payload when the BioBERT stage is not run."""
    return {
        "predicted_disease": None,
        "predicted_label_id": None,
        "model_score": 0.0,
        "status": status,
        "minimum_score_threshold": minimum_score,
        "top_two_score_margin": None,
        "minimum_margin_threshold": None,
        "top_predictions": [],
        "number_of_disease_classes": 773,
        "display_prediction": False,
    }


def print_result(result: dict[str, Any]) -> None:
    print("\n=== Voice-to-Disease Pipeline Result ===")
    if result.get("whisper_model") is not None:
        print(f"Whisper transcript: {result['transcript']}")
    else:
        print(f"Direct text: {result['transcript']}")
    extraction = result.get("symptom_extraction")
    if extraction is not None:
        print(f"Extracted current symptoms: {', '.join(extraction['present_symptoms']) or 'none'}")
        if extraction["negated_symptoms"]:
            print(f"Excluded negated symptoms: {', '.join(extraction['negated_symptoms'])}")
        if extraction["uncertain_symptoms"]:
            print(f"Uncertain symptoms: {', '.join(extraction['uncertain_symptoms'])}")
    urgency = result.get("urgency")
    if urgency is not None:
        print(f"Urgency level: {urgency['urgency_level']}")
        for rule in urgency["matched_rules"]:
            print(f"  Safety rule: {rule['title']} [{rule['level']}]")
        print(f"Safety action: {urgency['action_message']}")
    print(f"BioBERT input: {result['biobert_input'] or '(classification skipped)'}")
    prediction = result["classification"]
    display_prediction = prediction.get("display_prediction", False)
    if prediction["predicted_disease"] is not None and not display_prediction:
        print("Predicted disease class: withheld pending clarification")
    else:
        print(f"Predicted disease class: {prediction['predicted_disease'] or 'not available'}")
    if prediction["predicted_disease"] is not None and display_prediction:
        print(f"Model score: {prediction['model_score']:.2%}")
    print(f"Status: {prediction['status']}")
    if prediction["top_predictions"] and display_prediction:
        print("Top predictions:")
        for item in prediction["top_predictions"]:
            print(f"  {item['rank']}. {item['disease']} ({item['model_score']:.2%})")
    clarification = result.get("clarification")
    if clarification is not None and clarification["questions"]:
        print("Clarification questions:")
        for index, question in enumerate(clarification["questions"], start=1):
            safety_label = " [safety]" if question["safety_related"] else ""
            print(f"  {index}. {question['question']}{safety_label}")
    if clarification is not None and clarification["deferred_due_to_urgency"]:
        print("Clarification questions deferred: follow the safety action first.")
    response = result.get("rule_based_response")
    if response is not None:
        print("\n=== Rule-Based Chatbot Response ===")
        print(response["response_text"])
    print(
        "Safety note: This output is a research-model prediction, not a medical "
        "diagnosis. Emergency symptoms require immediate professional care."
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    print(f"Inference device: {device}")

    transcript: str
    audio_metadata: dict[str, Any] | None = None
    input_source: str

    if args.text is not None:
        transcript = args.text.strip()
        input_source = "direct_text"
        if not transcript:
            raise ValueError("--text cannot be empty.")
    else:
        if args.audio_file is not None:
            audio = load_audio_file(args.audio_file)
            input_source = str(args.audio_file.expanduser().resolve())
        else:
            audio = record_microphone(args.record_seconds)
            input_source = f"microphone_{args.record_seconds:g}_seconds"

        if args.skip_audio_preprocessing:
            processed_audio = audio
            audio_metadata = {
                "sample_rate_hz": TARGET_SAMPLE_RATE,
                "original_duration_seconds": round(audio.size / TARGET_SAMPLE_RATE, 3),
                "processed_duration_seconds": round(audio.size / TARGET_SAMPLE_RATE, 3),
                "preprocessing_skipped": True,
            }
        else:
            print("Applying audio preprocessing...")
            processed_audio, audio_metadata = preprocess_audio(audio)

        if args.save_processed_audio is not None:
            saved_path = save_audio(args.save_processed_audio, processed_audio)
            audio_metadata["saved_processed_audio"] = str(saved_path)
            print(f"Saved processed audio: {saved_path}")

        transcript = transcribe_audio(
            processed_audio,
            model_name=args.whisper_model,
            language=args.language,
            device=device,
            chunk_seconds=args.chunk_seconds,
        )

    processed_text = preprocess_text(transcript)
    if not processed_text:
        raise RuntimeError("No usable text remained after transcript preprocessing.")

    symptom_extraction: dict[str, Any] | None = None
    biobert_input = processed_text
    if not args.disable_symptom_extraction:
        extractor = SymptomExtractor(args.symptom_vocabulary)
        symptom_extraction = extractor.extract(
            transcript,
            minimum_present_symptoms=args.minimum_extracted_symptoms,
        )
        biobert_input = symptom_extraction["biobert_text"]

    urgency: dict[str, Any] | None = None
    if not args.disable_urgency_detection:
        safety_extraction = symptom_extraction
        if safety_extraction is None:
            safety_extraction = SymptomExtractor(args.symptom_vocabulary).extract(
                transcript,
                minimum_present_symptoms=args.minimum_extracted_symptoms,
            )
        urgency = detect_urgency(
            transcript,
            safety_extraction,
            emergency_number=args.emergency_number.strip(),
            urgent_service=args.urgent_service.strip(),
        )

    clarification = generate_clarification(
        transcript,
        symptom_extraction,
        urgency=urgency,
        max_questions=args.max_clarification_questions,
    )

    if urgency is not None and urgency["should_bypass_classification"]:
        classification = skipped_classification(
            f"classification_bypassed_{urgency['urgency_level']}_urgency",
            args.minimum_score,
        )
    elif clarification["should_pause_classification"]:
        reason_codes = set(clarification["reason_codes"])
        if "no_current_symptoms" in reason_codes:
            status = "no_current_symptoms_request_clarification"
        elif "uncertain_symptoms" in reason_codes:
            status = "uncertain_symptoms_request_clarification"
        else:
            status = "insufficient_symptoms_request_clarification"
        classification = skipped_classification(status, args.minimum_score)
    elif biobert_input:
        classification = classify_text(
            biobert_input,
            model_dir=args.model_dir,
            device=device,
            top_k=args.top_k,
            max_length=args.max_length,
            minimum_score=args.minimum_score,
            minimum_margin=args.minimum_margin,
        )
        clarification = generate_clarification(
            transcript,
            symptom_extraction,
            urgency=urgency,
            classification=classification,
            max_questions=args.max_clarification_questions,
        )
        if clarification["should_withhold_prediction"]:
            if classification["status"] == "ambiguous_prediction_review_symptoms":
                classification["status"] = "ambiguous_prediction_request_clarification"
            else:
                classification["status"] = "low_score_request_clarification"
            classification["display_prediction"] = False
    else:
        classification = skipped_classification(
            "no_current_symptoms_request_clarification",
            args.minimum_score,
        )

    rule_based_response = None
    if not args.disable_rule_based_response:
        rule_based_response = generate_rule_based_response(
            transcript,
            symptom_extraction,
            urgency,
            clarification,
            classification,
        )

    result = {
        "input_source": input_source,
        "audio_preprocessing": audio_metadata,
        "whisper_model": None if args.text is not None else args.whisper_model,
        "transcript": transcript,
        "processed_text": processed_text,
        "symptom_extraction_enabled": not args.disable_symptom_extraction,
        "symptom_extraction": symptom_extraction,
        "biobert_input": biobert_input,
        "urgency_detection_enabled": not args.disable_urgency_detection,
        "urgency": urgency,
        "clarification": clarification,
        "biobert_model_dir": str(args.model_dir.expanduser().resolve()),
        "classification": classification,
        "rule_based_response_enabled": not args.disable_rule_based_response,
        "rule_based_response": rule_based_response,
        "safety_notice": (
            "Research use only. The predicted class is not a medical diagnosis "
            "and must not replace professional medical assessment."
        ),
    }
    print_result(result)

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved result JSON: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
