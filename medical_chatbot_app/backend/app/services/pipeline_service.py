from __future__ import annotations

import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .. import config
from .model_runtime import ModelRuntime, get_model_runtime


if str(config.PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(config.PIPELINE_DIR))

from clarification_questions import generate_clarification  # noqa: E402
from rule_based_response import generate_rule_based_response  # noqa: E402
from symptom_extractor import SymptomExtractor  # noqa: E402
from urgency_detector import detect_urgency  # noqa: E402
from voice_to_disease import (  # noqa: E402
    load_audio_file,
    preprocess_audio,
    preprocess_text,
    skipped_classification,
    validate_audio,
)


class PipelineService:
    def __init__(self, runtime: ModelRuntime | None = None) -> None:
        self.runtime = runtime or get_model_runtime()
        self.extractor = SymptomExtractor()

    def transcribe_upload(
        self,
        content: bytes,
        filename: str,
        preprocess: bool = True,
        language: str = "en",
    ) -> dict[str, Any]:
        if not content:
            raise ValueError("The uploaded audio file is empty.")
        if len(content) > config.MAX_UPLOAD_BYTES:
            max_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
            raise ValueError(f"Audio files must be {max_mb} MB or smaller.")

        started = time.perf_counter()
        suffix = Path(filename or "recording.wav").suffix.lower()
        if suffix not in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}:
            suffix = ".wav"

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(content)
                temporary_path = Path(handle.name)
            audio = self._load_uploaded_audio(temporary_path)
            if preprocess:
                processed_audio, metadata = preprocess_audio(audio)
            else:
                processed_audio = audio
                metadata = {
                    "sample_rate_hz": 16_000,
                    "original_duration_seconds": round(audio.size / 16_000, 3),
                    "processed_duration_seconds": round(audio.size / 16_000, 3),
                    "preprocessing_skipped": True,
                }
            transcript = self.runtime.transcribe(
                processed_audio,
                language=language,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return {
            "request_id": str(uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "filename": filename,
            "transcript": transcript,
            "audio_preprocessing": metadata,
            "whisper_model": self.runtime.whisper_model_name,
            "device": self.runtime.device,
            "processing_time_seconds": round(time.perf_counter() - started, 3),
            "privacy_notice": "The temporary uploaded audio file was deleted after transcription.",
        }

    @staticmethod
    def _load_uploaded_audio(path: Path) -> np.ndarray:
        """Decode browser WebM/Opus recordings without requiring an FFmpeg executable."""
        if path.suffix.lower() != ".webm":
            try:
                return load_audio_file(path)
            except Exception as original_error:
                try:
                    return PipelineService._decode_with_pyav(path)
                except Exception:
                    raise original_error
        return PipelineService._decode_with_pyav(path)

    @staticmethod
    def _decode_with_pyav(path: Path) -> np.ndarray:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "Browser audio support is missing. Run setup.ps1 again to install PyAV."
            ) from exc

        chunks: list[np.ndarray] = []
        try:
            with av.open(str(path)) as container:
                audio_streams = [
                    stream for stream in container.streams if stream.type == "audio"
                ]
                if not audio_streams:
                    raise ValueError("The uploaded recording does not contain an audio stream.")

                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16_000)
                for frame in container.decode(audio_streams[0]):
                    for resampled in resampler.resample(frame):
                        chunks.append(resampled.to_ndarray().reshape(-1).astype(np.float32))
                for resampled in resampler.resample(None):
                    chunks.append(resampled.to_ndarray().reshape(-1).astype(np.float32))
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise ValueError(
                "The browser recording could not be decoded. Please record again or upload "
                "a WAV, MP3, M4A, OGG, FLAC, AAC, or WebM audio file."
            ) from exc

        if not chunks:
            raise ValueError("No audio samples were found in the uploaded recording.")
        return validate_audio(np.concatenate(chunks), source=str(path))

    def analyze_text(self, narrative: str) -> dict[str, Any]:
        narrative = narrative.strip()
        if not narrative:
            raise ValueError("Symptom text cannot be empty.")

        started = time.perf_counter()
        stage_timings: dict[str, float] = {}

        stage_started = time.perf_counter()
        processed_text = preprocess_text(narrative)
        if not processed_text:
            raise ValueError("No usable symptom text remained after preprocessing.")
        stage_timings["text_preprocessing"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        extraction = self.extractor.extract(
            narrative,
            minimum_present_symptoms=config.MINIMUM_EXTRACTED_SYMPTOMS,
        )
        biobert_input = extraction["biobert_text"]
        stage_timings["symptom_extraction"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        urgency = detect_urgency(
            narrative,
            extraction,
            emergency_number=config.EMERGENCY_NUMBER,
            urgent_service=config.URGENT_SERVICE,
        )
        stage_timings["urgency_detection"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        clarification = generate_clarification(
            narrative,
            extraction,
            urgency=urgency,
            max_questions=config.MAX_CLARIFICATION_QUESTIONS,
        )
        stage_timings["clarification_check"] = time.perf_counter() - stage_started

        classification_ran = False
        stage_started = time.perf_counter()
        if urgency["should_bypass_classification"]:
            classification = skipped_classification(
                f"classification_bypassed_{urgency['urgency_level']}_urgency",
                config.MINIMUM_SCORE,
            )
        elif clarification["should_pause_classification"]:
            reasons = set(clarification["reason_codes"])
            if "no_current_symptoms" in reasons:
                status = "no_current_symptoms_request_clarification"
            elif "uncertain_symptoms" in reasons:
                status = "uncertain_symptoms_request_clarification"
            else:
                status = "insufficient_symptoms_request_clarification"
            classification = skipped_classification(status, config.MINIMUM_SCORE)
        elif biobert_input:
            classification_ran = True
            classification = self.runtime.classify(
                biobert_input,
                top_k=config.TOP_K,
                max_length=config.MAX_LENGTH,
                minimum_score=config.MINIMUM_SCORE,
                minimum_margin=config.MINIMUM_MARGIN,
            )
            clarification = generate_clarification(
                narrative,
                extraction,
                urgency=urgency,
                classification=classification,
                max_questions=config.MAX_CLARIFICATION_QUESTIONS,
            )
            if clarification["should_withhold_prediction"]:
                if classification["status"] == "ambiguous_prediction_review_symptoms":
                    classification["status"] = (
                        "ambiguous_prediction_request_clarification"
                    )
                else:
                    classification["status"] = "low_score_request_clarification"
                classification["display_prediction"] = False
        else:
            classification = skipped_classification(
                "no_current_symptoms_request_clarification",
                config.MINIMUM_SCORE,
            )
        classification["inference_performed"] = classification_ran
        stage_timings["biobert_classification"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        response = generate_rule_based_response(
            narrative,
            extraction,
            urgency,
            clarification,
            classification,
        )
        stage_timings["rule_based_response"] = time.perf_counter() - stage_started

        pipeline_trace = [
            {"stage": "text_preprocessing", "status": "completed"},
            {"stage": "symptom_extraction", "status": "completed"},
            {"stage": "urgency_detection", "status": "completed"},
            {
                "stage": "clarification",
                "status": (
                    "required"
                    if clarification["clarification_required"]
                    else "completed"
                ),
            },
            {
                "stage": "biobert_classification",
                "status": "completed" if classification_ran else "bypassed",
            },
            {"stage": "rule_based_response", "status": "completed"},
        ]

        return {
            "request_id": str(uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "input_source": "confirmed_text",
            "transcript": narrative,
            "processed_text": processed_text,
            "symptom_extraction": extraction,
            "biobert_input": biobert_input,
            "urgency": urgency,
            "clarification": clarification,
            "classification": classification,
            "rule_based_response": response,
            "pipeline_trace": pipeline_trace,
            "stage_timings_seconds": {
                name: round(duration, 4) for name, duration in stage_timings.items()
            },
            "processing_time_seconds": round(time.perf_counter() - started, 3),
            "model_runtime": {
                "device": self.runtime.device,
                "biobert_model_dir": str(self.runtime.model_dir),
                "biobert_loaded": self.runtime.biobert_loaded,
                "whisper_model": self.runtime.whisper_model_name,
                "whisper_loaded": self.runtime.whisper_loaded,
            },
            "safety_notice": (
                "Research use only. This output is not a diagnosis and must not "
                "replace professional medical assessment."
            ),
        }

    def analyze_clarification(self, original_narrative: str, answer: str) -> dict[str, Any]:
        combined = f"{original_narrative.strip()} {answer.strip()}".strip()
        result = self.analyze_text(combined)
        if result["rule_based_response"]["response_type"] == "clarification":
            pre_limit_status = result["classification"].get("status")
            result["classification"]["status"] = "clarification_limit_reached"
            result["classification"]["pre_limit_status"] = pre_limit_status
            result["classification"]["display_prediction"] = False
            closed_clarification = {
                **result["clarification"],
                "clarification_required": False,
                "should_pause_classification": False,
                "should_withhold_prediction": False,
                "questions": [],
                "question_count": 0,
                "reason_codes": [
                    *result["clarification"].get("reason_codes", []),
                    "clarification_round_limit",
                ],
                "message": (
                    "The one-round clarification limit was reached and no result "
                    "met the requirements for display."
                ),
            }
            result["clarification"] = closed_clarification
            for stage in result["pipeline_trace"]:
                if stage["stage"] == "clarification":
                    stage["status"] = "limit_reached"
            result["rule_based_response"] = generate_rule_based_response(
                combined,
                result["symptom_extraction"],
                result["urgency"],
                closed_clarification,
                result["classification"],
            )
        result["input_source"] = "clarification_answer"
        result["conversation"] = {
            "original_narrative": original_narrative.strip(),
            "clarification_answer": answer.strip(),
            "combined_narrative": combined,
        }
        return result
