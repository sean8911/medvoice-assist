from __future__ import annotations

import logging
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .. import config


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

if str(config.PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(config.PIPELINE_DIR))

from voice_to_disease import TARGET_SAMPLE_RATE, resolve_device, validate_biobert_model_dir  # noqa: E402


LOGGER = logging.getLogger(__name__)


class ModelRuntime:
    """Lazily load and reuse Whisper and the final trained BioBERT classifier."""

    def __init__(
        self,
        model_dir: Path = config.BIOBERT_MODEL_DIR,
        whisper_model: str = config.WHISPER_MODEL,
        requested_device: str = config.DEVICE,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.whisper_model_name = whisper_model
        self.device = resolve_device(requested_device)
        self._biobert_tokenizer: Any | None = None
        self._biobert_model: Any | None = None
        self._whisper_processor: Any | None = None
        self._whisper_model: Any | None = None
        self._biobert_load_lock = threading.Lock()
        self._whisper_load_lock = threading.Lock()
        self._biobert_inference_lock = threading.Lock()
        self._whisper_inference_lock = threading.Lock()

    @property
    def biobert_loaded(self) -> bool:
        return self._biobert_model is not None

    @property
    def whisper_loaded(self) -> bool:
        return self._whisper_model is not None

    def health(self) -> dict[str, Any]:
        model_valid = False
        model_error: str | None = None
        try:
            validate_biobert_model_dir(self.model_dir)
            model_valid = True
        except Exception as exc:  # Health reports errors without crashing the API.
            model_error = str(exc)
        return {
            "device": self.device,
            "biobert_model_dir": str(self.model_dir),
            "biobert_model_valid": model_valid,
            "biobert_model_error": model_error,
            "biobert_loaded": self.biobert_loaded,
            "whisper_model": self.whisper_model_name,
            "whisper_loaded": self.whisper_loaded,
            "disease_classes": 773,
        }

    def _load_biobert(self) -> tuple[Any, Any]:
        if self._biobert_model is not None:
            return self._biobert_tokenizer, self._biobert_model

        with self._biobert_load_lock:
            if self._biobert_model is not None:
                return self._biobert_tokenizer, self._biobert_model

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            validate_biobert_model_dir(self.model_dir)
            LOGGER.info("Loading trained BioBERT model from %s", self.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
            model.to(self.device)
            if self.device == "cuda":
                model.half()
            model.eval()
            self._biobert_tokenizer = tokenizer
            self._biobert_model = model
            return tokenizer, model

    def classify(
        self,
        text: str,
        top_k: int = config.TOP_K,
        max_length: int = config.MAX_LENGTH,
        minimum_score: float = config.MINIMUM_SCORE,
        minimum_margin: float = config.MINIMUM_MARGIN,
    ) -> dict[str, Any]:
        import torch

        tokenizer, model = self._load_biobert()
        with self._biobert_inference_lock:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits[0]
                probabilities = torch.softmax(logits.float(), dim=-1)

        requested_top_k = min(top_k, int(probabilities.shape[0]))
        scores, indexes = torch.topk(probabilities, k=requested_top_k)
        top_predictions = [
            {
                "rank": rank,
                "label_id": int(label_id),
                "disease": str(model.config.id2label[int(label_id)]),
                "model_score": round(float(score), 6),
            }
            for rank, (label_id, score) in enumerate(
                zip(indexes.cpu().tolist(), scores.cpu().tolist()),
                start=1,
            )
        ]
        best = top_predictions[0]
        second_score = (
            top_predictions[1]["model_score"] if len(top_predictions) > 1 else 0.0
        )
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
            "number_of_disease_classes": int(probabilities.shape[0]),
            "display_prediction": status == "prediction_available",
        }

    def _load_whisper(self) -> tuple[Any, Any]:
        if self._whisper_model is not None:
            return self._whisper_processor, self._whisper_model

        with self._whisper_load_lock:
            if self._whisper_model is not None:
                return self._whisper_processor, self._whisper_model

            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            LOGGER.info("Loading Whisper model %s", self.whisper_model_name)
            processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.whisper_model_name
            )
            model.to(self.device)
            if self.device == "cuda":
                model.half()
            model.eval()
            self._whisper_processor = processor
            self._whisper_model = model
            return processor, model

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        chunk_seconds: float = 25.0,
    ) -> str:
        import torch

        processor, model = self._load_whisper()
        chunk_size = max(1, int(chunk_seconds * TARGET_SAMPLE_RATE))
        chunks = [
            audio[start : start + chunk_size]
            for start in range(0, audio.size, chunk_size)
        ]
        transcripts: list[str] = []

        with self._whisper_inference_lock:
            for chunk in chunks:
                inputs = processor(
                    chunk,
                    sampling_rate=TARGET_SAMPLE_RATE,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                generation_inputs: dict[str, Any] = {
                    "input_features": inputs.input_features.to(self.device),
                    "language": language,
                    "task": "transcribe",
                }
                attention_mask = getattr(inputs, "attention_mask", None)
                if attention_mask is not None:
                    generation_inputs["attention_mask"] = attention_mask.to(
                        self.device
                    )
                with torch.inference_mode():
                    predicted_ids = model.generate(**generation_inputs)
                text = processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True,
                )[0].strip()
                if text:
                    transcripts.append(text)

        transcript = " ".join(transcripts).strip()
        if not transcript:
            raise RuntimeError("Whisper returned an empty transcript.")
        return transcript


@lru_cache(maxsize=1)
def get_model_runtime() -> ModelRuntime:
    return ModelRuntime()
