from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = WORKSPACE_ROOT / "voice_biobert_pipeline"

BIOBERT_MODEL_DIR = Path(
    os.environ.get(
        "MEDICAL_CHATBOT_BIOBERT_MODEL",
        str(
            WORKSPACE_ROOT
            / "3 Models After Training"
            / "outputs"
            / "models"
            / "biobert"
        ),
    )
).expanduser()

WHISPER_MODEL = os.environ.get(
    "MEDICAL_CHATBOT_WHISPER_MODEL", "openai/whisper-small"
)
DEVICE = os.environ.get("MEDICAL_CHATBOT_DEVICE", "auto")
EMERGENCY_NUMBER = os.environ.get("MEDICAL_CHATBOT_EMERGENCY_NUMBER", "999")
URGENT_SERVICE = os.environ.get(
    "MEDICAL_CHATBOT_URGENT_SERVICE", "a same-day medical service"
)

MINIMUM_SCORE = float(os.environ.get("MEDICAL_CHATBOT_MINIMUM_SCORE", "0.50"))
MINIMUM_MARGIN = float(os.environ.get("MEDICAL_CHATBOT_MINIMUM_MARGIN", "0.10"))
MINIMUM_EXTRACTED_SYMPTOMS = int(
    os.environ.get("MEDICAL_CHATBOT_MINIMUM_SYMPTOMS", "2")
)
MAX_CLARIFICATION_QUESTIONS = int(
    os.environ.get("MEDICAL_CHATBOT_MAX_QUESTIONS", "1")
)
MAX_LENGTH = int(os.environ.get("MEDICAL_CHATBOT_MAX_LENGTH", "128"))
TOP_K = int(os.environ.get("MEDICAL_CHATBOT_TOP_K", "3"))
MAX_UPLOAD_BYTES = int(
    os.environ.get("MEDICAL_CHATBOT_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))
)

ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in os.environ.get(
        "MEDICAL_CHATBOT_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
)
