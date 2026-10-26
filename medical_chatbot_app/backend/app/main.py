from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .schemas import ClarificationAnswerRequest, TextAnalysisRequest
from .services.model_runtime import get_model_runtime
from .services.pipeline_service import PipelineService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Voice Assistant API",
    description=(
        "Research prototype integrating audio preprocessing, Whisper, symptom "
        "extraction, urgency rules, the trained 773-class BioBERT model, "
        "clarification questions, and rule-based responses."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

service = PipelineService()


def api_error(exc: Exception) -> HTTPException:
    LOGGER.exception("Application request failed")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(
        status_code=500,
        detail="The pipeline could not complete this request. Check the backend logs.",
    )


@app.get("/api")
def api_information() -> dict[str, Any]:
    return {
        "name": "Medical Voice Assistant API",
        "version": app.version,
        "research_use_only": True,
        "docs": "/docs",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    runtime = get_model_runtime()
    return {
        "status": "ok",
        "application": "medical-voice-assistant",
        "runtime": runtime.health(),
        "privacy": "No database is configured; uploaded audio is temporary.",
    }


@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    preprocess: bool = Form(True),
    language: str = Form("en"),
) -> dict[str, Any]:
    try:
        content = await audio.read()
        return await run_in_threadpool(
            service.transcribe_upload,
            content,
            audio.filename or "recording.wav",
            preprocess,
            language,
        )
    except Exception as exc:
        raise api_error(exc) from exc
    finally:
        await audio.close()


@app.post("/api/analyze/text")
async def analyze_text(payload: TextAnalysisRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(service.analyze_text, payload.text)
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/api/analyze/clarification")
async def analyze_clarification(
    payload: ClarificationAnswerRequest,
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            service.analyze_clarification,
            payload.original_narrative,
            payload.answer,
        )
    except Exception as exc:
        raise api_error(exc) from exc


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        requested = (FRONTEND_DIST / path).resolve()
        if requested.is_file() and FRONTEND_DIST.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
