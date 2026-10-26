# Local setup

## Prerequisites

- An account with access to the private repository and model release.
- Python 3.10 or 3.11.
- Node.js 20.19+ or 22.12+ for Vite 7, and pnpm.
- Audio dependencies from `medical_chatbot_app/backend/requirements.txt`.
- FFmpeg for the audio workflows described in the application documentation.
  Browser WebM is decoded by PyAV in the current backend.

The Windows PowerShell launch scripts are the supported convenience entry
points. CPU inference works; a CUDA-compatible PyTorch installation is optional
for local inference and must match the hardware when used.

## Restore the final trained model

From [the private model release](https://github.com/sean8911/medvoice-assist/releases/tag/initial-commit),
download `biobert-final.zip` and `biobert-final.sha256` into the repository root.
Do not put a GitHub token into application source or a frontend environment
variable. A signed-in browser with repository access can download the assets.

Verify and extract from the repository root:

```powershell
$expected = ((Get-Content -LiteralPath .\biobert-final.sha256).Trim() -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath .\biobert-final.zip -Algorithm SHA256).Hash
if ($actual.ToLowerInvariant() -ne $expected.ToLowerInvariant()) {
    throw "Model archive checksum does not match. Do not extract this download."
}
Expand-Archive -LiteralPath .\biobert-final.zip -DestinationPath ".\3 Models After Training\outputs\models" -Force
```

Use `-Force` only in a fresh clone or after checking/backing up an existing
checkpoint. It restores the archived originals of the tracked metadata and adds
the missing weights. The resulting directory must contain:

```text
3 Models After Training/outputs/models/biobert/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  MANIFEST.sha256
```

These four inference files were packaged unchanged from the final local trained
checkpoint. Their original hashes and the archive hash are recorded in
[model_artifacts.json](model_artifacts.json). The model has 773 output labels.
Earlier BERT/BioBERT checkpoints and the TF-IDF joblib are not included.

## First installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd medical_chatbot_app
.\setup.ps1
.\start_app.ps1
```

Setup installs the backend dependencies, installs frontend dependencies and
builds `frontend/dist`. The launcher opens http://127.0.0.1:8000. Whisper-small
is downloaded and cached on its first use; an internet connection is needed for
that first download. The trained BioBERT is read from the restored local path.
No external generative-response API key is needed.

If PowerShell execution policy prevents activation, use a Python/Conda
environment already configured for the project, or ask the machine administrator
for the approved activation method. Do not bypass institutional security policy.

## Later sessions

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
cd medical_chatbot_app
.\start_app.ps1
```

Do not retrain the model or reinstall dependencies on every launch. Keep the
launch terminal open while using the app. The script stops its own server when
the session ends.

## Development mode

From the repository root, with the backend environment active:

```powershell
python -m uvicorn medical_chatbot_app.backend.app.main:app --reload
```

In another terminal:

```powershell
cd medical_chatbot_app/frontend
pnpm dev
```

Open http://127.0.0.1:5173. Vite forwards `/api` requests to FastAPI.

## Configuration and troubleshooting

Optional settings are listed in
[the environment example](../medical_chatbot_app/.env.example). The application
reads environment variables; copying that file alone does not automatically load
them. For example, before starting from PowerShell:

```powershell
$env:MEDICAL_CHATBOT_DEVICE = "cpu"
```

- A missing-model error usually means that the separate release asset has not
  been extracted into the expected directory.
- A backend error should be checked in `medical_chatbot_app/backend/logs/`.
  Do not upload logs containing sensitive narratives.
- First-use model loading and CPU transcription may be slow. Keep the same
  Whisper model and document hardware when comparing timings.
- A request for clarification or an explained no-prediction response can be the
  expected result of evidence/score gates, not an installation failure.

The health endpoint is `/api/health`, and API documentation is at `/docs`.
This setup is local; no hosting, public endpoint or security certification is
created by cloning the repository.
