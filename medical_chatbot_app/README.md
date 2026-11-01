# MedVoice Assist

MedVoice Assist is the Thesis 2 full-stack medical-chatbot prototype. It is a
React application with a FastAPI backend and uses the models and safety modules
stored in this workspace. No OpenAI API key, Streamlit account, or external
generative-response API is required.

## Pipeline

```text
Microphone or uploaded audio
  -> audio preprocessing
  -> Whisper transcription
  -> user transcript confirmation
  -> text preprocessing
  -> narrative-to-symptom extraction
  -> urgency detection
  -> clarification questions when needed
  -> trained BioBERT disease classification
  -> score and ambiguity checks
  -> source-backed rule-based response
```

The backend uses the final trained 773-class model at:

```text
../3 Models After Training/outputs/models/biobert
```

The model path can be changed with `MEDICAL_CHATBOT_BIOBERT_MODEL`, but the
workspace model above is the default and is validated when `/api/health` runs.

## Interface features

- Text symptom input
- Browser microphone recording
- WAV, MP3, M4A, OGG, FLAC, AAC, or WebM audio upload
- Audio denoising toggle
- Editable Whisper transcript before medical analysis
- Free-form symptom descriptions with common conversational alias matching
- Inline current, negated, and uncertain symptom confirmation tags
- Emergency, urgent, clarification, routine, and no-prediction response designs
- One concise, adaptive clarification question with no required answer format
- One clarification round before a professional-assessment fallback
- BioBERT prediction hidden when its score is low or top classes are ambiguous
- Expandable authoritative guidance sources
- Thesis research panel with stage results, timings, top classes, and JSON export
- Responsive desktop and mobile layout

## Privacy behaviour

No database is configured. Conversation state stays in the browser tab. Uploaded
audio is written only to a temporary backend file and deleted immediately after
transcription. JSON is downloaded only when the user presses the download button.

## Requirements

- Python 3.10 or 3.11
- Node.js 20.19+ or 22.12+
- pnpm
- Audio dependencies described in [Setup](../docs/SETUP.md); browser uploads use PyAV
- CUDA-compatible PyTorch is optional; CPU inference is supported

## First setup

From the cloned repository root, with your Python environment active:

```powershell
cd medical_chatbot_app
.\setup.ps1
```

The setup installs FastAPI and the existing voice-pipeline requirements, installs
the React packages, and creates `frontend/dist`.

## Start the application

From `medical_chatbot_app/`:

```powershell
.\start_app.ps1
```

The launcher starts the local backend, opens `http://127.0.0.1:8000`, and stops
the server when its PowerShell session is closed. Server logs are written under
`backend/logs`.

## Development mode

Backend terminal, from the repository root:

```powershell
python -m uvicorn medical_chatbot_app.backend.app.main:app --reload
```

Frontend terminal, also starting from the repository root:

```powershell
cd medical_chatbot_app/frontend
pnpm dev
```

Open `http://127.0.0.1:5173`. Vite forwards `/api` requests to FastAPI.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Validate model path and report runtime status |
| POST | `/api/transcribe` | Preprocess and transcribe temporary audio |
| POST | `/api/analyze/text` | Run safety, clarification, BioBERT, and response stages |
| POST | `/api/analyze/clarification` | Append one answer and rerun the complete text pipeline |
| GET | `/docs` | Interactive FastAPI schema |

## Verification

From the repository root:

```powershell
python -m unittest discover -s medical_chatbot_app\backend\tests -t . -v

cd medical_chatbot_app\frontend
pnpm lint
pnpm build
```

## Safety limitation

This is a research prototype, not a diagnosis or validated clinical triage
system. The urgency and response rules require review by a qualified healthcare
professional before any study involving real users. A routine result only means
that no configured warning-sign rule matched.
