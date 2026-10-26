# MedVoice Assist

Speech-enabled medical-chatbot research prototype for a final-year thesis.
The application combines a React interface, a FastAPI backend, audio
preprocessing, Whisper transcription, deterministic symptom extraction, urgency
screening, clarification, a locally trained BioBERT classifier and rule-based
responses. Text input is also supported.

This repository is a private source and report snapshot, not an online
deployment or a complete backup of every file on the development laptop.

## Pipeline

```text
Voice input -> audio preprocessing -> Whisper -> transcript confirmation
                                                   |
Typed input ---------------------------------------+
                                                   v
Text preparation -> symptom extraction -> urgency screening
  -> bounded clarification -> trained BioBERT -> prediction checks
  -> rule-based response
```

Urgent and emergency rules can bypass classification. A withheld disease class
is a supported outcome, not a reason to add symptoms just to obtain a score.

## Project layout

| Location | Contents |
| --- | --- |
| `medical_chatbot_app/` | React frontend, FastAPI backend, setup and launch scripts |
| `voice_biobert_pipeline/` | Speech processing, symptom extraction, urgency, clarification and response modules |
| `voice_biobert_pipeline/evaluation/` | Existing speech-evaluation scripts and retained reports |
| `TF-IDF and Fine Tune BERT/` | Three-classifier training and comparison scripts |
| `3 Models After Training/outputs/` | Expanded-dataset reports, logs and model metadata |
| Root notebooks | Preliminary classifier development and comparisons |
| `scripts/` | Notebook, figure and thesis-document utilities |
| `tests/` | Notebook and experiment checks |
| `docs/` | Architecture, thesis drafts, algorithms and supporting material |

## Restore the trained BioBERT

The application uses the final **773-class** checkpoint, not an unfine-tuned
BioBERT download. Its default path is:

```text
3 Models After Training/outputs/models/biobert/
```

Large weights are deliberately excluded from Git history. Download
`biobert-final.zip` and `biobert-final.sha256` from the private
[initial project release](https://github.com/sean8911/medvoice-assist/releases/tag/initial-commit).
The archive includes the four required inference files and a SHA-256 manifest.
The asset checksums are also recorded in `docs/model_artifacts.json`.

From the repository root, after downloading the archive into that root:

```powershell
Get-FileHash -LiteralPath .\biobert-final.zip -Algorithm SHA256
Expand-Archive -LiteralPath .\biobert-final.zip -DestinationPath ".\3 Models After Training\outputs\models"
```

Compare the displayed hash with the checksum file before extracting. These
instructions assume a fresh clone; do not overwrite another model checkpoint
without checking it first. Cloning the source alone does not download the model.

## Run locally on Windows

Use Python 3.10 or 3.11, Node.js 20.19+ or 22.12+, pnpm and the audio dependencies
described in the [application README](medical_chatbot_app/README.md). Activate
the Python environment intended for this project first.

```powershell
cd medical_chatbot_app
.\setup.ps1
.\start_app.ps1
```

The application opens at `http://127.0.0.1:8000`. A new clone needs its frontend
dependencies installed and `frontend/dist` rebuilt by setup. Public Whisper
weights are downloaded on first use; the restored BioBERT is loaded locally.
No external generative-response API key is required.

## Datasets and retained results

The work compares TF-IDF with Logistic Regression, fine-tuned BERT and BioBERT
in two dataset settings: the preliminary 24-class Symptom2Disease experiment and
the expanded disease-diagnosis experiment. Obtain the expanded dataset from
[its source](https://huggingface.co/datasets/sajjadhadi/disease-diagnosis-dataset)
and follow the training-script documentation for its expected cleaned files.
Those large prepared datasets are not included in this Git repository.

Preliminary and expanded outputs must not be mixed. The retained supplied-split
results have documented duplicate-text overlap and do not establish independent
generalisation or clinical validity. The earlier speech pilot is not a completed
evaluation of every stage in the latest chatbot. Planned participant studies
must not be reported as completed without genuine collected records.

## Tests

From the repository root:

```powershell
python -m unittest discover -s medical_chatbot_app/backend/tests -t . -v
python -m unittest discover -s voice_biobert_pipeline/tests -v
```

Some API tests use controlled model doubles. Passing those tests is not a
measurement of the trained model's medical accuracy.

## Repository scope and privacy

Downloaded speech/noise datasets, large training data, duplicate checkpoints,
other model-weight files, dependency folders, scratch renders, raw recordings,
local conversations and server logs remain on the laptop. No local files are
deleted by these ignore rules. See [repository contents](docs/GITHUB_CONTENTS.md)
for the distinction between source, release assets and excluded local data.

The application is a research prototype, **not a diagnostic service or a
clinically validated urgency system**. Source-linked rules still require
qualified review. Model scores are not validated probabilities of disease.
The repository being private does not replace participant consent or secure
handling of research data. No deployment is created by uploading this project.
