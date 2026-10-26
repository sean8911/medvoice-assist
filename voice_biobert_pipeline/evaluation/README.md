# Medical Speech and Noise Evaluation

This evaluation compares the thesis audio paths while keeping Whisper and the
final trained BioBERT classifier fixed:

```text
raw audio -> Whisper -> BioBERT
preprocessed audio -> Whisper -> BioBERT
```

Each medical recording is tested clean and with RIRS background noise at 20,
10, 5, and 0 dB SNR. The evaluator reports WER, CER, BioBERT prediction
stability, low-score rate, and disease accuracy only for manually mapped
PriMock57 cases.

## Local datasets

- `datasets/medical_speech_intent`: 6,661 symptom utterances from 124 speakers.
- `datasets/primock57`: simulated primary-care consultation audio and TextGrid transcripts.
- `datasets/rirs_noises`: OpenSLR room responses and 843 point-source noises.

The Kaggle package contains two copies of the same 6,661-item dataset. The
evaluator locates one complete copy and does not double count it.

## Install evaluation dependencies

Install the main pipeline requirements first, followed by the evaluation extras:

```powershell
cd "C:\Year 3 Sem 3\Thesis\Denoise + Whisper\voice_biobert_pipeline"
pip install -r requirements.txt
pip install -r evaluation\requirements.txt
```

## Verify dataset inventory

```powershell
python evaluation\evaluate_audio_datasets.py --inventory-only
```

## Small validation run

One sample from each medical dataset, clean and at all four noise levels:

```powershell
python evaluation\evaluate_audio_datasets.py `
  --samples-per-dataset 1 `
  --snr-levels 20 10 5 0 `
  --device auto
```

## Larger thesis run

```powershell
python evaluation\evaluate_audio_datasets.py `
  --samples-per-dataset 3 `
  --snr-levels 20 10 5 0 `
  --device cuda
```

Outputs are saved incrementally under `evaluation/outputs`:

- `dataset_inventory.json`
- `dataset_test_results.csv`
- `dataset_test_summary.csv`

Medical Speech prompt labels are known to contain errors and are not treated as
disease ground truth. PriMock57 uses only patient-channel utterances and three
manually checked cases whose diagnoses exactly exist in the 773 BioBERT labels.
The selected PriMock57 intervals exclude administrative identity checks,
clinician speech, and diagnosis disclosure.
