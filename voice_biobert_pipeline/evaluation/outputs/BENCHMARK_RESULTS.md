# Pilot Dataset Benchmark Results

## Scope

This pilot used six speaker/sample cases:

- Three speaker-balanced recordings from Medical Speech, Transcription, and Intent.
- Three manually curated patient-only cases from PriMock57: migraine, urinary tract infection, and asthma.
- One fixed RIRS background-noise segment at clean, 20, 10, 5, and 0 dB conditions.
- Raw and preprocessed audio paths for every condition, producing 60 result rows.

The final local BioBERT model from `3 Models After Training/outputs/models/biobert`
was used without retraining.

## Whisper Results

Medical Speech had low mean clean WER of 0.0152 for both raw and preprocessed
audio. At 0 dB, raw WER was 0.0788 and preprocessed WER was 0.1242.

PriMock57 was harder conversational speech. Clean WER changed from 0.2973 raw
to 0.2744 after preprocessing. At 5 dB it changed from 0.4795 to 0.4562, and at
0 dB from 1.2229 to 1.2053. Preprocessing was slightly worse at 20 and 10 dB.

Because there were only three samples per medical dataset, these differences
are descriptive pilot results and are not yet evidence of statistical improvement.

## BioBERT Results

Every evaluated BioBERT output was below the 0.50 score threshold. Disease
accuracy on the three manually mapped PriMock57 cases was 0%. Importantly, the
model also failed on the human reference transcripts before Whisper errors were
introduced:

- Migraine narrative -> hyperkalemia.
- Urinary tract infection narrative -> pilonidal cyst.
- Asthma narrative -> hypertensive heart disease.

When the migraine narrative was manually converted into the training style,
`headache, nausea, vomiting, spots or clouds in vision`, the same trained model
predicted migraine at 53.23%. This indicates a text-domain mismatch: the model
was trained on short comma-separated symptom lists but the chatbot receives
natural conversational narratives.

## Interpretation

The audio-to-text pipeline is functioning, and Whisper is strong on short symptom
utterances. The present BioBERT classifier should not yet drive medical responses
from natural voice input. The next required component is a validated
narrative-to-symptom extraction stage, or additional BioBERT training with
conversational symptom descriptions. The low-score gate must remain enabled.
