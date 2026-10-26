# Chapter 4 Screenshot and Assembly Guide

Workspace: `C:/Year 3 Sem 3/Thesis/Denoise + Whisper`.

The Word file is a complete prose draft with explicit insertion slots. It does not claim screenshots, algorithms or participant results were already inserted.
The standalone `Chapter_4_Algorithms.tex` contains Algorithms 4.1-4.12. Compile it, export the required algorithm pages, and place each algorithm in its matching Word slot. Keep its caption once, not once in the crop and again as a duplicate caption.

## Code Captures

Use the current source, not a rewritten demo. Show the filename and useful line numbers. Crop one logical block, use a light editor theme and readable font, and exclude terminal logs, credentials and personal identifiers. Where two blocks are listed, combine two labelled panels instead of shrinking a whole script.
Listings 4.1-4.3 already contain selected actual model code excerpts. A screenshot is alternative visual evidence of the same source; avoid retaining both representations when they only repeat identical content in the final manuscript.

### Figure 4.3 Validation of prepared input fields and consistent label encoding
Place in Section 4.3.2. Core capture.
- [train_three_models.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/TF-IDF and Fine Tune BERT/train_three_models.py:175>) lines 175-196.
- [train_three_models.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/TF-IDF and Fine Tune BERT/train_three_models.py:244>) lines 244-258.
Purpose: Show schema checks and the common target mapping, not a printout of dataset rows.

### Figure 4.4 Fitted TF-IDF pipeline and test-set prediction
Place in Section 4.4.1. Core capture.
- [train_three_models.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/TF-IDF and Fine Tune BERT/train_three_models.py:283>) lines 283-315.
Purpose: Capture the vectoriser and Logistic Regression settings together with fit, predict and model saving.

### Figure 4.5 Transformer head construction and supervised training with artifact export
Place in Section 4.4.3. Core capture.
- [train_three_models.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/TF-IDF and Fine Tune BERT/train_three_models.py:439>) lines 439-448.
- [train_three_models.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/TF-IDF and Fine Tune BERT/train_three_models.py:464>) lines 464-478.
Purpose: Use two labelled panels: head/mapping configuration and Trainer training/prediction/saving.

### Figure 4.6 DC removal, boundary trimming, stationary denoising and peak normalisation
Place in Section 4.5. Core capture.
- [voice_to_disease.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/voice_biobert_pipeline/voice_to_disease.py:287>) lines 287-311.
Purpose: Show the four implemented operations, trim fallback and guarded normalisation in one readable crop.

### Figure 4.7 Chunk-level acoustic processing and Whisper token generation
Place in Section 4.6.1. Core capture.
- [model_runtime.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/medical_chatbot_app/backend/app/services/model_runtime.py:206>) lines 206-233.
Purpose: Capture processor input, language/task, inference-only generation and decoding; do not screenshot installation commands.

### Figure 4.8 Mention evidence, latest assertion consolidation and canonical classifier input
Place in Section 4.7.1. Core capture.
- [symptom_extractor.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/voice_biobert_pipeline/symptom_extractor.py:456>) lines 456-494.
Purpose: Show how evidence is stored and how present, negated and other statuses affect the model input; split into two crops if needed.

### Figure 4.9 Highest-priority urgency selection and classification-bypass decision
Place in Section 4.7.2. Core capture.
- [urgency_detector.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/voice_biobert_pipeline/urgency_detector.py:603>) lines 603-632.
Purpose: Show the maximum-priority decision, bypass flag and returned evidence rather than the full long list of rules.

### Figure 4.10 Loading the retained local BioBERT checkpoint and applying display thresholds
Place in Section 4.7.3. Core capture.
- [model_runtime.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/medical_chatbot_app/backend/app/services/model_runtime.py:85>) lines 85-100.
- [model_runtime.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/medical_chatbot_app/backend/app/services/model_runtime.py:137>) lines 137-166.
Purpose: Use two panels: local_files_only checkpoint loading and the score/margin/display gate. Include the default model path from config.py in a small supplemental crop.

### Figure 4.11 Urgency and clarification branches around disease-model inference
Place in Section 4.8.1. Core capture.
- [pipeline_service.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/medical_chatbot_app/backend/app/services/pipeline_service.py:177>) lines 177-198.
- [pipeline_service.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/medical_chatbot_app/backend/app/services/pipeline_service.py:199>) lines 199-226.
Purpose: Show bypass versus inference and post-inference withholding in two readable panels.

### Figure 4.12 Combining the follow-up with original text and closing unresolved clarification
Place in Section 4.8.2. Core capture.
- [pipeline_service.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/medical_chatbot_app/backend/app/services/pipeline_service.py:288>) lines 288-321.
Purpose: Capture combined text, reanalysis, pre_limit_status, hidden result and closed questions; use two panels if necessary.

### Figure 4.13 Response priority and explicit permission to display a dataset class
Place in Section 4.9. Core capture.
- [rule_based_response.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/voice_biobert_pipeline/rule_based_response.py:205>) lines 205-239.
- [rule_based_response.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/voice_biobert_pipeline/rule_based_response.py:240>) lines 240-258.
Purpose: Show the branch ordering and display_prediction check, not all static medical wording.

### Figure 4.18 Fixed-segment noise scaling at a requested SNR
Place in Section 4.12.2. Optional experiment capture.
- [evaluate_audio_datasets.py](<C:/Year 3 Sem 3/Thesis/Denoise + Whisper/voice_biobert_pipeline/evaluation/evaluate_audio_datasets.py:283>) lines 283-308.
Purpose: Capture seed-based segment selection, power calculations, scaling and peak handling; this is optional implementation evidence for the experiment.

## Notebook Evidence for the First Dataset

Cell numbers below are one-based positions including Markdown cells, not execution counters. If notebook contents change, use the quoted source anchor.
- `Three Model Comparison.ipynb`: cell 5 reproduces the 960/240 split; cell 7 defines and fits TF-IDF; cell 9 checks saved transformer labels. These are evidence for the 24-class comparison, not the expanded training.
- `BioBERT.ipynb`: cell 10 constructs the outer and internal validation split; cell 18 selects the validation epoch; cell 20 trains a fresh BioBERT on all 960 records. Stored output selects three epochs.
- `Fine Tune Bert.ipynb`: cells 21 and 23 configure training and pass `test_ds` as `eval_dataset`. Do not describe that test-monitoring procedure as a separate internal validation set.

## Interface Captures

- Figure 4.14: record or upload a fictional symptom audio and capture the editable Whisper transcript BEFORE pressing Confirm and analyse. Include confirm/cancel and the audio-enhancement option where practical.
- Figure 4.15: capture one initial clarification and the subsequent unresolved fallback in two panels. A one-symptom cough followed by a truthful no-other-symptoms answer is suitable for demonstrating retained context; a disease prediction is not required.
- Figure 4.16: open Research details after a scenario. Include recognised evidence, canonical input, status, score/margin and timings. Label model outputs as research scores. This panel may contain withheld rankings; it is not a clinical differential diagnosis.
- Do not change a model output or manufacture a percentage for a screenshot. State that a shown scenario is illustrative, not representative accuracy evidence.

## Algorithm to Section Map

| Algorithm | Section | Routine |
|---|---|---|
| 4.1 | 4.3 | Prepared data loading and retained partitions |
| 4.2 | 4.4.1 | TF-IDF + Logistic Regression |
| 4.3 | 4.4.3 | Expanded transformer training |
| 4.4 | 4.5 | Audio enhancement |
| 4.5 | 4.6.1 | Whisper chunk transcription |
| 4.6 | 4.7.1 | Symptom extraction and assertion states |
| 4.7 | 4.7.2 | Urgency priority |
| 4.8 | 4.7.3 | Trained BioBERT and score gates |
| 4.9 | 4.8.1 | Pipeline orchestration |
| 4.10 | 4.8.2 | Follow-up context and round limit |
| 4.11 | 4.9 | Rule-based responses |
| 4.12 | 4.12.1 | Legacy paired speech experiment |

## Complete Before Final Submission

- Insert the selected algorithms and genuine code/UI captures; remove the bracketed insertion notes.
- Diagrams 4.1, 4.2 and 4.17 are already inserted. Their PNG originals are in `figures/`.
- Record exact training and inference environments from their actual machines. Manifest version ranges are not installed version records.
- Keep the 24-class and expanded artifacts separate. Do not use the top-level preliminary comparison CSV as the expanded result table.
- Keep the supplied-split overlap limitation. No corrected split or retraining happened as part of this document request.
- The legacy speech pilot is not the completed benchmark of the current extractor/urgency/clarification pipeline.
- Conduct the consented five-person study, fill only genuine demographic/profile entries and retain negative feedback. P01-P05 currently reserve identifiers; they are not recruited participants.
- Obtain the original SUS form from the cited source; do not rewrite its ten items and still call the custom scale SUS.
- The UI JSON export contains the analysis object only. Capture audio/transcription metadata and timing separately for a full speech record.
- Chapter 5 should contain actual collected metrics and comments. Do not invent missing training times, clinician review or participant scores.