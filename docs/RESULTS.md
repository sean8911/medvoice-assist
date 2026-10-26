# Results and provenance

All numbers below describe retained experiment outputs, not newly trained or
independently re-evaluated models. Percentages are rounded for display.

## Expanded dataset comparison

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 87.1027 | 87.9733 | 87.1027 | 87.1032 | 74.7431 |
| Fine-tuned BERT | 87.2565 | 88.2637 | 87.2565 | 87.3104 | 73.4321 |
| BioBERT | 87.2727 | 88.2843 | 87.2727 | 87.3218 | 73.0820 |

The generated [summary CSV](../3%20Models%20After%20Training/outputs/three_model_comparison.csv)
collates the final per-model reports. Original evidence:

- [TF-IDF report](../3%20Models%20After%20Training/outputs/tfidf_logistic_regression_classification_report.csv)
- [BERT report](../3%20Models%20After%20Training/outputs/fine_tuned_bert_classification_report.csv)
- [BioBERT report](../3%20Models%20After%20Training/outputs/biobert_classification_report.csv)
- [Retained BioBERT run configuration](../3%20Models%20After%20Training/outputs/run_config.json)

Accuracy and weighted F1 favour BioBERT very narrowly. Macro F1 favours TF-IDF.
The 0.0162-percentage-point BioBERT/BERT accuracy gap is four additional correct
predictions out of 24,695 examples, not a demonstrated significant advantage.
These observations do not establish which approach is best for all medical
chatbots or for patients outside the dataset.

The reported configuration records three BioBERT epochs, batch size 16,
evaluation batch size 32, maximum token length 128, learning rate 2e-5 and seed
42, with automatic checkpoint resumption. The convenience BioBERT launcher in
the repository defaults to one epoch; that is not the same configuration as the
retained final report. Training times involving resumption must not be compared
as if all models were trained from scratch in identical conditions.

## Preliminary 24-class comparison

| Model | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 96.67 | 96.54 | 96.70 |
| Fine-tuned BERT | 87.92 | 87.25 | 86.94 |
| BioBERT | 88.75 | 88.01 | 87.57 |

Source: [the root preliminary comparison CSV](../three_model_comparison.csv).
Use that table for Symptom2Disease only. The seed-42 outer partition contains
960 training and 240 test descriptions. Historical BERT monitoring used the
outer test set, whereas preliminary BioBERT used an internal validation stage
for epoch selection. This is a qualification of the comparison, not a claim of
identical validation design.

## Dataset overlap qualification

The expanded cleaned files contain 469,195 training rows and 24,695 test rows,
with 773 labels overall. Only 697 labels have test support. Reports generated
against the complete label list therefore include zero-support classes in macro
summaries; other averaging conventions can produce different values.

A previous local exact-match audit found that 23,863 test rows (96.63%) also
have a stripped `(text_cleaned, diagnosis)` pair in the training file. These
counts concern repeated text/label pairs, not verified independent patient
identities. Even a non-matching pair is not automatically unseen text.

No duplicate-grouped replacement split or retraining has been performed. The
current scores are retained supplied-split outcomes and must not be presented
as independent generalisation estimates. Corrected future evaluation would
group normalised text before partitioning, resolve conflicting labels explicitly
and train fresh models on the revised split.

## Legacy speech pilot

The [pilot report](../voice_biobert_pipeline/evaluation/outputs/BENCHMARK_RESULTS.md)
uses six recordings across two medical-speech datasets, five noise conditions
and raw/enhanced routes (60 output rows). Improvements from preprocessing are
mixed, and direct narrative-to-class classification failed its small mapped
case set. The pilot motivated later symptom extraction.

That experiment predates the latest extraction, urgency, clarification and
response orchestration. It is not a completed benchmark of the current chatbot.
Reference-prediction agreement measures stability, not classification accuracy.
Participant usability and clinically reviewed safety evaluation are also
separate from the retained classifier scores.

## Figure regeneration

With the training/analysis dependencies installed, run from the repository root:

```powershell
python scripts/build_readme_assets.py
```

The script reads the retained CSV reports, writes the derived expanded summary
and regenerates `docs/assets/classifier-comparison.png`. It does not perform
inference, alter model scores or train any model.
