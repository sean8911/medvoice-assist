# Chapter 4 Three-Model Rewrite Design

## Objective

Create a revised copy of the latest thesis proposal and replace Chapter 4
with a concise preliminary-findings chapter that reflects the verified audio
preprocessing experiment and the comparison of TF-IDF + Logistic Regression,
fine-tuned BERT, and BioBERT.

## Structure

- 4.0 Preliminary Results
- 4.1 Audio Preprocessing Evaluation
- 4.1.1 Real Recording Comparison
- 4.1.2 Synthetic-Noise Evaluation
- 4.2 Three-Model Text Classification Comparison
- 4.3 Summary of Preliminary Findings

## Verified Results

### Audio preprocessing

- Real recording baseline: WER 11.11%, CER 9.52%.
- Real recording improved: WER 0.00%, CER 0.00%.
- Synthetic-noise mean baseline: WER 8.33%, CER 8.93%.
- Synthetic-noise mean improved: WER 5.56%, CER 4.76%.

### Text classification

- TF-IDF + Logistic Regression: accuracy 96.67%, weighted F1 96.54%.
- BioBERT: accuracy 88.75%, weighted F1 88.01%.
- Fine-tuned BERT: accuracy 87.92%, weighted F1 87.25%.

## Interpretation

State that TF-IDF + Logistic Regression achieved the highest performance for
the selected dataset and split. Do not claim universal superiority or make it
the only text-classification model considered by the research. Explain that
transformer models may benefit from larger, more diverse datasets,
domain-matched text, and additional hyperparameter tuning, but that this must
be evaluated rather than assumed.

## Document Treatment

- Preserve the original proposal and edit a separate copy.
- Preserve the surrounding university formatting and existing Chapter 4
  heading styles.
- Replace raw result screenshots with proper Word tables.
- Use the existing audio figures and the verified three-model comparison
  chart.
- Correct all Chapter 4 subsection and figure numbering.
- Render the revised document and visually inspect all affected pages.
