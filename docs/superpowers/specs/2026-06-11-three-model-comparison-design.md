# Three-Model Text Classification Comparison Design

## Objective

Create and execute `Three Model Comparison.ipynb` to compare TF-IDF +
Logistic Regression, fine-tuned BERT, and BioBERT fairly for single-label,
24-class disease classification on `Symptom2Disease.csv`.

## Fair Comparison Protocol

- Remove missing rows and unnamed CSV index columns.
- Encode the 24 disease labels once using `LabelEncoder`.
- Reproduce the existing unstratified 80/20 split using
  `random_state=42`, giving 960 training and 240 held-out test samples.
- Train TF-IDF + Logistic Regression on the 960 training samples.
- Load the existing `bert_symptom_model` and `biobert_symptom_model`.
- Generate fresh predictions from all three models for the same ordered set
  of 240 test samples.
- Validate that each model produces exactly 240 predictions and that each
  prediction is a valid class ID.

## Metrics

Report percentage values for:

- accuracy
- macro precision
- macro recall
- macro F1-score
- weighted precision
- weighted recall
- weighted F1-score

Rank the three models using weighted F1 as the primary comparison metric.
Calculate each model's percentage-point difference from the highest score for
every metric.

## Outputs

- `three_model_comparison.csv`
- `three_model_differences.csv`
- `tfidf_three_model_classification_report.csv`
- `bert_three_model_classification_report.csv`
- `biobert_three_model_classification_report.csv`
- `three_model_comparison.png`

## Interpretation

The notebook must state that the best result applies only to this dataset,
split, and model configurations. It may explain that transformer models often
require larger and more diverse datasets, additional hyperparameter tuning,
or domain-matched data to realize their full potential. It must not promise
that their accuracy will necessarily become higher.

## Verification

- Validate notebook JSON and compile all code cells.
- Confirm 1,200 rows, 24 classes, and the 960/240 split.
- Confirm fresh predictions reproduce the existing recorded metrics.
- Confirm exported metrics are between 0 and 100.
- Visually inspect the generated comparison chart.
