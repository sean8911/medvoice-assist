# BioBERT Classifier Comparison Design

## Objective

Create `BioBERT.ipynb` to fine-tune BioBERT for single-label, 24-class
disease classification and compare it fairly with TF-IDF + Logistic
Regression on `Symptom2Disease.csv`.

## Experimental Protocol

- Read the `label` and `text` columns and remove missing rows and unnamed
  index columns.
- Encode labels once with `LabelEncoder`.
- Reproduce the existing unstratified 80/20 train/test split using
  `random_state=42`, producing 960 training and 240 test samples.
- Use the same held-out test rows for BioBERT and TF-IDF + Logistic
  Regression.
- Fit all preprocessing and models using only the training split.
- Do not select hyperparameters or models using the held-out test results.

## Models

### TF-IDF + Logistic Regression

Reproduce the existing baseline:

- lowercase text
- English stop-word removal
- word unigrams and bigrams
- maximum 20,000 TF-IDF features
- sublinear term frequency
- Logistic Regression with `max_iter=2000` and seed 42

### BioBERT

- Checkpoint: `dmis-lab/biobert-base-cased-v1.2`
- Architecture: `AutoModelForSequenceClassification`
- Maximum sequence length: 128 tokens
- Training epochs: 3
- Learning rate: 2e-5
- Weight decay: 0.01
- Batch size: 8 on GPU; reduce automatically when necessary
- Reproducibility seed: 42
- Keep the best epoch according to validation macro F1

To avoid using the held-out test set for model selection, split a validation
subset from the 960 training rows. The final comparison is performed once on
the unchanged 240-row test set.

## Evaluation

Report the following percentages for both models:

- accuracy
- macro precision
- macro recall
- macro F1-score
- weighted precision
- weighted recall
- weighted F1-score

Also report training time, generate a grouped comparison chart, calculate
BioBERT's percentage-point difference from TF-IDF + Logistic Regression, and
identify the better model for this dataset using weighted F1-score as the
primary selection metric.

The conclusion must be limited to this dataset and split. The notebook must
not claim that either model is universally better for medical chatbots.

## Outputs

The notebook will save:

- `biobert_model_comparison.csv`
- `biobert_vs_tfidf_difference.csv`
- `biobert_classification_report.csv`
- `biobert_confusion_matrix.png`
- `biobert_vs_tfidf_comparison.png`
- `biobert_symptom_model/`

## Runtime and Error Handling

- Print whether CUDA or CPU is being used.
- Show clear installation guidance when required packages are missing.
- Show a clear message if the BioBERT checkpoint cannot be downloaded.
- Allow training to run on CPU, while warning that it may take substantially
  longer.
- Use `zero_division=0` for classification metrics.

## Verification

- Validate that the dataset contains exactly 24 classes.
- Confirm the split sizes are 960 training and 240 test rows before creating
  the internal validation subset.
- Confirm that both models predict the same 240 held-out samples.
- Validate that all exported metric values are within 0 to 100.
- Execute all non-training setup and structural checks locally.
- If full BioBERT training cannot be completed in the available environment,
  leave the notebook ready for the user to run and state that the winner is
  not known until execution completes.
