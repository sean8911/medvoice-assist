# BioBERT Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a runnable `BioBERT.ipynb` that fairly compares fine-tuned BioBERT with TF-IDF + Logistic Regression on the same held-out Symptom2Disease test set.

**Architecture:** The notebook will load and validate the dataset, reproduce the existing outer train/test split, create an internal validation split for BioBERT model selection, train both classifiers, evaluate both exactly once on the unchanged test set, and export tables, reports, figures, and the BioBERT model. A separate validator will inspect notebook structure and execute lightweight setup cells without downloading or training BioBERT.

**Tech Stack:** Jupyter Notebook, Python, pandas, NumPy, scikit-learn, PyTorch, Hugging Face Transformers, matplotlib, seaborn

---

### Task 1: Add notebook structural validation

**Files:**
- Create: `scripts/validate_biobert_notebook.py`
- Test: `BioBERT.ipynb`

- [x] Write a validator that fails when `BioBERT.ipynb` is absent.
- [x] Validate that the notebook contains the BioBERT checkpoint, seed 42, the exact 80/20 outer split, an internal validation split, all seven required metrics, and every specified output path.
- [x] Validate that BioBERT selects its best checkpoint using validation macro F1 rather than test metrics.
- [x] Run the validator and confirm it fails because the notebook does not exist.

### Task 2: Create the BioBERT comparison notebook

**Files:**
- Create: `BioBERT.ipynb`

- [x] Add explanatory Markdown describing the fair comparison and correcting “Linear Regression” to “Logistic Regression.”
- [x] Add dependency checks and reproducibility configuration.
- [x] Load `Symptom2Disease.csv`, encode 24 labels, and reproduce the existing 960/240 outer split.
- [x] Split the 960 training rows into BioBERT training and validation subsets without changing the 240-row test set.
- [x] Train and time the existing TF-IDF + Logistic Regression configuration.
- [x] Tokenize text with `dmis-lab/biobert-base-cased-v1.2` and create PyTorch datasets.
- [x] Fine-tune `AutoModelForSequenceClassification` for three epochs and select the best validation macro F1 checkpoint.
- [x] Evaluate both models on the same 240 test rows using accuracy, macro/weighted precision, recall, and F1.
- [x] Export comparison CSVs, BioBERT classification report, confusion matrix, comparison chart, and the saved BioBERT model.
- [x] Print a dataset-specific winner statement based on weighted F1.

### Task 3: Verify structure and lightweight execution

**Files:**
- Verify: `BioBERT.ipynb`
- Verify: `scripts/validate_biobert_notebook.py`

- [x] Run the structural validator and require all checks to pass.
- [x] Parse the notebook as valid JSON and compile every code cell.
- [x] Execute the lightweight dataset/split logic independently and confirm 1,200 rows, 24 classes, 960 outer training rows, and 240 held-out test rows.
- [x] Confirm the notebook contains no saved fake BioBERT results and clearly states that the winner is known only after training completes.
- [x] Record that full BioBERT training requires downloading the pretrained checkpoint and may require a GPU.
