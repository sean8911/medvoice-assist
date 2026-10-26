# Three-Model Text Classification Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute one notebook that compares TF-IDF + Logistic Regression, fine-tuned BERT, and BioBERT using fresh predictions on the same held-out test samples.

**Architecture:** A notebook builder creates the reproducible experiment, while a validator checks its required structure and safeguards. The notebook recreates the seed-42 dataset split, fits TF-IDF, loads both saved transformer models, calculates identical metrics, exports thesis-ready artifacts, and limits conclusions to the observed experiment.

**Tech Stack:** Jupyter Notebook, Python, pandas, NumPy, scikit-learn, PyTorch, Hugging Face Transformers, matplotlib, seaborn

---

### Task 1: Add a failing notebook validator

**Files:**
- Create: `scripts/validate_three_model_comparison.py`
- Test: `Three Model Comparison.ipynb`

- [x] Validate notebook existence and valid JSON.
- [x] Require the three model names, seed-42 80/20 split, seven metrics, prediction-length checks, six exports, ranking, and dataset-specific interpretation.
- [x] Reject `LinearRegression`, copied-only metric tables, or missing fresh transformer prediction functions.
- [x] Run the validator and confirm it fails because the notebook does not exist.

### Task 2: Build the comparison notebook

**Files:**
- Create: `scripts/build_three_model_comparison.py`
- Create: `Three Model Comparison.ipynb`

- [x] Add Markdown explaining the comparison scope and terminology.
- [x] Load and validate 1,200 rows and 24 disease classes.
- [x] Reproduce the 960/240 split using `random_state=42`.
- [x] Train TF-IDF + Logistic Regression with the existing configuration.
- [x] Load `bert_symptom_model` and `biobert_symptom_model`.
- [x] Validate each transformer model's 24-class output and label mapping.
- [x] Generate fresh batched predictions for both transformer models.
- [x] Calculate the seven metrics for all three models.
- [x] Rank models by weighted F1 and calculate percentage-point gaps from the best score.
- [x] Export the comparison tables, three class-level reports, and grouped bar chart.
- [x] Print balanced thesis-ready interpretation.

### Task 3: Execute and verify the experiment

**Files:**
- Verify: `Three Model Comparison.ipynb`
- Verify: `three_model_comparison.csv`
- Verify: `three_model_differences.csv`
- Verify: `three_model_comparison.png`

- [x] Compile every notebook code cell.
- [x] Execute the notebook using an environment that can load all three models.
- [x] Confirm zero notebook error outputs.
- [x] Confirm all models predict the same 240 test samples.
- [x] Confirm TF-IDF and BERT reproduce their existing metrics, and document any investigated CPU/GPU numerical difference for BioBERT.
- [x] Confirm all exported metric values are between 0 and 100.
- [x] Visually inspect the comparison chart.
