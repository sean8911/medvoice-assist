# Thesis 2 Three-Model Server Training

This folder trains and compares the three Thesis 2 disease diagnosis models:

1. TF-IDF + Logistic Regression
2. Fine-tuned BERT
3. BioBERT

The code uses Python scripts only. There are no notebook files in this package.

## Folder Layout

```text
thesis2_three_model_server/
  train.sh
  train_three_models.py
  requirements.txt
  data/
    disease-diagnosis-dataset/
  logs/
  outputs/
```

The dataset should be located at:

```text
data/disease-diagnosis-dataset/
```

The default files are:

```text
disease_diagnosis_train_cleaned.csv
disease_diagnosis_test_cleaned.csv
```

## Run on the School Server

Create and activate the conda environment before submitting the job:

```bash
cd thesis2_three_model_server
conda create -n thesis2 python=3.10 -y
conda activate thesis2
pip install --upgrade pip
pip install -r requirements.txt
```

Then submit the SLURM job:

```bash
sbatch train.sh
```

The `train.sh` file intentionally follows the simple school-server format:
SBATCH lines first, then one Python command. Do not add `conda`, `pip`,
`mkdir`, or `module load` commands inside `train.sh` if your server blocks
them.

If the server needs GPU PyTorch installed separately, run this before
`pip install -r requirements.txt`:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick Smoke Test

Before running the full dataset, you can test the code with fewer rows:

```bash
python train_three_models.py \
  --dataset-dir data/disease-diagnosis-dataset \
  --output-dir outputs_smoke \
  --models tfidf \
  --max-train-samples 2000 \
  --max-test-samples 500
```

## Main Outputs

After training, check the `outputs/` folder:

```text
three_model_comparison.csv
three_model_differences.csv
three_model_comparison.png
tfidf_logistic_regression_classification_report.csv
fine_tuned_bert_classification_report.csv
biobert_classification_report.csv
model_predictions.csv
label_mapping.csv
models/
```

The comparison table includes accuracy, macro precision, macro recall, macro
F1, weighted precision, weighted recall, and weighted F1. Models are ranked by
weighted F1 score.

For a complete three-model thesis comparison, `three_model_comparison.csv`
must contain these three rows:

```text
TF-IDF + Logistic Regression
Fine-tuned BERT
BioBERT
```

If the server log shows TF-IDF and fine-tuned BERT but not BioBERT, the job
stopped before the third model. Check `logs/error_<jobid>.txt` for a time,
memory, CUDA, or download error, then rerun after fixing that server issue.
