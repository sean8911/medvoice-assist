# Thesis 2 Three-Model Server Training

This folder trains and compares the three Thesis 2 disease diagnosis models:

1. TF-IDF + Logistic Regression
2. Fine-tuned BERT
3. BioBERT

The code uses Python scripts only. There are no notebook files in this package.

## Folder Layout

```text
TF-IDF and Fine Tune BERT/
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
cd "TF-IDF and Fine Tune BERT"
conda create -n thesis2 python=3.10 -y
conda activate thesis2
pip install --upgrade pip
pip install -r requirements.txt
```

Within a permitted school-server job allocation, run the Python entry point:

```bash
python train_three_models.py --dataset-dir data/disease-diagnosis-dataset --output-dir outputs_new --models tfidf bert biobert --epochs 3 --batch-size 16 --max-length 128
```

The unrelated YOLO `train.sh` formerly present in this folder is excluded from
the curated repository. Do not submit it for this experiment. A SLURM batch
launcher must invoke the Python command above and request resources permitted
by the school. Set up the environment before submitting the job.

Install the PyTorch build appropriate to the server's CUDA/driver environment
before installing the remaining requirements. Do not assume a wheel chosen for
another machine will work. Starting BioBERT from upstream `.bin` weights also
requires a sufficiently recent Torch version for the installed Transformers.

The current final reports are retained separately in
`3 Models After Training/outputs/`. New runs should use a new output directory.
See [experiment provenance](../docs/EXPERIMENTS.md) and
[evaluation limitations](../docs/RESULTS.md) before interpreting results.

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
