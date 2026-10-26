# Thesis 2 BioBERT-Only Server Run

This folder is for finishing the Thesis 2 three-model comparison after TF-IDF
and fine-tuned BERT already completed.

It trains only:

```text
BioBERT
```

Then it combines the new BioBERT report with the existing baseline reports:

```text
baseline_outputs/tfidf_logistic_regression_classification_report.csv
baseline_outputs/fine_tuned_bert_classification_report.csv
```

## Why This Folder Exists

The previous full run finished TF-IDF and fine-tuned BERT, then failed while
loading BioBERT because the server environment had Torch below `2.6`.
BioBERT's upstream checkpoint loads from PyTorch `.bin` weights, and current
Transformers blocks that load path unless Torch is at least `2.6`.

If `outputs/checkpoints/biobert/checkpoint-*` already exists, this script can
resume from that saved checkpoint with Torch `2.5.1` because it no longer needs
to load the original upstream `.bin` weights.

## Conda Setup

Run these commands outside `train.sh`:

```bash
conda create -n thesis2_biobert python=3.10 -y
conda activate thesis2_biobert
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

If your server CUDA version needs a different PyTorch wheel, use the matching
command from the PyTorch website. Torch `2.6` or newer is required only when
starting from the original BioBERT model with no saved checkpoint.

Check the installed Torch version:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If you are resuming from an existing BioBERT checkpoint, Torch `2.5.1+cu124`
is acceptable. If you are starting from scratch, it must show Torch `2.6` or
newer.

## Run

```bash
cd BioBERT
sbatch train.sh
```

The `train.sh` file intentionally follows the simple school-server format:
SBATCH lines first, then one Python command.

The default BioBERT run is now set to **1 epoch** because a 3-epoch run is too
long for a 30-hour server limit. During training, the script saves checkpoints
every 1000 optimizer steps in:

```text
outputs/checkpoints/biobert/
```

This convenience default is not the retained final run configuration, which
records three epochs with automatic checkpoint resumption. The final reports
are under `3 Models After Training/outputs/`; see
[experiment provenance](../docs/EXPERIMENTS.md). Do not describe a new one-epoch
run as reproduction of the three-epoch artifact.

If the server stops the job before it finishes, submit the same script again:

```bash
sbatch train.sh
```

The script uses `--resume-from-checkpoint auto`, so it continues from the latest
checkpoint instead of starting over.

## Outputs

After the job finishes, check:

```text
outputs/biobert_classification_report.csv
outputs/three_model_comparison.csv
outputs/three_model_differences.csv
outputs/three_model_comparison.png
outputs/models/biobert/
```

The final comparison table includes:

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted Precision
Weighted Recall
Weighted F1
```
