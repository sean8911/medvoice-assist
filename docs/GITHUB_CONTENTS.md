# GitHub snapshot contents

Repository: `sean8911/medvoice-assist` (private).
Initial commit message: `Initial commit`.

## Included in Git

- Application and pipeline source, tests, requirements and setup scripts.
- Training/comparison notebooks and scripts.
- Retained comparison reports, plots, run metadata and lightweight model metadata.
- Thesis proposal, chapter drafts, diagrams and LaTeX algorithms in this workspace.
- Dataset source README and summary where explicitly retained by `.gitignore`.

Existing code and report limitations are preserved. The upload does not retrain
models, alter classification thresholds, conduct human evaluation or deploy a
public medical service.

## Supplied as a private release attachment

The `initial-commit` release supplies `biobert-final.zip`, containing:

```text
biobert/config.json
biobert/model.safetensors
biobert/tokenizer_config.json
biobert/tokenizer.json
biobert/MANIFEST.sha256
```

These inference files are copied unchanged from
`3 Models After Training/outputs/models/biobert/`. The pickle-based
`training_args.bin` is unnecessary for inference and is not included. Hashes and
the archive checksum are recorded in `docs/model_artifacts.json`.

GitHub Release storage is used instead of Git LFS. Download the attachment
separately after cloning; GitHub's automatic source ZIP does not contain it.
The release inherits the private repository's access restrictions.

## Kept only on the development laptop

- Downloaded medical speech and environmental-noise datasets.
- Raw and prepared training datasets, including the root Symptom2Disease CSV.
- Earlier BERT/BioBERT weights, TF-IDF joblib files and duplicate training checkpoints.
- Python/JavaScript caches, environments and `node_modules`.
- Generated frontend `dist`, which setup rebuilds.
- Scratch files under `tmp`, server logs and local voice-pipeline outputs.
- Raw audio recordings, credentials and any private participant/consent records.

This is not a complete 36 GB laptop backup. All excluded files remain in place.
Keep a separate appropriate backup of irreplaceable data and weights. Review
source and data licensing before making this repository public. Do not add a
software/data licence on the assumption that it covers third-party datasets or
pretrained weights.

## Future commits

From the repository root, review the change list before staging:

```powershell
git status
git add <specific-files-you-reviewed>
git commit -m "Describe the change"
git push
```

Keep `.env` secrets, raw participant responses, identifiable recordings and
consent records out of Git, even while the repository remains private.
