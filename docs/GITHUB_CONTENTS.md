# Repository Scope

This private repository contains the MedVoice Assist application and the
research code and results needed to understand its development. It is a source
repository, not a backup of the complete thesis-writing workspace.

## Included

- React/FastAPI application, configuration examples and launch scripts.
- Speech processing, symptom extraction, urgency, clarification and response
  modules, with their tests and rule documentation.
- Preliminary research notebooks and the reports from those experiments.
- Expanded three-model training code, BioBERT continuation code, required
  baseline reports, final comparison reports and run configuration.
- The earlier speech/noise pilot's evaluation code and principal results.
- Dataset-source documentation, without the downloaded datasets.
- Setup and reproducibility documentation, architecture diagrams, report-based
  visualisations and genuine fictional app-demo screenshots.

Distinct reports from different experiment settings remain separate even when
their filenames are similar. Paths used by the training scripts and historical
notebooks have been preserved to avoid changing their behaviour during cleanup.

## Kept Locally but Excluded from Git

- The `chapter4_rewrite_figures/` folder and its figure-generation helper.
- Thesis proposal, chapter drafts, bibliography, LaTeX algorithms, assembly
  guides and other report-writing exports.
- Superseded proposal-formatting and document-generation scripts.
- Old checkpoint metadata, duplicate intermediate training outputs and retired
  tests tied to former directory names.
- Downloaded audio/noise and text datasets, raw recordings, conversation
  outputs, logs, scratch renders, virtual environments and frontend builds.
- Full trained weights and duplicate training checkpoints.
- Participant records, consent forms, credentials and local secrets.

The ignore rules prevent these categories from being accidentally re-added.
Removing a file from the current Git tree does **not** delete its local copy or
erase it from earlier commits. The repository and existing release remain
private. This cleanup is not a history rewrite or a sensitive-data purge.

## Final Trained Model

The [private model release](https://github.com/sean8911/medvoice-assist/releases/tag/initial-commit)
contains `biobert-final.zip` and `biobert-final.sha256`. The archive restores:

```text
biobert/config.json
biobert/model.safetensors
biobert/tokenizer_config.json
biobert/tokenizer.json
biobert/MANIFEST.sha256
```

The four inference files were packaged from the final local 773-class
checkpoint. [model_artifacts.json](model_artifacts.json) records the hashes;
[Setup](SETUP.md#restore-the-final-trained-model) describes checksum verification
and extraction. The release is unchanged by the repository cleanup. Source
archives and ordinary Git clones do not include the model weights, and Git LFS
is not used for the checkpoint.

Keep an independent backup of irreplaceable research data, reports and weights.
