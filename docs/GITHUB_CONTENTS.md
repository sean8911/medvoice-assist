# Repository scope

`sean8911/medvoice-assist` remains private. Repository cleanup changes the current
tracked file set; it does not erase earlier Git history or delete the original
files from the development laptop.

## Retained

- The working React/FastAPI app, pipeline modules and their current tests.
- Both dataset stages: preliminary notebooks and canonical comparison reports,
  expanded training source, final reports and run metadata.
- The BioBERT continuation script and its required baseline input reports.
- Legacy speech-pilot source, its principal result files and dataset references.
- Thesis proposal, methodology/implementation chapters, algorithms and diagrams.
- Notebook/figure utilities relevant to the retained experiments.
- Setup, results and experiment documentation, plus genuine report-based graphs.

## Removed from the current tracked view

- Retired 24-class checkpoint metadata without the corresponding weights.
- Duplicated intermediate training outputs and copied dataset metadata.
- Superseded two-model difference/comparison files and exact duplicate reports.
- Old authoring plans and proposal-formatting/export utilities.
- Package tests hard-coded to former folder names no longer present.
- Smoke-test inventories/plots duplicated beside the principal speech pilot.
- The unrelated YOLO launcher mistakenly present in the three-model folder.

Files in those categories remain locally and are now ignored to avoid accidental
re-addition. Distinct reports that document separate experiment settings are not
removed simply because their filenames look similar.

## Trained model release

The existing private `initial-commit` release remains unchanged. Its assets are
`biobert-final.zip` and `biobert-final.sha256`; the archive contains:

```text
biobert/config.json
biobert/model.safetensors
biobert/tokenizer_config.json
biobert/tokenizer.json
biobert/MANIFEST.sha256
```

The four inference files match the original final 773-class checkpoint. Hashes
are recorded in [model_artifacts.json](model_artifacts.json). Restore as described
in [Setup](SETUP.md); the automatic source archive does not include these weights.
No Git LFS storage is used for the model.

## Excluded local data

Downloaded speech/noise data, raw/prepared training datasets, large weights,
duplicate checkpoints, virtual environments, `node_modules`, generated `dist`,
scratch renders, raw recordings, conversation outputs and logs remain excluded.
Participant/consent records and credentials must not be committed, even while
the repository is private.

This is a curated thesis source repository, not a complete laptop backup or a
publicly deployed medical service. A separate backup remains necessary for
irreplaceable data and model checkpoints.
