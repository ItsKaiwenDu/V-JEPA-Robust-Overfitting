# Reproducibility and artifacts

## Before a confirmatory run

Record the project Git commit and any uncommitted diff, the upstream Meta
commit, checkpoint SHA-256, resolved configuration and its hash, data
manifest hash, preprocessing version, all random seeds, software versions,
and GPU model/VRAM. Record the fixed-target protocol explicitly.

Save the exact environment after the CUDA pilot. `requirements.txt` currently
describes only scaffold dependencies and is not an experiment lockfile.

## Directory contract

Use the same run name across these locations:

```text
checkpoints/<condition>/[diagnostic/]<run-name>/
runs/<condition>/[diagnostic/]<run-name>/
report/<condition>/[diagnostic/]<run-name>/
```

Examples are `adversarial-finetune/seed-0/` and
`clean-finetune/diagnostic/seed-0/`. Use a new name for every rerun or protocol
change. Diagnostics never overwrite main-study runs. Keep the frozen
reference at `pretrained/reference/`.

The future runner will save these artifacts:

| Artifact | Location | Contents |
| --- | --- | --- |
| `config.resolved.yaml` | Run directory | Actual settings used, including resolved local runtime choices |
| `metadata.json` | Run directory | Provenance, seeds, device, environment, and start/end status |
| `metrics.jsonl` | Run directory | Epoch/step, split, loss, attack settings, elapsed time |
| Epoch weights and resume state | Checkpoint directory | Model/optimizer/scheduler/scaler/RNG state as appropriate |
| `evaluation_results.csv` | Curated report run | Checkpoint, split, attack ID, metrics, sample counts |
| Per-video evaluation records | Run directory | Video/group ID, checkpoint, attack ID, measured outcomes |
| Figures and summary | Curated report run / `overall/` | Verified per-seed and aggregate results |

These filenames document planned interfaces; writers are not implemented.
An attack ID must resolve to norm, epsilon, step size, iterations, restarts,
objective, mask generation, precision, and candidate-selection rules. Do
not mix results with different attack definitions under one CSV label.

## Split and selection discipline

Track source-group-separated manifests with repository-relative or
dataset-relative paths. Keep machine-specific path maps local. Reuse the
same manifests across training seeds. Detect duplicate files and group
overlap before training.

Record the selected epoch from validation and the selection rule before
final test evaluation. Fit probes using training labels and freeze probe
choices through validation. Separate monitoring, attack calibration,
checkpoint selection, and final test results in records and figure labels.

## Reporting

Show every training seed. Report losses in their defined units, accuracies
as fractions or explicitly labeled percentages, and accuracy gaps in
percentage points. A positive native-loss gap is final minus selected-best.
Bootstrap by source-video group, not by independently sampled clips from
the same source.

Record completed, interrupted, failed, and excluded runs with reasons. Do
not replace missing measurements with zeros or omit unsuccessful runs from
the progress log. A low prediction loss does not by itself establish useful
representations; inspect downstream behavior and collapse indicators.

## Git and storage

Commit source, configs, relative-path manifests, notes, and small curated
reports. Keep videos, cached features, model weights, raw logs, and credentials
out of Git. Preserve external artifact locations and hashes in the report
when those artifacts are eventually released.

Export needed artifacts before terminating a rental instance. Count file
transfer, checkpoint storage, and idle time in the budget. Neither result
hosting nor a public artifact release has been configured yet.

## Implemented diagnostic records (October 20)

The table above remains a proposed report-export interface. Actual bounded
workflows now write `workflow.json`, checksum-bound per-clip `evaluation.json`,
`calibration.json`, checkpoint `manifest.json`, and feature/probe files plus
`summary.json`. They preserve source/data/runtime identity. Explicit in-place
recovery of the new workflow is distinct from a rerun; current receipt paths
require the same absolute artifact locations. See
[workflow diagnostics](workflow_diagnostics.md) for exact behavior and limits.
