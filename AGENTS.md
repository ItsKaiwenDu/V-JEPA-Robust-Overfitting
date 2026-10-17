# Project guidance

## Current stage

This repository has a scaffold and locally tested model integration. Read README.md, progress.md, and
docs/experimental_protocol.md before adding research code. Do not describe
planned entry points, unrun experiments, or proposed results as implemented
or verified. Keep progress.md factual.

Small CPU fixtures pass; full pretrained execution, video data, attacks,
training, CUDA feasibility, and research results remain pending. See
docs/smoke_test.md before running model checks. No cloud rental is automatic.

## Research scope

- V-JEPA 2 ViT-L/16, fixed clean target encoder, native masked L1 prediction.
- Update the context encoder and predictor; do not silently introduce EMA,
  a reinitialized predictor, a classifier training loss, or V-JEPA 2.1.
- Record protocol changes in docs/decisions.md and configs/study.yaml.
- Preserve group-separated train/validation/test manifests. Never select
  checkpoints or tune attacks on the final test set.
- Generate fresh adversaries per checkpoint. Measure epsilon in raw [0, 1]
  pixel space, and keep target inputs clean and target gradients disabled.
- Treat robust overfitting as an empirical question. Report failed checks,
  negative findings, and insufficient training budgets plainly.

## Repository conventions

- Keep the familiar root models/, scripts/, notes/, and report/ layout.
- Put reusable implementation in modules; keep command-line scripts thin.
- Use condition/seed run directories and unique names for reruns.
- Track small curated reports and relative-path manifests; keep weights,
  videos, feature caches, credentials, and raw logs out of Git.
- Pin upstream code and checkpoint hashes before any experiment claims.
- Preserve upstream licenses and attribution; our MIT license covers only
  original project code.

## Verification

Run python scripts/verify_setup.py for scaffold changes. This is not a
scientific correctness test. When implementing attacks or training, add
meaningful gradient, projection, masking, target-freezing, split-integrity,
and checkpoint-resume checks, as described in tests/README.md.

## Compute

The total project budget is at most $200, with $160 planned for compute.
Estimate the remaining experiment from measured throughput before launching
full runs. Count evaluation, idle instance time, storage, and taxes. Keep
the university GPU and local development paths usable.
