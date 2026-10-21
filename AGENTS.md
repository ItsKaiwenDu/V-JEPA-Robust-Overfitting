# Project guidance

## Current stage

Read README.md, progress.md, docs/experimental_protocol.md, and
docs/workflow_diagnostics.md before adding research code. Do not describe
planned entry points, unrun experiments, or proposed results as implemented
or verified. Keep progress.md factual.

Pretrained hardware feasibility and staged data preparation passed. The repaired
suite passes 225 tests on the CUDA host, including small independent-process
CUDA replay. The full-model workflow pilot failed exact clean recovery. Its
strict deterministic retry was stopped after a primary attack took 16x longer;
full-model recovery is still unvalidated. Read
report/checks/2026-10-21-cuda-workflow.md and docs/cuda_workflow_pilot.md.
Both evidence exports are verified locally; the user confirmed Pod termination.
Reported spending is $1.195 after the corrected $0.603 session charge.
The university component profile is verified: all 48 workers completed; all 12
strict cases replay exactly. Original Conv3d input backward rose from 1.102 ms
to 3.400 s under strict policy; default attention gradients fail exact replay.
Read report/checks/2026-10-21-university-cuda-components.md and
docs/cuda_component_profile.md. This is isolated RTX 3070 evidence, not a
full-model RTX 5090 recovery or performance result. The bounded same-Conv3d
comparison completed on CUDA, but all 12 workers rejected before timing:
even the original TF32-permitted cuDNN reference fails the FP64 accuracy gate.
Native matches FP64 but differs from that reference. Read
report/checks/2026-10-21-university-cuda-convolution.md for that failed attempt.
The explicit TF32-off control now passes all 12 workers, numerical gates, and
observed exact replay in both gradient modes. Original cuDNN forward/backward
is 3.410 / 3.431 seconds; native is 5.897 / 7.043 milliseconds (input / parameter
gradient modes). Channels-last provides no useful improvement. Read
report/checks/2026-10-21-university-cuda-convolution-precision.md and
docs/cuda_convolution_comparison.md. This resolves the isolated precision gate
and qualifies native as a component candidate; it does not establish RTX 5090
behavior or full-model correctness/speed. No study backend/precision is adopted.
Next: prepare an opt-in integration with explicit cache/resume identities and
full-model gradient/recovery checks before another measured pilot.
Latest local suite: 283 passed, one actual-CUDA skip. Research calibration,
measured budget admission, and research results remain pending. Confirmation/test pixels remain closed.
No cloud rental is automatic.

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
