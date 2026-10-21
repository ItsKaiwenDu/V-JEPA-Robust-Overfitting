# Scientific acceptance tests

The latest local suite passes **248 tests with one actual-CUDA skip** and 85
upstream warnings. This includes 24 component-profiler checks: independent
gradients, representative masks, actual subprocess/descendant timeouts, rejected
tampered evidence and changed precision/source, fresh-process fixture replay,
and optional profiler tables. New university CUDA profiling remains pending.

The repaired suite has **225 passing tests on the CUDA host**, with no
failures/skips and 82 upstream FutureWarnings. The original local CPU suite had
219 passing tests; the targeted local runtime/training recheck passed 28 with
one actual-CUDA skip. The preceding 53 additions check fixed ridge probes
against an independent primal solution, training-only standardization, raw
feature diagnostics, calibration evidence/replay, and scheduled recovery.
Small pinned-upstream clean/adversarial models exactly match uninterrupted
training after recovery, including weights, optimizer, history, and RNG.

Interruption tests cover per-clip evaluation, a saved epoch before evaluation,
and completed evaluation before its workflow-record commit. Stale source/runtime,
changed schedules, missing/corrupt receipts, and mixed-run records are rejected.
The previous 144 checks retain their coverage of native losses/gradients,
attacks, group splits, decoding, checkpoint state, primary selection, and costs.

The preceding 22 additions exercise the bounded pilot phases, exact recovery and
optimizer-tampering rejection, incomplete/corrupt evidence, fresh-versus-reused
evaluation timings, worker deadlines, and reproducible/archive-safe handoffs.
These use small real upstream models. Six additional runtime checks cover
strict configuration, conflict/late-init rejection, incompatible resume policy,
and actual independent-process CUDA gradient/update replay (skipped without
CUDA). The full-model pilot failed replay under the original runtime, and its
strict retry was stopped for runtime cost. A passing small CUDA check does not
establish full-model correctness or feasibility. See the
[CUDA outcome](../report/checks/2026-10-21-cuda-workflow.md).

Run `python -m pytest -q tests --upstream external/vjepa2` after
[local setup](../docs/smoke_test.md). GitHub CI currently checks scaffold and
syntax only. These tests do not require downloading weights or renting a GPU.

The separate full pretrained CPU feature command passed on eight real training
clips; its training-set probe score is not held-out performance. Full-model
CUDA recovery/calibration, convergence validation, near-duplicate review, and
research admission remain pending. See [workflow diagnostics](../docs/workflow_diagnostics.md)
and the [latest evidence](../report/checks/2026-10-20-workflow-checks.json).

Add tests alongside the
corresponding implementation, including:

- Pixel-space perturbation bounds, clipping, and zero-epsilon equivalence.
- Finite, nonzero input gradients and a small FP32 finite-difference check.
- Frozen target weights, clean target inputs, and correct detached features.
- Correct context/target token alignment and independent mask evaluation.
- Agreement with the upstream clean native loss on the same batch and masks.
- Dataset source-group separation, manifest replay, and cache invalidation.
- Parameter updates restricted to context encoder and predictor.
- Checkpoint resume preserving optimizer, scheduler, scaler, and RNG state.
- Stronger-attack/restart checks and per-example candidate selection.

Use small fixtures for correctness tests; do not require a full dataset or
large checkpoint in ordinary CI. GPU integration checks should be explicit,
with provenance and measured runtime. Do not label skipped scientific tests
as evidence of a validated pipeline.
