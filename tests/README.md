# Scientific acceptance tests

The full local suite now has **144 passing tests**, including the earlier 29
training/input checks and 49 new staged-data, native-evaluation, recovery,
anchor-loading, and primary-selection checks. Six interrupted/uninterrupted
comparisons cover clean and adversarial trajectories on small pinned upstream
models. Weights, optimizer state, histories, and RNG match exactly on CPU.
Failure tests cover partial updates, failed commits, low disk space,
corruption, changed run contracts, and invalid input provenance. The cached
mask sampler matches the existing upstream sampling path.

Existing checks cover native-loss/gradient agreement, pixel attacks, grouped
splits, FFmpeg decoding, equivalent patch projection, pilots, and both cost
proposals. No test failed or skipped in the October 20 run; 54 FutureWarnings
come from upstream deprecated imports/SDPA APIs. Run
`python -m pytest -q tests --upstream external/vjepa2` after
[local setup](../docs/smoke_test.md).

GitHub CI currently checks only scaffold and syntax. Separate pretrained
CPU/MPS/CUDA diagnostic reports do not establish GPU resume equivalence for
the new runner. See [training and recovery](../docs/training_runner.md).
Exploration clips pass deterministic decoding and exact-duplicate checks; the
primary-selection core is tested. Near-duplicate review, full-model convergence,
GPU resume, probes/calibration, and research orchestration remain pending. See
[the new evaluation evidence](../docs/staged_data_and_evaluation.md). Add tests alongside the
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
