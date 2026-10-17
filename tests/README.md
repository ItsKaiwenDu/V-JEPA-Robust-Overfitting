# Scientific acceptance tests

Forty-eight local tests exercise the pinned upstream architecture,
pixel attacks, grouped dataset splits, video sampling/decoding, and the
pilot's optimizer update. This includes an FFmpeg AVI regression test;
it is explicitly skipped when FFmpeg/FFprobe are unavailable. Run with
`python -m pytest -q tests
--upstream external/vjepa2` after [local setup](../docs/smoke_test.md).

The GitHub workflow still checks only the scaffold and Python syntax; the
tests have been run locally, and separate full-pretrained/real-clip reports
are in `report/checks/`. CUDA, complete dataset integrity, stronger attack
convergence, and training-resume checks remain upcoming. Add tests alongside the
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
