# Shared-forward training integration — October 22, 2026

**Outcome:** explicit diagnostic integration passes local CPU full-update and
exact recovery checks. **420 passed, 2 actual-CUDA skips, 159 warnings
in 204.35s.** The 47 new integration cases use small actual pinned
upstream models and generated pixels. No pretrained CUDA optimizer recovery,
new GPU performance result or research admission follows from these CPU checks.

## Implemented behavior

`shared-fp64-cudnn` and `shared-fp64-native` opt into the validated FP64 context
patch forward / FP32 tokens with explicitly selected FP32 convolution VJPs.
Original parameter objects/state keys, clean frozen target, trained-predictor
loading, normalization and native L1 are preserved. The projection autograd
arithmetic is AST-identical to the previously validated implementation.
Geometry/backend checks were strengthened; imports are lazy where needed to
avoid a circular dependency with the execution/training contract.

The schema-2 runtime binds actual modules/geometry, normalization values,
forward algorithm and transient/token/parameter/VJP precision, backward
selection, activation checkpointing, and existing strict global runtime flags.
Existing cache/feature/calibration/evaluation/checkpoint equality checks now
carry this identity without weakening their comparisons. Missing precision or
another forward/backward rejects; resume and anchor mismatches reject before
loading trainable tensors or optimizer state. The original policies retain
schema 1, and the earlier temporary precision scope stays separately rejected.
CUDA candidates require the exercised PyTorch 2.8.0 boundary; MPS/autocast and
higher-order operations are unsupported. Defaults remain `original`.

## Verification and retained failures

Command: `.venv/bin/python -m pytest -q tests`.

- Twelve clean/adversarial full-small-model recovery cases cover both backwards
  at three interruption cuts, warmup/decay, partial batches, exact AdamW state,
  history, sampler order, seeds, RNG, nonzero permitted updates/moments and
  unchanged targets.
- Four fresh-process recovery cases use twelve CPU workers; complete saved
  model/optimizer/history/RNG state and subsequent RNG draws match exactly.
- Four scheduled trajectories compare exact logical state and evaluation
  metrics. Three additional failure cases cover inside-panel progress, a
  committed epoch before evaluation, and an evaluation completed before its
  workflow-record write. No pending evaluation is skipped.
- Positive anchor loads preserve targets; mismatched or absent precision,
  drifted modules/parameters/geometry/normalization/dtype/target/checkpointing,
  autocast, stale calibration/evaluation and incompatible feature probes reject.
- Existing independent FP64 gradient/projection, PGD bound, target, historical
  failure and runtime tests remain in the full suite. Two CUDA-only skips remain
  skips; the earlier verified university CUDA input-only result is separate.

The first new focused run had 43 passes and two failed **test setup assertions**.
One read the new engine step before the workflow's lazy restoration; the other
changed global flags before serializing the source engine. Tests now read the
committed step and serialize before selecting the destination runtime. The
corrected eight-case recheck passed; the final full suite above is clean.
Initial JUnit failure evidence is retained with final XML/logs under ignored
`runs/pretrained/diagnostic/shared-forward-integration-20261022/`.
No numerical or exact-equality limit was relaxed.

## Provenance and limits

The [JSON record](2026-10-22-shared-forward-integration.json) retains implementation, final/initial test
artifact and unchanged scientific/configuration hashes. Study/proposal YAML,
training core, original model/loss/attack definitions, oracle tolerances,
full-model gradient tolerances and exact comparison implementation are unchanged.
Historical rejected FP32 runs and their archives remain preserved.

No real dataset pixels, new CUDA job or paid compute were used. No commit or
push was performed by this integration work. Spending remains $1.195, shared
pilot reserve $8.805, total cap $200 / compute plan $160.

**Next:** prepare the fresh offline bounded pretrained CUDA update/resume and
representative-input pilot, with actual new dispatch, full checkpoint/cache/
evaluation costs and a current quote before deployment. Earlier full AdamW
exceeded the university GPU's 8 GB; input-only memory does not establish fit.
Full research calibration/convergence, novelty refresh, protocol freezing and
budget admission remain pending. [Integration guide](../../docs/shared_forward_training.md).
