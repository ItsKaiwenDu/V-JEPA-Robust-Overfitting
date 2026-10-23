# Shared-forward training integration

October 22, 2026. The shared-forward candidate is now an explicit **diagnostic**
option in training, evaluation, caches, calibration, features and checkpoints.
Local small upstream CPU models pass full clean/adversarial AdamW recovery,
including fresh processes. This is not full pretrained CUDA optimizer/recovery
acceptance or study admission. See the [local verification record](../report/checks/2026-10-22-shared-forward-integration.md).

## Explicit policy and arithmetic

The bounded `check_workflow.py` and `run_workflow_pilot.py` commands accept:

| Policy | Context patch forward | Context patch backward |
| --- | --- | --- |
| `shared-fp64-cudnn` | FP64 unfolded patch matrix, separate FP64 bias, cast once to FP32 transposed tokens | Strict FP32 cuDNN convolution VJP |
| `shared-fp64-native` | The identical FP64 expression and FP32 token layout | Strict FP32 native convolution VJP |

Both reuse the original projection parameter objects and state keys. Learned
tensors, raw-pixel normalization, tokens, target encoder, transformer/predictor,
and native masked L1 remain FP32. FP64 temporaries exist only inside the context
patch forward. The target keeps its original clean, frozen Conv3d path; its
normalized features remain detached. No predictor layer is reinitialized.

The first-order convolution VJP approximates the real affine derivative and
has separate numerical evidence against FP64. It is not a derivative through
discrete rounding. Autocast, higher-order differentiation and MPS execution
are unsupported. CUDA integration requires PyTorch 2.8.0, matching the already
exercised private ATen boundary. Configuring a policy on another CUDA version
rejects before CUDA initialization.

The existing `original`, `cudnn-fp32`, and `native-fp32` defaults and schema-1
execution identities remain unchanged. The temporary `shared-fp64-forward-*`
scope used by the earlier precision control remains a separate identity that
the general training/cache APIs reject. The two integrated names above do not
make that historical scope interchangeable with training.

## Runtime and artifact identity

[execution.py](../models/execution.py) records a schema-2 contract with the
forward algorithm ID, transient accumulation and token/parameter/VJP dtypes,
selected backward algorithm, normalization values, actual context/target and
model classes, convolution geometry, and activation checkpointing. The outer
runtime still records deterministic settings, cuDNN/TF32/cuBLAS flags, hardware
and library versions. Actual tensors, frozen target state, normalization,
autocast and dispatch must match the declaration.

Existing equality checks carry this complete identity into fresh in-memory
target caches, feature/probe bundles, calibration, evaluation journals, training
states and saved anchors. Records lacking forward precision, using an old
forward, or selecting another backward cannot match. Resume and anchor loading
reject before loading trainable weights or optimizer state. Equal weight hashes
and compatible state keys do not imply compatible computation.

Training permits ordinary context/predictor weight updates, while checking
parameter/module identity and frozen-target mutation before update/save/restore.
Cached targets use the same process-local execution guard. Target hashes remain
independently checked in saved state. Reconstruct the model and fresh artifacts
after changing policy; do not relabel a saved record to force acceptance.

The resume payload's existing `precision: float32` describes learned tensors
and optimizer storage. Its schema-2 `runtime.model_execution.precision` also
binds the FP64 context-patch accumulation; the storage label alone is not an
execution identity.

## Local acceptance

The [integration tests](../tests/test_shared_execution.py) exercise both
backwards and both training conditions using small actual pinned upstream
transformers, generated raw pixels, both mask families and activation
checkpointing. They retain the original AdamW settings, partial-batch
normalization and exact comparison. Checks cover:

- Nonzero context/predictor updates and moments, unchanged target weights,
  detached clean targets, parameter objects and checkpoint keys.
- Interrupted recovery at partial-batch, warmup and decay boundaries, including
  optimizer moments, learning rates, sampler order, mask/attack seeds, history,
  and Python/NumPy/Torch RNG. Separate fresh processes also compare exactly.
- Scheduled evaluations and per-clip/epoch/evaluation-record interruptions;
  training cannot skip a pending evaluation.
- Valid anchor round trips and incompatible forward/backward or missing
  precision rejected before tensor/optimizer restoration.
- Cache/runtime/module/geometry/normalization/precision/checkpointing/target
  drift, and incompatible calibration, evaluation and feature-probe artifacts.

Run the project suite with:

```bash
.venv/bin/python -m pytest -q tests
```

The original gradient, projection, oracle, target-freezing and failed-evidence
tests remain in that suite. Earlier numerical failures stay rejected. The
two CUDA-only pytest skips on this Mac are not CUDA passes.

## Next gate

The [October 22 pretrained CUDA workflow](../report/checks/2026-10-22-shared-forward-cuda-workflow.md)
now passes the bounded representative checks and clean/adversarial two-update
exact recovery, including integrated evaluation. All eleven phases and final
controls completed; copied evidence is independently verified. The earlier
university 8 GB AdamW OOM remains unresolved. Peak allocator use on RTX 5090
does not establish that another GPU fits.

Next, reconcile the provisional charge, rebuild measured costs and complete
data review, research calibration and precision/protocol freezing before any
research trajectory. No input-only speed ratio or old planning envelope admits
the study. Study/proposal YAML and original numerical limits are unchanged;
confirmation/test pixels remain closed. The user fully terminated the Pod.
