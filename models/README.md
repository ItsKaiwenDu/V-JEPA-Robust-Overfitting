# Model integration

This folder replaces the previous project's PreActResNet implementation with
the V-JEPA integration. It contains no model weights.

Implemented in [vjepa2.py](vjepa2.py):

- Construct the official V-JEPA 2 ViT-L/16 architecture from a pinned source
  revision and strictly load context, target, and predictor states from a
  local checkpoint. The official checkpoint has passed CPU and MPS smoke tests.
- Expose visible-context prediction and clean-target feature extraction.
- Preserve upstream target normalization, mask indexing, and loss averaging.
- Freeze the target encoder and expose differentiable context inputs for attacks.
- Support the same adapter in training, evaluation, and scientific checks.

[diagnostics.py](diagnostics.py) checks native-loss parity against the pinned
upstream training function, raw-pixel and parameter gradients, replay, and
target freezing. Local tests cover small upstream models and supporting
data/attack code. The full released weights also passed the native-loss
CPU and MPS checks. [pilot.py](pilot.py) contains the effective-batch update;
[runtime.py](runtime.py) provides explicit device access, synchronized GPU
timing, and clearly labeled CUDA peak/MPS sampled memory observations.
The [October 22 shared-forward CUDA workflow](../report/checks/2026-10-22-shared-forward-cuda-workflow.md)
passes bounded pretrained clean/adversarial two-update recovery and integrated
evaluation. Research precision/protocol and budget admission remain separate.
See [the guide](../docs/smoke_test.md).

[patch_embed.py](patch_embed.py) provides an explicit equivalent linear
implementation of the context encoder's nonoverlapping patch projection.
It reuses the original parameters and state keys; the target encoder is
unchanged. The default remains Conv3d. See [Mac performance checks](../docs/mac_pilot.md)
for correctness evidence and the explicit option used in the MPS pilot.

Keep the upstream source in [external/](../external/README.md) and weights in
ignored `checkpoints/`. Wrap upstream code rather than copying a second model
implementation without provenance. Preserve its license if any code is
vendored later.

The raw checkpoint's `encoder`, `target_encoder`, and `predictor` states must
be audited. A convenience loader that returns only a downstream encoder is
not a substitute for verifying the full prediction path.

`training.py` provides the resumable FP32 native-loss engine and cached upstream
mask sampler. `checkpointing.py` commits checksummed resume states atomically
and retains two generations plus anchor weights. See the
[runner guide](../docs/training_runner.md) for evidence and limits.

`evaluation.py` generates fresh native attacks with independent scoring masks.
`evaluation_store.py` preserves per-clip progress and selects only from complete
primary-validation anchors. `evaluation_checkpoint.py` verifies saved training
anchors while preserving the fixed target. See
[staged data and evaluation](../docs/staged_data_and_evaluation.md).

[features.py](features.py) extracts clean pooled context features, fits a fixed
training-only ridge probe, and measures raw variance/effective rank.
[calibration.py](calibration.py) selects a primary PGD step from complete
training-monitor evidence. [schedule.py](schedule.py) couples epoch checkpoints
with ordered, resumable evaluations; [study_artifacts.py](study_artifacts.py)
provides checksummed JSON records. The small diagnostic setup lives in
[workflow_diagnostic.py](workflow_diagnostic.py). See
[workflow diagnostics](../docs/workflow_diagnostics.md) for tested scope and limits.

[cuda_profile.py](cuda_profile.py) implements the separate synthetic component
profiler: actual pinned upstream operations, fixed operands/cotangents,
finite gradient signatures, verified replay evidence, optional operator tables,
and bounded worker groups. It does not replace model components. Read the
[component profile guide](../docs/cuda_component_profile.md) before use.

[cuda_convolution.py](cuda_convolution.py) compares strict execution choices
for the original patch Conv3d using synthetic operands. An independent CPU FP64
patch matrix supplies predictions and analytic input/weight/bias derivatives.
Fixed numerical gates precede timed fresh-process replay; rejected/incomplete
options cannot receive a speed ratio. No research component is replaced.
The original TF32-permitted cuDNN reference failed the FP64 gate on CUDA, so
no option qualified or was timed. The explicit precision control is separate
from the study: `--cudnn-tf32 off` is recorded in each fresh worker and cannot
be mixed with TF32-on evidence. See the
[comparison guide](../docs/cuda_convolution_comparison.md). Its TF32-off CUDA
control passes numerical gates and observed replay; native execution qualifies
as an isolated component candidate, with full-model acceptance still pending.

[native_patch.py](native_patch.py) adds an explicitly first-order native Conv3d
diagnostic; [execution.py](execution.py) binds actual backend, precision and
cache/recovery identities. [native_diagnostic.py](native_diagnostic.py) compares
full-model input gradients and attacks on one training clip without an optimizer
step. See [the opt-in guide](../docs/native_execution.md); defaults and study
admission are unchanged.

[gradient_attribution.py](gradient_attribution.py) isolates patch backward from
incoming tail derivatives using shared cotangents, an independent FP64 raw-pixel
oracle, retained residuals and reconstruction of the saved rejected CUDA run.
It is a diagnostic intervention, never a training/backend acceptance path. See
[the control guide](../docs/native_gradient_control.md); the verified CUDA control rejects forward accuracy while fixed-cotangent backwards and exact reconstruction pass.

[precision_patch.py](precision_patch.py) and [precision_diagnostic.py](precision_diagnostic.py)
implement the separate FP64-context-patch-forward/FP32-token diagnostic. Only
backward dispatch differs between arms; a distinct guarded scope is rejected
by existing research/cache APIs. The bounded CUDA input-only control is now verified: oracle/replay gates pass
with identical shared tokens and attacks on one clip. Explicit integration and small-upstream CPU AdamW recovery now pass; full
pretrained CUDA optimizer recovery remains pending. See
[the declared precision/reference](../docs/shared_forward_control.md).

The named `shared-fp64-cudnn/native` options in [execution.py](execution.py)
carry a complete schema-2 precision identity into training and artifact
consumers. See [shared-forward training](../docs/shared_forward_training.md)
for exact CPU recovery evidence, supported precision and remaining CUDA gates.

[workflow_preflight.py](workflow_preflight.py) gates shared-forward CUDA workflow
calibration with three fixed training inputs, both backwards, independent
oracles and retained scientific tensors. See [the fresh handoff](../docs/shared_forward_cuda_pilot.md).
