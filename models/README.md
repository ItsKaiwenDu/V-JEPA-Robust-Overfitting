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
The October 19 RTX 5090 diagnostic pilot passed; new full-model recovery and
evaluation validation remain pending.
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
