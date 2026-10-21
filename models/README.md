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
