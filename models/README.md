# Model integration

This folder replaces the previous project's PreActResNet implementation with
the V-JEPA integration. It contains no model weights.

Implemented in [vjepa2.py](vjepa2.py):

- Construct the official V-JEPA 2 ViT-L/16 architecture from a pinned source
  revision and strictly load context, target, and predictor states from a
  local checkpoint. The official checkpoint has passed the CPU smoke test.
- Expose visible-context prediction and clean-target feature extraction.
- Preserve upstream target normalization, mask indexing, and loss averaging.
- Freeze the target encoder and expose differentiable context inputs for attacks.
- Support the same adapter in training, evaluation, and scientific checks.

[diagnostics.py](diagnostics.py) checks native-loss parity against the pinned
upstream training function, raw-pixel and parameter gradients, replay, and
target freezing. Local tests cover small upstream models and supporting
data/attack code. The full released weights also passed the native-loss
CPU check. [pilot.py](pilot.py) contains a tested effective-batch update for
the upcoming CUDA feasibility run, which remains unexecuted on CUDA.
See [the guide](../docs/smoke_test.md).

Keep the upstream source in [external/](../external/README.md) and weights in
ignored `checkpoints/`. Wrap upstream code rather than copying a second model
implementation without provenance. Preserve its license if any code is
vendored later.

The raw checkpoint's `encoder`, `target_encoder`, and `predictor` states must
be audited. A convenience loader that returns only a downstream encoder is
not a substitute for verifying the full prediction path.
