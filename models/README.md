# Model integration

This folder replaces the previous project's PreActResNet implementation with
the upcoming V-JEPA integration. It currently contains no executable model
adapter and no model weights.

Planned responsibilities:

- Load the official V-JEPA 2 ViT-L/16 context encoder, target encoder, and
  predictor from a pinned upstream revision and verified checkpoint.
- Expose visible-context prediction and clean-target feature extraction.
- Preserve upstream target normalization, mask indexing, and loss averaging.
- Freeze the target encoder and expose differentiable context inputs for attacks.
- Support the same adapter in training, evaluation, and scientific checks.

Keep the upstream source in [external/](../external/README.md) and weights in
ignored `checkpoints/`. Wrap upstream code rather than copying a second model
implementation without provenance. Preserve its license if any code is
vendored later.

The raw checkpoint's `encoder`, `target_encoder`, and `predictor` states must
be audited. A convenience loader that returns only a downstream encoder is
not a substitute for verifying the full prediction path.
