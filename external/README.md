# Upstream source

The intended upstream is [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2).
This repository does not currently contain a clone, vendor copy, or submodule.

The model integration pins commit `204698b45b3712590f06245fbfba32d3be539812`.
Place a checkout at `external/vjepa2/` using the commands in
[the smoke-test guide](../docs/smoke_test.md). The code checks the revision
and rejects modified model/training files. Content beneath `external/`
other than this file is ignored by our Git repository. Local tests so far
used a separate temporary checkout, not a committed or vendored copy.

Use the official V-JEPA 2 ViT-L checkpoint advertised by the upstream project.
Store weights under `checkpoints/` and record their SHA-256. Verify the
context/target/predictor keys and shapes before claiming that the native
prediction objective is available.

Preserve upstream notices. Our MIT license does not apply to upstream files,
weights, or datasets. If local upstream changes become necessary, save a
reviewable patch or fork reference and include it in run provenance.
