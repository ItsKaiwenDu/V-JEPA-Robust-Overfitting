# Local model checks

This is the first implementation step. It does not require Lambda, a dataset,
or access to the university machine. The CPU tests use small randomly
initialized instances of Meta's architecture. A separate command is ready
for the full released checkpoint, but that command has not yet been run.

## Create an isolated environment

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 torchvision==0.29.1
python -m pip install -r requirements-smoke.txt
```

This pair was tested locally on macOS ARM64 with Python 3.11.16, using the
CPU. These are not validated CUDA environment pins. For a later Linux CPU
check, use PyTorch's CPU wheel index for the torch/torchvision installation;
for a GPU pilot, select a compatible pair using the
[official installation instructions](https://pytorch.org/get-started/locally/)
and record the resulting environment. No video decoder is needed here.

Fetch the separately licensed upstream source at the recorded revision:

```bash
git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
git -C external/vjepa2 checkout --detach 204698b45b3712590f06245fbfba32d3be539812
```

On a case-insensitive Mac filesystem, upstream may warn about colliding
`vitG`/`vitg` paths in V-JEPA 2.1 configuration folders. This integration uses
V-JEPA 2 only and checks the revision and cleanliness of `src/`, `app/vjepa/`,
and `configs/train/vitl16/`. It does not modify upstream code.

## Run the small CPU tests first

```bash
python -m pytest -q tests --upstream external/vjepa2
python scripts/verify_setup.py --check-torch
```

The tests check:

- Native L1 loss against the actual function extracted from the pinned
  training source, using upstream encoder/predictor wrappers independently.
- Equal averaging over differently sized mask families and per-example losses.
- Finite raw-pixel gradients, an FP32 finite-difference comparison, and zero
  gradients on hidden pixels when only one context mask is used.
- Clean detached targets, a target encoder that stays frozen in training
  mode, and updates restricted to the context encoder and predictor.
- Strict three-component loading, all ten predictor mask tokens, rejection
  of missing/extra/wrong-shaped tensors, and checkpoint hash verification.
- Deterministic block-mask replay, disjoint mask indices, and input validation.
- Input and parameter differentiation with activation checkpointing enabled.

The fixtures preserve the released attention head dimensions but have fewer
heads, layers, and tokens. They do not measure pretrained robustness.
Upstream emits deprecation warnings for timm imports and PyTorch's SDPA
context manager; these are recorded rather than hidden or patched upstream.

## Full pretrained smoke test — subsequent step

The official file is approximately 5.13 GB (4.78 GiB). Downloading and
running this check is explicit; neither the unit tests nor the script will
download it automatically. No GPU rental is part of these commands.

```bash
curl --fail --location --output checkpoints/vitl.pt.partial \
  https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
mv checkpoints/vitl.pt.partial checkpoints/vitl.pt
```

Choose a fresh destination if that checkpoint already exists. Record its
SHA-256 (`shasum -a 256 checkpoints/vitl.pt` on macOS or `sha256sum` on
Linux). A locally computed hash records which bytes were used; it is not an
independent signature of authenticity. Keep the official source URL too.

On an available CUDA machine with enough memory, for example:

```bash
python scripts/smoke_test.py \
  --upstream external/vjepa2 \
  --checkpoint checkpoints/vitl.pt \
  --device cuda \
  --output runs/pretrained/diagnostic/native-loss-01/smoke.json
```

Use `--expected-sha256 <recorded-hash>` to require a known hash. The script
always computes and records the actual hash, even when this option is
omitted. It loads with `weights_only=True`, audits all three states, and
uses `strict=True`; it never falls back to reinitializing trained layers.
`--device cpu` and `--device mps` are explicit alternatives. Full ViT-L
memory use and runtime on those devices remain unmeasured; the default is
CPU, not automatic GPU selection.

This command checks one synthetic 16-frame, 256-square FP32 clip. It checks
native-loss parity, finite nonzero input and parameter gradients, target
freezing, and replay. It does not take an optimizer step. Reports record
the upstream revision, checkpoint hash, package versions, project commit
and dirty state, mask dimensions, and check results. Existing report files
are rejected. An exception or nonzero exit status is a failed check, not a
partial pass. Save the console output when investigating failures.

After a successful full-checkpoint run, record its hash in
`configs/study.yaml` and promote the small reviewed report to `report/checks/`.
The next stages are real-video preprocessing, attack correctness, and an
end-to-end GPU cost pilot. Only that pilot can justify a training schedule
or a Lambda rental.

## Upstream implementation details

The integration pins
[Meta's source revision](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812).
At this revision, `src/hub/backbones.py` uses a localhost testing base URL
and non-strict loading. Our code constructs the upstream architectures
directly and takes an explicit local checkpoint instead.

The released checkpoint metadata contains 292 encoder state tensors, 292
target-encoder tensors, and 160 predictor tensors, including ten mask
tokens. The adapter keeps all ten. For this single sequence-length group,
both mask families use mask token index 0, matching
`PredictorMultiSeqWrapper`; mask-family index and sequence-group index are
different concepts. ImageNet mean/std normalization stays inside the
differentiable raw-pixel path, and target features receive the upstream
layer normalization before masked L1 comparison.
