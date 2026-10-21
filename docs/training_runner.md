# Training and checkpoint recovery

October 20, 2026. The FP32 training core and a bounded diagnostic command are
implemented. **Full research execution remains pending**: the staged evaluation/selection core and exploration data now have separate
[implementation evidence](staged_data_and_evaluation.md), and bounded epoch orchestration now has
[workflow evidence](workflow_diagnostics.md). Full-model CUDA recovery validation
and end-to-end budget admission remain pending.

The training-core verification recorded **95 passing tests**, including 29
new training/input checks; the subsequent data/evaluation milestone had 144.
The latest workflow suite has 197. Small models built from the pinned upstream JEPA implementation
produce exactly matching CPU model weights, optimizer moments, histories,
and random states after interruption and resume. Both clean and adversarial
conditions were checked at three interruption points, spanning warmup,
learning-rate drops, and final partial batches. This is implementation
evidence, not pretrained-model training or a robust-overfitting result.
See the [verification record](../report/checks/2026-10-20-training-resume-checks.json).

## Implemented behavior

- [Training core](../models/training.py): matched epoch ordering and mask seeds,
  separate clean/adversarial conditions, AdamW, per-update warmup, epoch-based
  LR drops, and microbatch-one accumulation. It reuses the previously tested
  [native update](../models/pilot.py), preserving PGD-5 at epsilon 4/255.
- [Prepared input loader](../datasets/prepared.py): accepts only the pinned
  training manifest, checks membership and file metadata, verifies raw-pixel
  hashes, and recomputes fixed clean targets into CPU memory per attempt.
  No validation/test data enters the training command.
- [Checkpoint store](../models/checkpointing.py): atomic commit manifest,
  SHA-256 verification, two rolling resume generations, retained anchor
  weights, and free-space checks before writes.
- [Diagnostic CLI](../scripts/train.py): explicit device, condition and seed;
  at most eight additional updates on at most eight prepared training clips;
  fresh attempt directory; bounded wall-time and graceful stop requests.

The final partial effective batch divides its loss by its actual number of
examples. The 100-video proposal therefore has 12 eight-example updates and
one four-example update per epoch. In the CPU fixture, a one-example final
batch matches an independently computed single-example optimizer update.

Warmup is linear over the first epoch's update count, with multipliers
`1 / updates_per_epoch` through `1`. LR reductions take effect in epochs
16 and 24 for the proposed drops after epochs 15 and 23. There is no separate
mutable scheduler object: the saved completed-update count and immutable
settings determine the next LR. FP32 has no gradient scaler; the checkpoint
records this explicitly and rejects a mixed-precision continuation.

Training seeds use compact JSON serialization of
`["training-v1", seed, namespace, ...]`, SHA-256's first eight bytes interpreted
big-endian, modulo `2**31 - 1` for compatibility with upstream mask sampling.
Order uses the zero-based epoch; masks use epoch and video ID; attack starts
use the zero-based update, with a per-microbatch offset inside the existing
PGD routine. Condition is omitted so matched pairs share order/masks. This
training seed rule is separate from the evaluation-bank
seed rule. The cached upstream mask sampler reproduces the existing smoke
sampler while avoiding repeated Git/config inspection inside every update.

## What a resume checkpoint contains

The checkpoint includes context/predictor state, AdamW state, completed update
count, scientific update history, and Python, NumPy, PyTorch CPU, and active
GPU RNG state. The run contract binds settings, ordered video IDs, initial
checkpoint/target identity, prepared-input and manifest hashes, source hashes,
and key runtime fields. The fixed target is reconstructed from the verified
original checkpoint, not duplicated in every resume file.

Resuming rejects changed contracts, invalid model tensors, inconsistent
histories, changed optimizer settings, or missing moment state before loading
the trainable weights. Input bytes are checked again when loaded. An exception
inside accumulation marks the engine unsavable; recovery starts from the last
committed boundary. Abrupt termination can lose work since the last checkpoint,
which must be recomputed. It does not justify serializing partial gradients.

Each save writes unique temporary files, flushes them, hashes them, renames
them, then atomically replaces the manifest. Only after that commit does it
retire the oldest referenced resume generation. Anchor weight files remain.
If a transaction fails, previously committed files survive; an orphan from
the failed write may remain and is not considered a checkpoint. A corrupted
latest generation causes an error. Recovery from `previous` must be explicit.

Every `train.py` CLI attempt writes to a **new directory**, including resumes.
Its new run record links the parent checkpoint filename/hash and attempt directory.
Earlier attempt logs and anchor files remain intact. Two rolling generations
are retained per attempt; this does not implement automatic archival pruning
across attempts. Checkpoint I/O, CPU target-cache construction, disk occupancy,
and transfer costs still need measurement on the experiment machine.

The newer `check_workflow.py` uses explicit in-place recovery of one trajectory;
see [its directory contract](workflow_diagnostics.md).

PyTorch documents optimizer state as part of resumable checkpoints in its
[checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html).
Its [reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
also explains why identical seeds alone cannot promise equality across
versions, platforms, or devices. The exact-match evidence here is local CPU
evidence. CUDA/MPS RNG save/restore paths are implemented but have not been
validated by the new resume tests on those devices.

## Local verification

After the existing development dependencies and pinned upstream checkout are
available, run from the repository root:

```bash
python -m pytest -q tests/test_training.py tests/test_prepared.py
python scripts/train.py --help
```

These fixture tests require no full checkpoint, dataset download, GPU rental,
or network connection. Existing scientific and planning tests were also run.
The 35 warnings in that full run come from upstream deprecated import/SDPA
APIs; no tests failed or were skipped.

## Future full-model diagnostic command

These commands are for a later explicitly chosen device session. They have
not been run on the full model in this implementation step. They use the
existing eight pilot clips and therefore do not constitute the proposed
100-video experiment. A new GPU session still needs its bounded spending
plan and current quote.

```bash
python scripts/train.py --diagnostic \
  --checkpoint checkpoints/vitl.pt \
  --clips data/processed/pilot-train-v1 \
  --condition adversarial-finetune --seed 0 --device cuda \
  --max-updates 2 --max-seconds 600 \
  --output runs/adversarial-finetune/diagnostic/resume-check-01

python scripts/train.py --diagnostic \
  --checkpoint checkpoints/vitl.pt \
  --clips data/processed/pilot-train-v1 \
  --condition adversarial-finetune --seed 0 --device cuda \
  --max-updates 2 --max-seconds 600 \
  --resume-from runs/adversarial-finetune/diagnostic/resume-check-01 \
  --output runs/adversarial-finetune/diagnostic/resume-check-02
```

Run clean and adversarial conditions from separate initializations. Keep the
same environment, sources, plan, clips, condition and seed when resuming. For
an explicitly verified recovery from an unusable latest file, add
`--resume-generation previous` and use another fresh output directory.

SIGINT/SIGTERM request a stop after the active optimizer update finishes. The
wall-time check also runs at update boundaries, so setup, a slow update, or
checkpoint writes can overrun that limit. Neither this limit nor a completed
Python process stops a cloud Pod or its billing. The later launch workflow
must handle instance shutdown and verified artifact export separately.

The staged training/exploration clips and native evaluation/selection core are
now checked separately. Fixed probes, calibration records, and bounded epoch
evaluation orchestration also have [local evidence](workflow_diagnostics.md).
Next: full-model CUDA recovery and missing-cost measurement before an
exploratory trajectory. No research protocol has been frozen.
