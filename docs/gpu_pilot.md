# CUDA feasibility and the university workstation

The pretrained model and local correctness checks can run on CPU. Measuring
training feasibility requires the actual GPU that will run the experiment.
No Lambda instance is created by this project, and no paid compute has been
used. The reported allocation is `linux10604`, with an RTX 3070 (standard
8 GB), i9-11900, and 62 GiB system RAM. Its `nvidia-smi` currently fails with
a driver/library mismatch (NVML 580.178); system Python is 3.9.25. CUDA
feasibility remains untested until access is repaired and a compatible
Python environment is prepared. The [Apple GPU guide](mac_pilot.md) covers
the separately available M4 Max.

## First, inspect the assigned machine

Run these commands on an allocated GPU machine, following university rules
for SSH and any scheduler:

```bash
nvidia-smi
python3 --version
```

If that machine already has a PyTorch environment, also run:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Share those outputs and any scheduler instructions. Alternatively, provide
an existing SSH alias/hostname and the allocation procedure so the check can
be run through the configured access. No credentials are needed in this chat.

## Prepare the environment and pilot assets

CUDA dependency versions have not been validated. Choose a compatible
PyTorch/torchvision pair using the [official installer](https://pytorch.org/get-started/locally/),
then install `requirements-smoke.txt`. The Mac's wheel build is not a CUDA
environment. Keep the pinned upstream checkout under `external/vjepa2/`.

Bring the official `checkpoints/vitl.pt` and the eight prepared clips plus
`index.json` from `data/processed/pilot-train-v1/`. Videos, caches, and weights
are ignored by Git, so cloning this repository alone does not transfer them.
The checkpoint hash recorded in `configs/study.yaml` must match. Prepared
pixel hashes are checked by the pilot; re-decoding requires preserving the
recorded preprocessing environment or recording a new cache version.

Run the correctness check before timing the pilot:

```bash
python scripts/verify_setup.py --require-cuda
python scripts/smoke_test.py \
  --checkpoint checkpoints/vitl.pt --device cuda \
  --expected-sha256 5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012 \
  --output runs/pretrained/diagnostic/cuda-smoke-01/smoke.json
```

Then run the small, explicit feasibility job:

```bash
python scripts/pilot.py \
  --checkpoint checkpoints/vitl.pt \
  --clips data/processed/pilot-train-v1 \
  --device cuda --steps 3 \
  --output runs/pretrained/diagnostic/cuda-pilot-01/pilot.json
```

This invocation requires CUDA and refuses to overwrite a report. It performs
one warmup plus three measured optimizer updates for each condition, starting
both from the released weights. Each update accumulates eight single-clip
microbatches. It uses FP32, activation checkpointing, clean cached targets,
and native PGD-5 for the adversarial condition. The pilot's AdamW settings
are explicit: encoder/predictor LRs 1e-5/1e-4, weight decay 0.04, betas
0.9/0.999, epsilon 1e-8, `foreach=False`, and no gradient clipping.

It also times PGD-20, two restarts, four attack-mask draws, and four separate
scoring draws on one training clip. The pilot uses step size 1/255 solely
for this timing check; confirmatory evaluation settings remain unresolved.
Memory exhaustion or a failed check must be reported, not counted as a pass.
No model trained by this pilot is saved or used as an experimental result.

## Interpret the report before allocating a full run

The report provides update times, peak allocated GPU memory, evaluation
seconds per clip, and a rough projection for the six 30-epoch training runs.
That projection is training-only and uses a small, warm-cache workload.
It excludes routine validation, stronger attacks, epsilon sweeps, target-cache
construction, checkpoint I/O, probes, storage, and idle time. There is no
automatic budget approval.

Use the measurements to choose precision/microbatch and routine evaluation
sizes, then include all work in the $160 compute / $200 total budget. If the
university GPU cannot fit the workload or meet the timeline, compare rental
options at that point. A rental is not the next automatic action.
