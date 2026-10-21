# Initial cloud pilot handoff

Historical October 19 preparation. The [October 21 CUDA workflow handoff](cuda_workflow_pilot.md)
is the current next diagnostic; this earlier package does not contain the new workflow.

This is a bounded hardware measurement, not the six-run experiment. Review
the [compute proposal](evaluation_compute_plan.md) first. No cloud resource
has been created. Proposed first session: one GPU, no more than two billed
hours, inside the $10 combined pilot allowance. Confirm the provider's live
price, cloud tier, disk cost, and availability before launch.

Kaiwen confirmed an existing Lambda account on October 19. The first
candidate is a **single RTX A6000 with 48 GB VRAM**. Its public listing is
$1.09/hour, or $2.18 for two hours before other charges; confirm the account's
live quote and availability. See [Lambda's catalog](https://lambda.ai/instances).
Account preference does not establish that a rental has been launched.

## Prepared local inputs

The local package directory is `runs/pretrained/diagnostic/cloud-pilot-handoff-20261019/`.
It is ignored by Git and contains:

- `vjepa-pilot-source.tar.gz`: a Git bundle at base commit
  `03e87ddf8f78810ccece272b9ad548c1cf03486e` plus the current tested working-tree
  files. This preserves uncommitted changes; cloning the GitHub branch alone
  would omit them. No `.git/config`, credentials, raw videos, or checkpoints
  are included in this archive.
- `vjepa-pilot-clips.tar.gz`: the eight already-validated training clips and
  their index, under `data/processed/pilot-train-v1/`. No validation/test clips.
- `manifest.json`: file sizes and SHA-256 hashes for the two archives and
  the existing official checkpoint, plus per-source-file hashes.

The 5.13 GB checkpoint remains at local `checkpoints/vitl.pt`; it is not
duplicated inside the archives. Transfer it separately or explicitly download
it from the pinned [official URL](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt).
The smoke check below requires its recorded hash. Nothing is uploaded by
the packaging step. Verify transfer hashes before execution.

## Reconstruct in a new remote working directory

After transferring the two archives, extract into an empty directory:

```bash
tar -xzf /path/to/vjepa-pilot-source.tar.gz
git clone vjepa-pilot-source/repository.bundle V-JEPA-Robust-Overfitting
cp -R vjepa-pilot-source/working-tree/. V-JEPA-Robust-Overfitting/
cd V-JEPA-Robust-Overfitting
tar -xzf /path/to/vjepa-pilot-clips.tar.gz
git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
git -C external/vjepa2 checkout --detach 204698b45b3712590f06245fbfba32d3be539812
```

Put the separately transferred checkpoint at `checkpoints/vitl.pt`. Use the
manifest to verify archive hashes with `sha256sum` before extraction. The
report will correctly record the base commit with a dirty working tree;
individual source hashes identify the executed code.

## Environment check

Choose a provider image with Python 3.11 or newer and a working CUDA-enabled
PyTorch/torchvision pair. The Mac's wheel builds are not a Linux environment.
Inspect the actual instance first:

```bash
nvidia-smi
python3 --version
python3 -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())'
```

If the base image passes those checks, an isolated environment can inherit
its CUDA PyTorch packages while installing the project dependencies:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-smoke.txt
python scripts/verify_setup.py --require-cuda
python -m pytest -q tests --upstream external/vjepa2
```

The AVI decoder regression test requires FFmpeg/FFprobe; report a skip if
they are absent. This pilot reads the already prepared tensors and does not
need to re-decode videos. A broken base environment is a pilot setup failure;
do not consume the whole allowance building drivers or changing the study.
Record the provider image, exact GPU, package versions, and setup time.

## Native correctness and timing

Use fresh output names. Start with FP32 and the original Conv3d path on CUDA.
The Mac's optional linear projection has not been benchmarked on CUDA; test
any later change separately before comparing throughput.

```bash
python scripts/smoke_test.py \
  --checkpoint checkpoints/vitl.pt --device cuda --patch-embedding conv3d \
  --expected-sha256 5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012 \
  --output runs/pretrained/diagnostic/cuda-pilot-01/smoke.json

python scripts/check_attacks.py \
  --checkpoint checkpoints/vitl.pt --device cuda --patch-embedding conv3d \
  --clip data/processed/pilot-train-v1/v_ApplyEyeMakeup_g08_c04.pt \
  --output runs/pretrained/diagnostic/cuda-pilot-01/attacks.json

python scripts/pilot.py \
  --checkpoint checkpoints/vitl.pt \
  --clips data/processed/pilot-train-v1 \
  --device cuda --patch-embedding conv3d --steps 5 \
  --output runs/pretrained/diagnostic/cuda-pilot-01/pilot.json
```

Save console logs, including exceptions. Stop after an out-of-memory error,
failed correctness check, or exhaustion of the session budget. Do not
automatically shrink input shape, freeze more parameters, or weaken attacks.
On success, compare the measured clean/adversarial update and PGD-20 times
with the workload counts. The first pilot can rule out an unsuitable setup;
it cannot approve the full study while other timing rows remain unknown.

## Export and end the session

Export the JSON reports, logs, environment metadata, GPU/driver details,
provider quote, launch/termination timestamps, and actual billed cost. Verify
the exported files before terminating the instance using that provider's
controls. Disconnecting SSH, ending a terminal, or stopping Python does not
establish that billing has stopped. Confirm any separate storage charges too.

The two-hour limit is a planned operator limit, not an automatic provider
cap. Monitor elapsed billed time and leave time to export and terminate.
No full training, test evaluation, or background research job is part of
this handoff. The next human input is Lambda's live single-A6000 availability
and quote. Once an instance is ready, use its displayed SSH connection
command; no password or private key should be shared in chat.
