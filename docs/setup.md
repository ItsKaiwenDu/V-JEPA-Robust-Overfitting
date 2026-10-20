# Development and GPU setup

## What works now

The repository verifier needs Python 3.11 or newer and PyYAML. The V-JEPA
adapter, CPU fixture tests, and full-checkpoint smoke-test command are also
implemented; use the separate [model-check guide](smoke_test.md) for those
dependencies. Python 3.11 is the initial target for the research environment;
actual CUDA compatibility must be checked during the pilot.

Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_setup.py
```

If `python3.11` is unavailable, use an existing compatible interpreter for
scaffold work and record its version. Do not assume a newer local Python
version supports the eventual GPU dependency stack.

The checker resolves the repository from its own location, so it also works
when invoked from another working directory. Optional `--check-torch`
inspects an installed PyTorch runtime; `--require-cuda` or `--require-mps`
additionally requires the selected backend. These checks do not install
packages or load weights.

## Local MacBook

The M4 Max with 36 GiB unified memory can run the pretrained model on MPS.
GPU access works outside the restricted execution environment. Follow the
[Mac pilot guide](mac_pilot.md) for CPU/MPS correctness comparison and
device-specific timing; shared memory is not equivalent to dedicated VRAM.

## University RTX 3070

The user's output identifies an RTX 3070 on `linux10604`, with standard 8 GB
GPU memory. `nvidia-smi` fails with a driver/library mismatch; university IT
must resolve that before CUDA testing. Python 3.9.25 also needs a separate
3.11-compatible environment. Use established SSH and scheduling instructions,
including an allocated GPU node when a scheduler is provided.

On the assigned GPU machine, inspect:

```bash
python3 --version
nvidia-smi
```

Install a PyTorch build compatible with the machine's driver using the
[official installation selector](https://pytorch.org/get-started/locally/).
Resolve Meta's dependencies against the pinned upstream revision. Keep that
environment separate from the small scaffold environment, and record it
with the pilot results. No CUDA package version has been validated here yet.

Then run:

```bash
python scripts/verify_setup.py --require-cuda
```

The full-checkpoint model/loss smoke test and small local CPU checks have
passed. The [CUDA pilot guide](gpu_pilot.md) describes the next human action.
A CUDA availability check alone does not test forward passes, backpropagation, masking, decoding,
or checkpoint compatibility.

## Upstream model and dataset

Follow [external/README.md](../external/README.md) for the upstream source
location and provenance. Store downloaded weights under `checkpoints/` and
videos under `data/raw/`; both are ignored. Follow [data conventions](../data/README.md)
before preparing clips or features.

No automated downloader exists yet. Inspect checkpoint contents safely,
verify the official source, and record the hash before loading weights. Do
not copy the old CIFAR-10 automatic-download behavior into the UCF101 setup.

## Git and GitHub

The local repository uses branch `main` and is connected to
`https://github.com/ItsKaiwenDu/V-JEPA-Robust-Overfitting.git`. Before committing, inspect:

```bash
git status --short
git diff --check
```

Review both new files and the diff before staging. Once staged, also run
`git diff --cached --check`. Publish with the normal Git push workflow when
ready; implementing a change does not automatically commit or push it.

GitHub Actions checks scaffold integrity and Python syntax. It does not run
scientific tests, use a GPU, fetch data, or claim the research is validated.
