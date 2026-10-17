# Development and GPU setup

## What works now

Only the repository verifier is implemented. It needs Python 3.11 or newer
and PyYAML. Python 3.11 is the initial target for the research environment;
actual CUDA and upstream compatibility must be checked during the pilot.

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
inspects an installed PyTorch runtime; `--require-cuda` additionally fails
when CUDA is unavailable. Neither option installs packages or loads weights.

## Local MacBook

Use the MacBook for documentation, source development, metadata inspection,
and analysis. MPS availability and unified RAM are not a validated substitute
for the CUDA training environment. Test any future MPS model path separately.

## University RTX 3080

Use the university's established SSH and scheduling instructions. Inspect
the assigned GPU and its actual VRAM rather than assuming the 10 GB or 12 GB
variant. Do not run training on a login node if the university provides a
separate GPU scheduler.

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

The next implementation step is the model/loss smoke test. A CUDA availability
check alone does not test forward passes, backpropagation, masking, decoding,
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

The local repository uses branch `main`. Before the first commit, inspect:

```bash
git status --short
git diff --check
```

Create and connect a GitHub remote when the repository name and visibility
are chosen. No remote URL is assumed in this scaffold. After committing
and connecting the remote, publish with the normal Git push workflow.

GitHub Actions checks scaffold integrity and Python syntax. It does not run
scientific tests, use a GPU, fetch data, or claim the research is validated.
