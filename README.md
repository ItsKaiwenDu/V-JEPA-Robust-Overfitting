# V-JEPA Robust Overfitting

**Research question:** During adversarial fine-tuning of V-JEPA 2's native masked prediction loss, does performance on unseen videos deteriorate after initially improving, even while adversarial training loss continues to decrease?

**Status: repository scaffold and draft experimental protocol.** No model has been loaded, no training or evaluation has run, and no experimental results are available. The setup checker is implemented; research entry points are planned.

This project extends our [previous robust-overfitting research](https://github.com/ItsKaiwenDu/Robust-Overfitting) to latent video prediction. It retains the familiar `models/`, `scripts/`, `notes/`, and `report/` organization, along with weekly goals, progress reports, and a Lambda Labs guide.

## Project Overview

V-JEPA predicts representations of masked video regions from visible context. We will attack that prediction objective directly, then study the generalization dynamics of adversarial fine-tuning.

The first study uses **V-JEPA 2 ViT-L/16 with a fixed target encoder**. The context encoder and predictor will be updated; the target encoder will remain frozen. This keeps the target representation stable across checkpoints while retaining the native masked latent L1 loss. Findings will concern this adaptation protocol, not the original EMA pretraining procedure.

The intended contribution is a controlled study of training dynamics and their downstream consequences. Robust overfitting is a hypothesis to test, not an assumed result.

## Start Here

| Document | Purpose |
| --- | --- |
| [Experimental protocol](docs/experimental_protocol.md) | Threat model, data split, training conditions, evaluation, and interpretation |
| [Study configuration](configs/study.yaml) | Machine-readable draft settings and unresolved provenance fields |
| [Setup guide](docs/setup.md) | Local development and university GPU preparation |
| [Lambda Labs guide](setup_lambda_labs.md) | Rental workflow and the $200 spending limit |
| [Repository structure](docs/repository_structure.md) | How the familiar folders map to this project |
| [Goals](goals.md) / [Progress](progress.md) | Planned milestones and completed work |
| [Literature audit](notes/2026-10-16_novelty-audit.md) | Prior work, overlap, and limits of the novelty search |
| [Decision log](docs/decisions.md) | Reasons behind the current research scope |

## Experiment Design

| Setting | Draft protocol |
| --- | --- |
| Model | Official V-JEPA 2 ViT-L/16 and released predictor |
| Target encoder | Fixed, clean-input, gradient-stopped |
| Dataset | UCF101 official split 1; source-group-separated validation holdout |
| Subset | 20 train / 5 validation / 5 test videos per class: 2,020 / 505 / 505 |
| Input | One deterministic 16-frame, 256 x 256 clip per video |
| Conditions | `pretrained`, `clean-finetune`, `adversarial-finetune` |
| Fine-tuning seeds | 0, 1, 2; identical data manifests across seeds |
| Training | 30 epochs; effective batch size 8; encoder and predictor updated |
| Training attack | Random-start PGD-5, L-infinity radius 4/255, step size 1/255 |
| Main evaluation | PGD-20, two restarts, expectation over four masks |
| Budget | Up to $200 total; $160 compute plus $40 overhead and contingency |

The full [protocol](docs/experimental_protocol.md) defines the controls and limitations. The [configuration](configs/study.yaml) records proposed settings; it is not yet a runnable training configuration or a frozen preregistration.

## Setup Guide

From the repository root, with Python 3.11 available:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_setup.py
```

The current dependency file installs only what the scaffold checker needs. PyTorch and Meta's training dependencies will be installed and locked during the hardware pilot. See [setup](docs/setup.md) for that boundary.

Optional checks after installing a suitable PyTorch environment:

```bash
python scripts/verify_setup.py --check-torch
python scripts/verify_setup.py --require-cuda
```

These commands inspect the environment without downloading models, preparing data, launching jobs, or renting GPUs. A passing scaffold check does not establish training readiness.

### Entry Point Status

| Script | Status |
| --- | --- |
| [`scripts/verify_setup.py`](scripts/verify_setup.py) | Implemented: scaffold, YAML, and optional device checks |
| `scripts/prepare_data.py` | Planned: grouped manifests and deterministic clips |
| `scripts/train.py` | Planned: clean and adversarial fine-tuning |
| `scripts/evaluate.py` | Planned: checkpoint and downstream evaluation |
| `scripts/run_report_checks.py` | Planned: attack and implementation diagnostics |
| `scripts/plot_results.py` | Planned: per-seed curves and aggregate summaries |

## Project Directory Structure

```text
Research/
├── .github/                 # Repository checks and contribution templates
├── assets/                  # Small README and presentation assets
├── configs/                 # Study settings and local path example
├── data/                    # Local videos/features; tracked split manifests
│   ├── manifests/
│   ├── raw/                 # Ignored
│   ├── processed/           # Ignored
│   └── features/            # Ignored
├── docs/                    # Protocol, setup, decisions, reproducibility
├── external/                # Ignored upstream checkout; instructions tracked
├── models/                  # Future V-JEPA adapter and target/loss utilities
├── notes/                   # Literature notes and dated novelty audits
├── report/                  # LaTeX source and curated results
│   ├── main.tex
│   ├── references.bib
│   ├── pretrained/
│   ├── clean-finetune/
│   ├── adversarial-finetune/
│   ├── checks/
│   ├── figures/
│   └── tables/
├── scripts/                 # Familiar command-line entry points
├── tests/                   # Planned scientific acceptance tests
├── checkpoints/             # Ignored model weights
├── runs/                    # Ignored raw metrics and run metadata
├── run_logs/                # Ignored console/job logs
├── AGENTS.md                # Project guidance for coding assistants
├── goals.md
├── progress.md
├── setup_lambda_labs.md
├── requirements.txt
└── LICENSE
```

Future artifacts retain the familiar condition/seed layout:

```text
checkpoints/<condition>/[diagnostic/]<run-name>/
runs/<condition>/[diagnostic/]<run-name>/
report/<condition>/[diagnostic/]<run-name>/
report/<condition>/overall/
```

Use `seed-0`, `seed-1`, and `seed-2` for the first locked experiment. Repeated runs need distinct names, such as `seed-0-retry-01`; never overwrite a previous run. The pretrained reference uses `pretrained/reference/` because it has no fine-tuning seed. See [artifact conventions](docs/reproducibility.md).

## Results

No results yet. Planned outputs include clean/adversarial loss trajectories, best-versus-final gaps, attack-strength checks, and linear-probe accuracy. Report observed overfitting, no observed effect, and inconclusive experiments distinctly.

## Weekly Goals & Progress

The target is a focused report by day 21, with days 22-27 reserved for corrections and review. Runtime and cost must pass the pilot before the full experiment begins. Follow [goals.md](goals.md) and record completed work in [progress.md](progress.md).

## Researcher & License

Researcher: **Kaiwen Du**. This is a separate project from the previous study; collaborators and acknowledgments will be added when confirmed.

Original repository code uses the [MIT License](LICENSE), matching the previous project. Upstream code, model weights, and datasets retain their respective licenses and terms; they are not relicensed by this repository.

## References

- [Assran et al. — V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [Rice, Wong, and Kolter — Overfitting in adversarially robust deep learning](https://proceedings.mlr.press/v119/rice20a.html)
- [Robust CLIP](https://arxiv.org/abs/2402.12336) and [ASTrA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17b08a9de93e2accf13429643e7eafdc-Abstract-Conference.html)
- [Literature map and novelty limits](notes/2026-10-16_novelty-audit.md)
