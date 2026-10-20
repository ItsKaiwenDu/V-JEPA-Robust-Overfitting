# V-JEPA Robust Overfitting

**Research question:** During adversarial fine-tuning of V-JEPA 2's native masked prediction loss, does performance on unseen videos deteriorate after initially improving, even while adversarial training loss continues to decrease?

**Status: hardware pilots passed; a smaller staged study is proposed, with full training still pending.** The original study exceeds the $200 cap. The [October 20 proposal](docs/budget_study_plan.md) starts with one exploratory clean/adversarial pair on 20 classes, followed conditionally by three new confirmation pairs. Its planning envelope is $193.91 including headroom and reserves, with some costs still unmeasured; it is not admitted for execution. Yesterday's Runpod session cost $0.592 according to the user. CPU, Apple GPU, and RTX 5090 diagnostics passed; university RTX 3070 FP32 training ran out of memory. The local training/resume core is now implemented with 95 tests passing; full research execution and new GPU resume validation remain pending. None of these checks establishes robust overfitting.

This project extends our [previous robust-overfitting research](https://github.com/ItsKaiwenDu/Robust-Overfitting) to latent video prediction. It retains the familiar `models/`, `scripts/`, `notes/`, and `report/` organization, along with weekly goals, progress reports, and a Lambda Labs guide.

## Project Overview

V-JEPA predicts representations of masked video regions from visible context. We will attack that prediction objective directly, then study the generalization dynamics of adversarial fine-tuning.

The first study uses **V-JEPA 2 ViT-L/16 with a fixed target encoder**. The context encoder and predictor will be updated; the target encoder will remain frozen. This keeps the target representation stable across checkpoints while retaining the native masked latent L1 loss. Findings will concern this adaptation protocol, not the original EMA pretraining procedure.

The intended contribution is a controlled study of training dynamics and their downstream consequences. Robust overfitting is a hypothesis to test, not an assumed result.

## Start Here

| Document | Purpose |
| --- | --- |
| [Latest staged study proposal](docs/budget_study_plan.md) | Scoped research question, separate validation panels, staged costs, and decision gates |
| [Experimental protocol](docs/experimental_protocol.md) | Threat model, data split, training conditions, evaluation, and interpretation |
| [Study configuration](configs/study.yaml) | Machine-readable draft settings and unresolved provenance fields |
| [Setup guide](docs/setup.md) | Local development and university GPU preparation |
| [Local model checks](docs/smoke_test.md) | Pretrained checkpoint and native-loss correctness checks |
| [Training and recovery](docs/training_runner.md) | Verified training inputs, optimizer/resume state, atomic checkpoints, and current limits |
| [Data and attacks](docs/data_and_attacks.md) | Grouped subsets, reproducible clips, and pixel-space attacks |
| [Mac GPU pilot](docs/mac_pilot.md) | MPS access, correctness checks, and explicit performance options |
| [Measured Mac feasibility](report/checks/2026-10-19-mac-feasibility.md) | Update times, memory, evaluation cost, and limits of a four-week schedule |
| [RTX 5090 feasibility results](report/checks/2026-10-19-runpod-5090-feasibility.md) | CUDA timings, university memory limit, verified export, and budget implications |
| [Evaluation and compute proposal](docs/evaluation_compute_plan.md) | Proposed checkpoints, complete workload ledger, price checks, and unresolved timing costs |
| [CUDA pilot handoff](docs/gpu_pilot.md) | University workstation status and CUDA feasibility commands |
| [Lambda Labs guide](setup_lambda_labs.md) | Rental workflow and the $200 spending limit |
| [Repository structure](docs/repository_structure.md) | How the familiar folders map to this project |
| [Goals](goals.md) / [Progress](progress.md) | Planned milestones and completed work |
| [Literature audit](notes/2026-10-16_novelty-audit.md) | Prior work, overlap, and limits of the novelty search |
| [Decision log](docs/decisions.md) | Reasons behind the current research scope |

## Experiment Design

The table below preserves the **original larger draft**, which is not admitted within the budget. The [latest proposal](docs/budget_study_plan.md) explicitly changes the data, seeds, evaluation schedule, and claim scope; it has not replaced or frozen `configs/study.yaml`.

| Setting | Original draft protocol |
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

The root dependency file installs only what the scaffold checker needs. For the implemented model integration and CPU tests, follow [local model checks](docs/smoke_test.md) and use `requirements-smoke.txt`. The RTX 5090 pilot passed with PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128; the feasibility report records the tested environment.

Optional checks after installing a suitable PyTorch environment:

```bash
python scripts/verify_setup.py --check-torch
python scripts/verify_setup.py --require-cuda
python scripts/verify_setup.py --require-mps
```

These commands inspect the environment without downloading models, preparing data, launching jobs, or renting GPUs. A passing scaffold check does not establish training readiness.

### Entry Point Status

| Script | Status |
| --- | --- |
| [`scripts/verify_setup.py`](scripts/verify_setup.py) | Implemented: scaffold, YAML, and optional device checks |
| [`scripts/smoke_test.py`](scripts/smoke_test.py) | Passed on the official pretrained checkpoint using CPU, MPS, and CUDA |
| [`scripts/prepare_data.py`](scripts/prepare_data.py) | Implemented: source-group-separated manifests from official split lists |
| [`scripts/prepare_clips.py`](scripts/prepare_clips.py) | Implemented: deterministic training/validation clips; eight pilot clips validated |
| [`scripts/check_attacks.py`](scripts/check_attacks.py) | Real-clip FGSM/PGD sanity checks passed on CPU, MPS, and CUDA |
| [`scripts/check_mps.py`](scripts/check_mps.py) | CPU/MPS comparison of losses, targets, and gradients on one real clip |
| [`scripts/check_patch_embedding.py`](scripts/check_patch_embedding.py) | Isolated correctness/timing comparison of equivalent patch projections |
| [`scripts/pilot.py`](scripts/pilot.py) | FP32 batch-eight MPS and RTX 5090 updates and full attack timing passed; RTX 3070 training ran out of memory |
| [`scripts/estimate_compute.py`](scripts/estimate_compute.py) | Reproduces partial planning projections; missing costs stay unknown and cannot approve a run |
| [`scripts/estimate_staged_study.py`](scripts/estimate_staged_study.py) | Implemented: staged cost comparison, source-hash checks, and metadata-only subset audit; never approves a run |
| [`scripts/train.py`](scripts/train.py) | Implemented: bounded FP32 training/resume diagnostic; CPU recovery tests pass; research evaluation and admission pending |
| `scripts/evaluate.py` | Planned: checkpoint and downstream evaluation |
| `scripts/run_report_checks.py` | Planned: attack and implementation diagnostics |
| `scripts/plot_results.py` | Planned: per-seed curves and aggregate summaries |

## Project Directory Structure

```text
V-JEPA-Robust-Overfitting/
├── .github/                 # Repository checks and contribution templates
├── assets/                  # Small README and presentation assets
├── attacks/                 # Bounded FGSM/PGD and fixed-mask/EOT objectives
├── configs/                 # Study settings and local path example
├── data/                    # Local videos/features; tracked split manifests
│   ├── manifests/
│   ├── raw/                 # Ignored
│   ├── processed/           # Ignored
│   └── features/            # Ignored
├── docs/                    # Protocol, setup, decisions, reproducibility
├── datasets/                # Official split parser and deterministic decoder
├── external/                # Ignored upstream checkout; instructions tracked
├── models/                  # V-JEPA adapter, frozen targets, loss, diagnostics
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
├── tests/                   # Small CPU model checks; more checks planned
├── checkpoints/             # Ignored model weights
├── runs/                    # Ignored raw metrics and run metadata
├── run_logs/                # Ignored console/job logs
├── AGENTS.md                # Project guidance for coding assistants
├── goals.md
├── progress.md
├── setup_lambda_labs.md
├── requirements.txt
├── requirements-smoke.txt
└── LICENSE
```

Future artifacts retain the familiar condition/seed layout:

```text
checkpoints/<condition>/[diagnostic/]<run-name>/
runs/<condition>/[diagnostic/]<run-name>/
report/<condition>/[diagnostic/]<run-name>/
report/<condition>/overall/
```

Use the seeds from the eventual frozen protocol. The original draft uses `seed-0`, `seed-1`, and `seed-2`; the staged proposal uses exploratory seed 0 and separate confirmation seeds 11, 22, and 33. Repeated runs need distinct names, such as `seed-0-retry-01`; never overwrite a previous run. The pretrained reference uses `pretrained/reference/` because it has no fine-tuning seed. See [artifact conventions](docs/reproducibility.md).

## Results

No research results yet. [Correctness diagnostics](report/checks/2026-10-17-pretrained-smoke.json), [one training-clip attack check](report/checks/2026-10-17-real-clip-attacks.json), and [Mac feasibility measurements](report/checks/2026-10-19-mac-feasibility.md) are available. Planned research outputs include clean/adversarial loss trajectories, best-versus-final gaps, attack-strength checks, and linear-probe accuracy. Report observed overfitting, no observed effect, and inconclusive experiments distinctly.

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
