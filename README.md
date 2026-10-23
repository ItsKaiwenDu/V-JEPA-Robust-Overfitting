# V-JEPA Robust Overfitting

**Research question:** During adversarial fine-tuning of V-JEPA 2's native masked prediction loss, does performance on unseen videos deteriorate after initially improving, even while adversarial training loss continues to decrease?

**Status: bounded shared-forward CUDA recovery passes; the measured study projection exceeds the $200 cap.** The [new budget review](report/checks/2026-10-22-measured-budget-review.md) puts the existing 20-class staged design at about $304–$307 with the retained allowances and reserves. The ten-class alternative is about $219–$221. These are incomplete projections; neither is admitted. The [October 20 proposal](docs/budget_study_plan.md) and its old $193.91 envelope remain historical. Hardware feasibility checks passed on CPU, Apple GPU, and RTX 5090; university RTX 3070 FP32 training ran out of memory. Staged manifests and deterministic preparation of 120 training/exploration clips are verified.

The [October 21 CUDA workflow report](report/checks/2026-10-21-cuda-workflow.md) records the failed exact full-model recovery comparison and a strict deterministic retry stopped after an attack took 16× longer. The repaired suite passes 225 tests on the CUDA host, including a small independent-process GPU replay test; that original-runtime attempt remains rejected. Both historical evidence exports are verified locally.

The [university component profile](report/checks/2026-10-21-university-cuda-components.md) completed all 48 workers: strict replay passes all 12 cases, while default attention gradients fail exact replay. Original Conv3d input backward increased from 1.102 ms to 3.400 s under strict settings on RTX 3070. This isolates a bottleneck on that host; the next gate is a validated execution option for the same Conv3d, followed by RTX 5090/full-model confirmation.

The [first convolution comparison](report/checks/2026-10-21-university-cuda-convolution.md) rejected all 12 workers before timing: the TF32-permitted cuDNN reference fails the fixed FP64 gate. Native matches FP64 but differs from that reference; no option qualified. The subsequent [TF32-off control](report/checks/2026-10-21-university-cuda-convolution-precision.md) passes all 12 university workers and observed replay, with unchanged tolerances. Native execution reduces isolated forward/backward from 3.410 s to 5.897 ms for input gradients and 3.431 s to 7.043 ms for parameter gradients. This qualifies a component candidate; whole-model speed/recovery and study adoption remain pending. The [opt-in native integration](report/checks/2026-10-21-native-integration.md) now binds backend/precision/cache/recovery identities and passes small-model gradients, PGD, exact recovery and an extracted four-process CPU fixture. The first university attempt stopped at a small probe with an incorrect kernel-name check. The [corrected CUDA comparison](report/checks/2026-10-21-university-native-model-comparison.md) verifies native kernel dispatch and exact within-backend replay; clean loss agrees, while input gradients reject the pointwise gate (0.3583% relative L2, 242,613 failed derivatives). The [verified derivative control](report/checks/2026-10-21-university-gradient-attribution.md) completes in 66.915s: all fixed-cotangent backward checks and bitwise saved-gradient reconstructions pass, while both FP32 forward paths reject the FP64 output gate (12 cuDNN / 35 native values). Forward-token differences change incoming tail derivatives; the backend remains unadopted. The [verified shared-forward CUDA control](report/checks/2026-10-21-university-shared-forward.md) passes forward/gradient oracle gates and exact within-backend replay on one training clip; FGSM/PGD-5 outputs also match between backwards. Native PGD-5+score takes 6.324s versus 40.745s with cuDNN (median of two single attempts). The [explicit training integration](docs/shared_forward_training.md) now passes CPU clean/adversarial AdamW recovery, including fresh processes and scheduled evaluation; 434 local tests pass with two CUDA-only skips. The [October 22 pretrained CUDA workflow](report/checks/2026-10-22-shared-forward-cuda-workflow.md) now completes all eleven phases: 436 CUDA-host tests pass; three representative inputs pass independently verified numerical/attack checks; clean and adversarial two-update restart comparisons match exactly. Feature/probe and PGD-50/FGSM/random controls complete on training data. The user confirmed Pod termination after verified export. The new billing increment is provisionally $1.085 from a combined summary, making provisional cumulative spending $2.280. Study calibration, precision/protocol freezing, measured budget admission and research results remain pending.

This project extends our [previous robust-overfitting research](https://github.com/ItsKaiwenDu/Robust-Overfitting) to latent video prediction. It retains the familiar `models/`, `scripts/`, `notes/`, and `report/` organization, along with weekly goals, progress reports, and a Lambda Labs guide.

## Project Overview

V-JEPA predicts representations of masked video regions from visible context. We will attack that prediction objective directly, then study the generalization dynamics of adversarial fine-tuning.

The first study uses **V-JEPA 2 ViT-L/16 with a fixed target encoder**. The context encoder and predictor will be updated; the target encoder will remain frozen. This keeps the target representation stable across checkpoints while retaining the native masked latent L1 loss. Findings will concern this adaptation protocol, not the original EMA pretraining procedure.

The intended contribution is a controlled study of training dynamics and their downstream consequences. Robust overfitting is a hypothesis to test, not an assumed result.

## Start Here

| Document | Purpose |
| --- | --- |
| [Lower-cost GPU benchmark](docs/lower_cost_gpu_benchmark.md) | Retained 20-class design, verified offline screen and user-reported $0.29/hour A5000 quote; manual deployment pending |
| [Measured budget review](report/checks/2026-10-22-measured-budget-review.md) | Current unit-cost projections, remaining unknowns and required price/speed improvements |
| [Latest staged study proposal](docs/budget_study_plan.md) | Scoped research question, separate validation panels, staged costs, and decision gates |
| [Experimental protocol](docs/experimental_protocol.md) | Threat model, data split, training conditions, evaluation, and interpretation |
| [Study configuration](configs/study.yaml) | Machine-readable draft settings and unresolved provenance fields |
| [Setup guide](docs/setup.md) | Local development and university GPU preparation |
| [Local model checks](docs/smoke_test.md) | Pretrained checkpoint and native-loss correctness checks |
| [Staged data and evaluation](docs/staged_data_and_evaluation.md) | Verified exploration clips, native attacks, evaluation recovery, and provisional primary selection |
| [Feature probes and scheduled recovery](docs/workflow_diagnostics.md) | Train-only probe/calibration rules, epoch evaluation barriers, and bounded integration commands |
| [Latest shared-forward CUDA outcome](report/checks/2026-10-22-shared-forward-cuda-workflow.md) | Exact bounded clean/adversarial recovery, representative checks, controls and measured costs |
| [CUDA workflow pilot outcome](report/checks/2026-10-21-cuda-workflow.md) | Replay failure, deterministic runtime cost, verified exports, and the next profiling gate |
| [University CUDA component outcome](report/checks/2026-10-21-university-cuda-components.md) | Verified operator timings, strict Conv3d bottleneck, attention replay, and next check |
| [Convolution comparison outcome](report/checks/2026-10-21-university-cuda-convolution.md) | Verified numerical rejection, TF32 precision hypothesis, and unavailable timing/replay |
| [Verified TF32-off outcome](report/checks/2026-10-21-university-cuda-convolution-precision.md) | Native component accuracy/replay and timing; full-model acceptance remains pending |
| [Opt-in native integration](docs/native_execution.md) | Explicit backend/precision identities, first-order limits, and the next university full-model input/attack check |
| [Strict convolution precision control](docs/cuda_convolution_comparison.md) | Explicit TF32-off follow-up, unchanged numerical gates, and detached launch |
| [CUDA component profiler](docs/cuda_component_profile.md) | Bounded synthetic operator timing, exact gradient replay, and detached university launch |
| [CUDA workflow guide](docs/cuda_workflow_pilot.md) | Offline handoff, explicit deterministic policy, and bounded execution instructions |
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
| [`scripts/estimate_measured_study.py`](scripts/estimate_measured_study.py) | Offline shared-forward timing projection; rejects changed inputs, preserves unknowns and never admits a run |
| [`scripts/estimate_staged_study.py`](scripts/estimate_staged_study.py) | Implemented: staged cost comparison, source-hash checks, and metadata-only subset audit; never approves a run |
| [`scripts/train.py`](scripts/train.py) | Bounded training/resume; exact two-update CUDA recovery passes under the explicit shared-forward policy; research admission pending |
| [`scripts/export_staged_manifests.py`](scripts/export_staged_manifests.py) | Implemented: exact replay and export of the proposed subset identities |
| [`scripts/prepare_staged_clips.py`](scripts/prepare_staged_clips.py) | Passed: 100 training and 20 exploration clips decoded twice; confirmation/test pixels closed |
| [`scripts/evaluate.py`](scripts/evaluate.py) | Bounded native evaluation, atomic recovery and saved-anchor loading pass in the shared-forward pretrained CUDA pilot |
| [`scripts/check_workflow.py`](scripts/check_workflow.py) | Bounded feature/probe, calibration and scheduled recovery pass on pretrained CUDA with the explicit shared-forward policy |
| [`scripts/check_native_model.py`](scripts/check_native_model.py) | Original FP32 full-model numerical comparison rejected; the separate shared-forward candidate passes its bounded CUDA checks |
| [`scripts/profile_cuda.py`](scripts/profile_cuda.py) | Passed: all 48 university CUDA workers; strict replay on 12 cases; isolated timing evidence |
| [`scripts/university_profile.sh`](scripts/university_profile.sh) | Implemented: checksummed source handoff and detached, externally bounded university launcher |
| [`scripts/compare_cuda_convolution.py`](scripts/compare_cuda_convolution.py) | Passed: 12 TF32-off university workers and component replay; native timing qualifies a full-model candidate |
| [`scripts/university_convolution.sh`](scripts/university_convolution.sh) | Prepared: checksummed detached university comparison, no model/data downloads |
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
