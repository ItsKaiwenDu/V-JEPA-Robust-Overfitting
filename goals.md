# Weekly Goals & Objectives

Track planned work here and completed work in [progress.md](progress.md).
Day 1 begins with the first implementation/pilot work session. The target is
a focused report by day 21, with days 22-27 reserved for review and repairs.

## Current planning update — October 22

The [staged proposal](docs/budget_study_plan.md) remains a candidate within the
$200 limit; it is not admitted for study execution. The larger milestones
below remain context and are not all included in the narrowed scope.
Training/resume, native evaluation/selection, fixed probes, calibration records,
and epoch orchestration are implemented with local tests. Staged manifests and
120 training/exploration clips are prepared; confirmation/test pixels remain closed.

Full-model CUDA recovery failed under the original runtime, and strict execution
was too slow. The [TF32-off university component control](report/checks/2026-10-21-university-cuda-convolution-precision.md)
now qualifies a numerically checked, reproducible native-Conv3d candidate.
Opt-in integration now binds explicit runtime/cache/resume identities;
small-model gradients/recovery and fresh CPU workers are checked. The verified
university derivative control isolates failed forward-oracle gates; fixed-
cotangent backwards pass and saved gradients reconstruct bitwise. The shared-forward precision control now passes verified university forward/
gradient/replay gates with faster native input-only backward. Explicit integration now passes local clean/adversarial AdamW exact recovery,
including fresh processes and scheduled evaluation. No study backend is adopted.
Next is a fresh bounded sufficient-VRAM CUDA handoff, full pretrained recovery
and representative checks, then measured costs before an exploratory pair. No research trajectory is complete; confirmation is conditional.

## Repository setup

- [x] Inspect the previous robust-overfitting repository and retain familiar conventions.
- [x] Initialize the local `main` branch and create the research scaffold.
- [x] Record the agreed scope, draft settings, literature map, and budget.
- [x] Connect the GitHub remote for `ItsKaiwenDu/V-JEPA-Robust-Overfitting`.

## Week 1 — Feasibility and first trajectory

**Objective:** Establish a correct native-loss pipeline and start fine-tuning early.

- [x] Identify the university allocation (RTX 3070, 8 GB); record its driver mismatch.
- [x] Verify Mac GPU access, pretrained CPU/MPS agreement, and bounded attack checks.
- [x] Measure Mac batch-eight update time, primary attack time, and sampled memory.
- [x] Draft an evaluation schedule and reproducible partial workload/price calculation.
- [x] Complete university and RTX 5090 pilots and retain their limitations.
- [x] Draft a staged budget proposal with separate exploration/confirmation panels.
- [ ] Admit a measured monitoring schedule and backend within the budget.
- [x] Implement the FP32 training/resume core and atomic checkpoint retention; verify interrupted CPU trajectories.
- [x] Export staged identities and verify training/exploration preparation; leave confirmation/test pixels unopened.
- [x] Implement and test native evaluation recovery, saved anchors, and provisional primary selection.
- [x] Add fixed probes, calibration records, and epoch orchestration; validate local fixtures.
- [x] Isolate strict CUDA component costs and qualify a native Conv3d candidate on the university GPU.
- [x] Integrate the candidate as an explicit diagnostic option; verify small-model gradients/recovery.
- [x] Implement and locally test the derivative attribution control and prepare its bounded university handoff.
- [x] Execute/verify pretrained derivative attribution without changing the prior numerical rejection.
- [x] Implement a separately identified shared FP64 patch forward and verify trained-projection/CPU replay gates.
- [x] Verify shared-forward pretrained CUDA accuracy/input gradients/attacks/replay on the bounded one-clip control.
- [x] Integrate the explicit precision identity into research/cache/checkpoint contracts and verify CPU optimizer recovery.
- [ ] Accept full pretrained CUDA input gradients/attacks and optimizer recovery for the candidate.
- [ ] Validate full-model GPU resume and integrated evaluation before research execution.
- [x] Pin upstream source and implement the adapter and local CPU correctness tests.
- [x] Audit full ViT-L tensor names/shapes against official checkpoint metadata.
- [x] Pin and load the official V-JEPA 2 model and predictor with verified weights on CPU.
- [x] Prepare grouped manifests, eight deterministic training clips, and real-clip attack sanity checks.
- [ ] Prepare group-separated UCF101 manifests and deterministic clips.
- [x] Verify clean loss, target freezing, masks, input gradients, and pixel projection on the prepared fixtures/clips.
- [x] Implement native-loss PGD/FGSM with meaningful diagnostics.
- [ ] Add embedding-distance baselines and complete stronger-attack/convergence diagnostics.
- [ ] Measure training, evaluation, and storage cost; apply the feasibility gate.
- [ ] Resolve pilot decisions and freeze the confirmatory protocol.
- [ ] Launch the first complete clean/adversarial training trajectories.

**Expectations:** Correctness evidence and an affordable measured workload.

**Deliverables:** Pilot logs, provenance, frozen manifests/config, cost estimate,
and the first training curves. A failed gate is documented, not hidden.

## Week 2 — Repeated experiments

**Objective:** Complete comparable training trajectories across three seeds.

- [ ] Run clean and adversarial fine-tuning for seeds 0, 1, and 2.
- [ ] Track clean and adversarial training/validation loss every epoch.
- [ ] Save weights and resume state, maintaining the fixed target encoder.
- [ ] Inspect collapse indicators and record all failed or interrupted runs.
- [ ] Select checkpoints using the frozen validation criterion.
- [ ] Reconcile actual compute spending with the remaining evaluation budget.

**Expectations:** Repeated, auditable curves rather than a single favorable seed.

**Deliverables:** Six run records, checkpoint selections, and preliminary
per-seed summaries with no final-test-driven changes.

## Week 3 — Evaluation and report

**Objective:** Evaluate robustness credibly and explain what the results support.

- [ ] Complete attack-strength, restart, epsilon, and independent-mask checks.
- [ ] Compare native-loss attacks with random noise, FGSM, and embedding attacks.
- [ ] Fit matched linear probes and evaluate direct classifier attacks.
- [ ] Open final test evaluation after choices are locked.
- [ ] Summarize best-versus-final gaps, uncertainty, and per-seed behavior.
- [ ] Repeat the novelty audit and halt if a matching study is found.
- [ ] Write the research report with explicit limitations and negative findings.

**Expectations:** A defensible answer: observed, not observed in this regime,
or inconclusive for a stated reason.

**Deliverables:** Curated CSVs/figures, report source, reproducibility records,
and a total cost account. Target completion: day 21.

## Days 22-27 — Review buffer

Resolve material implementation or interpretation problems, perform justified
reruns within the remaining budget, and seek scientific review. Broader model,
dataset, EMA, and robotics studies remain follow-up work.
