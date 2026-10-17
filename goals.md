# Weekly Goals & Objectives

Track planned work here and completed work in [progress.md](progress.md).
Day 1 begins with the first implementation/pilot work session. The target is
a focused report by day 21, with days 22-27 reserved for review and repairs.

## Repository setup

- [x] Inspect the previous robust-overfitting repository and retain familiar conventions.
- [x] Initialize the local `main` branch and create the research scaffold.
- [x] Record the agreed scope, draft settings, literature map, and budget.
- [x] Connect the GitHub remote for `ItsKaiwenDu/V-JEPA-Robust-Overfitting`.

## Week 1 — Feasibility and first trajectory

**Objective:** Establish a correct native-loss pipeline and start fine-tuning early.

- [ ] Confirm university GPU access and inspect actual VRAM.
- [x] Pin upstream source and implement the adapter and local CPU correctness tests.
- [x] Audit full ViT-L tensor names/shapes against official checkpoint metadata.
- [x] Pin and load the official V-JEPA 2 model and predictor with verified weights on CPU.
- [x] Prepare grouped manifests, eight deterministic training clips, and real-clip attack sanity checks.
- [ ] Prepare group-separated UCF101 manifests and deterministic clips.
- [ ] Verify clean loss, target freezing, masks, input gradients, and pixel projection.
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
