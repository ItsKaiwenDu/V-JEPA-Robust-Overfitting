# Proposed staged study within the $200 cap

**Measured budget outcome — October 22:** the
[new review](../report/checks/2026-10-22-measured-budget-review.md) prices the
existing 20-class design at $304–$307 and the existing ten-class alternative
at $219–$221, with the historical overhead allowances/reserves and further
unknowns. Both exceed the caps. The historical estimate below is superseded
for the explicit shared-forward candidate; no scope or YAML was changed.

**October 22 measurement follow-up:** the [bounded CUDA workflow](../report/checks/2026-10-22-shared-forward-cuda-workflow.md)
passed. Its measured update/evaluation costs must replace the proxies below
before admission. The $193.91 figure is a historical planning estimate,
not a current verified cost. The original YAML and design are unchanged.

October 20, 2026. **Proposal, not an admitted experiment or a research result.**
Machine-readable settings: [budget_study_plan.yaml](../configs/budget_study_plan.yaml).
Reproducible costs and proposed video identities:
[projection and metadata audit](../report/checks/2026-10-20-staged-study-projection.json).

Recommend a **20-class, 100-training-video native-loss study**: first one
clean/adversarial pair, then three new matched seed pairs only if the first
trajectory and measured remaining costs justify proceeding. Preserve the
model, video shape, fixed clean target, and primary attack strength. Narrow
the dataset and claims explicitly. The original [protocol 0.3](experimental_protocol.md)
and its hashed configuration remain unchanged as the larger draft design.

The planning envelope is **$193.91**, including a $10 total pilot reserve,
$40 noncompute reserve, 25% runtime headroom, and eight hours allowed for
essential work that has not been timed. This is close to the cap. Allowances
are not measurements: **the plan cannot yet authorize a rental or full run**.

## What this would establish

The question is whether held-out **adversarial native prediction loss**
initially improves and later worsens while training-side robust loss keeps
improving in this small-data, fixed-target fine-tuning regime. A useful output
is a reproducible trajectory, attack-validation evidence, and a documented
answer within this regime, including an inconclusive or negative result.

Supervised adversarial training provides the motivation for investigating
late deterioration and early stopping; it does not establish that JEPA has
the same behavior. See [Rice, Wong, and Kolter (2020)](https://proceedings.mlr.press/v119/rice20a.html).
Fresh adaptive attacks and convergence checks matter because weak gradients
can mislead robustness evaluation; see [Athalye, Carlini, and Wagner (2018)](https://arxiv.org/abs/1802.00420).

This is an exploratory research contribution, not yet evidence of a broad
foundation-model phenomenon or a conference-ready result. It cannot establish
robustness of action classification, EMA pretraining, robotics, or V-JEPA 2.1.
Embedding attacks, epsilon sweeps, and direct classifier adversaries are
deferred extensions. Clean probes and feature diagnostics remain required
to help identify collapse or forgetting, but do not prove semantic robustness.

## Data proposal and leakage controls

Select classes by SHA-256 ranking of identifiers using seed 20261020, before
observing trajectories. Select videos by ranked source groups, taking one
video per group before taking additional videos from a group. The metadata
selection is implemented in [staged_subset.py](../datasets/staged_subset.py);
the report contains the exact proposed rows and their parent file hashes.

| Panel | Videos | Source and purpose |
| --- | ---: | --- |
| Actual training | 100 | Five videos from five distinct source groups per class |
| Training monitor | 20 | One fixed member of actual training per class |
| Exploration validation | 20 | One group per class withheld from the parent training pool |
| Confirmation validation | 20 | One video per class from the original validation partition |
| Final test | 60 | Three videos per class from the original test partition |

The existing validation partition has only one or two groups per class.
Splitting its clips into two panels would not reliably create independent
groups. Instead, withhold the entire exploration group before selecting
actual training videos. None of its unselected videos may return to training.
Groups already used for the eight diagnostic pilot clips are excluded from
this new holdout. Confirmation validation remains untouched during exploration.

The audit passes video-ID and source-group separation across all four
partitions. The proposed test set has **59 source groups for 60 videos**;
one class contributes two clips from one group. Do not treat all clips as
independent. This checks metadata only: source integrity, duplicate content,
full deterministic decoding, and preprocessing checks remain pending. No test
pixels or outcomes were accessed to prepare this proposal.

Use the same actual training videos in both phases. New seeds and new
validation groups provide replication of this fixed training-subset result,
not independent dataset replications. These adaptation splits also cannot
establish absence of overlap with foundation-model pretraining data.

## Training and measurements

Keep released V-JEPA 2 ViT-L/16 context/predictor weights, the frozen clean
target, native masked L1 loss, 16 frames at 256 x 256, and raw-pixel epsilon
4/255. Use the measured FP32 Conv3d configuration, microbatch one, accumulation
to eight, activation checkpointing, and cached fixed clean targets.

Both conditions restart from the released weights and use matched data order
and training mask seeds within a pair. Use the pilot's AdamW defaults: encoder
LR 1e-5, predictor LR 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.04,
no gradient clipping, one warmup epoch, and LR drops after epochs 15 and 23.
The adversarial condition uses PGD-5 with the original training attack settings.

Each run has 30 epochs and **390 optimizer updates**. Retain the final four
examples of each epoch and normalize that accumulated loss by four, not eight.
The cost calculator conservatively charges a full batch-eight update for it.
The original 2,020-video design had 7,590 updates/run. Equal epoch counts do
not mean equal optimization exposure; a negative result here cannot rule out
robust overfitting under longer training or larger datasets.

| Measurement | Schedule and purpose |
| --- | --- |
| Clean loss and PGD-5 | Initialization and every epoch on the 20 train/20 phase-validation clips; monitoring only |
| Primary PGD-20, R2, EOT4, four independent scoring draws | Initialization and epochs 1, 15, 16, 23, 24, 30 on both panels |
| Primary checkpoint selection | Minimum phase-validation loss among initialization and those six anchors; earliest exact tie |
| PGD-50, R2, EOT8, same primary scoring bank | Ten phase-validation clips at initialization, adversarial best/final, and clean final for each seed |
| Random-noise and FGSM checks | Same ten clips and selected checkpoints; native independent-mask scoring |
| Scoring-bank sensitivity | Two extra independent scoring banks on existing selected-checkpoint adversaries; do not select adversaries using these banks |
| Clean linear probe and feature checks | Initialization and selected checkpoints; fixed train-only fit, validation diagnostics, final test only after lock |

Routine PGD-5 never selects the best checkpoint. Sparse primary anchors may
miss deterioration or a better checkpoint between anchors. Training-side
strong losses describe the fixed 20-clip monitor; also record actual optimizer
losses over all training clips without conflating the two metrics.

Calibrate primary PGD step size on ten training-monitor clips at initialization
using 0.5/255, 1/255, and 2/255. Pick the highest mean independent-scoring loss,
breaking ties toward the smaller step. Freeze the choice before trajectories.
The 1/255 timing is only a proxy for other choices until measured. Keep the
training attack step size 1/255. All attacks are generated fresh for each
checkpoint; reuse random-seed definitions, never reuse stale perturbations.
Derive mask seeds from the declared fields with distinct attack/scoring
namespaces; freeze and test the serialization before running the study.

For the clean probe, mean-pool encoder tokens, center and scale features using
actual-training statistics, and fit a ridge least-squares one-hot classifier
with fixed lambda 1 and an unpenalized intercept. Fit each representation's
own probe; no hyperparameter sweep. Record unstandardized feature variance,
effective rank, clean native loss, and accuracy so standardization cannot hide
collapse. This small fixed probe is a diagnostic, not a tuned benchmark.

## Stages and decision gates

**Stage 0 — Finish locally and measure missing operations.** Implement the
training/resume runner, checkpoint retention, masks, evaluations, and fixed
probe interfaces. Freeze validated manifests and check content duplication
across groups. Repeat the novelty search before the paid experiment, honoring
the original instruction to halt if an exact matching study is found.
Use at most the remaining pilot reserve for a bounded measurement session.
Measure stronger attacks, actual data loading, diagnostics, and checkpoint
I/O. A successful short pilot does not establish full-study readiness.

Admit exploration only when measured unit costs replace proxies, every
essential missing category has a measured bound, a current quote is recorded,
and spending so far plus the remaining work, 25% headroom, and reserves fit
both $160 compute and $200 total. The eight-hour allowance is a target to
verify, not a substitute for timing. Use billed wall time, including setup,
transfer and idle time; reconcile actual charges after each session.

**Stage 1 — Exploratory pair, seed 0.** Complete clean and adversarial runs
and their diagnostics using only exploration validation. A proposed candidate
gate requires all of the following, fixed before training:

- The adversarial primary validation minimum improves on initialization by
  at least 1% of the initial loss.
- At least two later anchors among epochs 23, 24, and 30 exceed that minimum
  by at least 1% of initial loss, while the final primary training-monitor loss
  is lower than at the selected validation minimum.
- The direction survives the independent scoring-bank and stronger-attack
  checks. No collapse, broken gradients, target drift, or corrupted data
  explains the apparent effect. Examine the clean-control trajectory too.
- Reconciled costs still admit all three confirmation pairs and final testing.

The 1% thresholds are planning choices, not established significance tests.
If the effect is comparable to mask sensitivity, stronger attacks change the
interpretation, or diagnostics fail, call it inconclusive and repair/rebudget
before proceeding. If there is no candidate signal, report the entire
exploration and stop this proposal. Do not try new class subsets, seed lists,
or LR schedules until one produces a positive result. A justified new design
must be labeled a separate exploratory revision with its own budget.

**Stage 2 — Three confirmation pairs, seeds 11, 22, 33.** Freeze the design
before accessing the fresh confirmation validation groups. Run both conditions
for every seed, including unfavorable or failed outcomes. Selection uses
confirmation validation at the declared anchors. Do not pool seed 0 into
confirmatory effect estimates. Keep the same settings for all three pairs;
do not inspect final-test outcomes between seeds to decide whether to continue.

After every checkpoint choice and analysis rule is locked, evaluate the
60-video test set at the shared initialization, each adversarial best/final,
and each clean final checkpoint: **at most ten distinct checkpoints**. Reuse
identical best/final checkpoints when their full state and evaluation settings
match; budget for the upper bound. No exploratory test pass is included.

Report each confirmation seed's native-loss gap `final - validation_selected`
on test data, initial-to-selected improvement, training/validation curves,
and clean-probe change. A positive validation gap alone is selection-biased.
Consistent trajectories and positive held-out gaps in all three seeds support
the scoped claim; mixed seeds require a mixed/inconclusive conclusion. Use
paired source-group resampling, retaining the shared videos across seeds;
do not treat three evaluations of the same video as three independent videos.
Report per-seed effects and uncertainty, not just a pooled positive average.
Clean-final test results provide context, not a best-to-final clean-control
test curve or proof that deterioration is exclusive to adversarial training.

## Cost ledger and alternatives

Units come from the [October 19 RTX 5090 pilot](../report/checks/2026-10-19-runpod-5090-feasibility.md):
4.026 s/clean update, 24.425 s/adversarial update, 3.014 s/PGD-5 clip, and
73.442 s/primary-attack clip. These are short warm-cache measurements, not
end-to-end training throughput. The historical displayed $1.20/hour is an
assumption; a new session needs a current quote. No unmeasured speedup is used.

| Item | Planning allocation |
| --- | ---: |
| Pilot already spent, user-reported, including about 10 minutes idle | $0.592 |
| Remaining shared pilot reserve | $9.408 |
| Exploration pair and evaluation, with headroom and 2-hour untimed allowance | $33.64 |
| Three confirmation pairs and final evaluation, with headroom and 6-hour untimed allowance | $110.27 |
| Storage, taxes, transfers, other noncompute costs and contingency | $40.00 |
| **Total planning envelope** | **$193.91** |
| Nominal room below $200 cap | $6.09 |

The user reported the $0.592 charge on October 20; no invoice/API was checked.
The $10 deposited credit and $5 bonus are account funding, not experiment
spending or an increase in the $200 cap. Past pilot spending is inside the
$10 reserve. The confirmation amount is incremental after exploration.

Timed-unit extrapolations total 73.05 hours. Calibration and PGD-50 add 14.89
proxy hours; eight additional allowed hours bring the nominal work to 95.94
hours before 25% runtime headroom (119.93 hours after it). This corresponds to
about five continuous GPU-days, not five calendar days guaranteed. Software,
integrity checks, availability, and analysis still determine the schedule.

Counts include 9,660 routine clip-checkpoints, 1,980 primary train/validation
clip-checkpoints, 600 final-test clip-checkpoints, 140 stronger-attack
clip-checkpoints, and 30 training calibration evaluations. Initial train
evaluation is shared across phases; initial validation is not shared because
the panels differ. PGD-50's fivefold gradient-work proxy is not a measured
runtime or evidence of convergence. Zero missing-cost entries are not used.

| Candidate, with the same staged seeds/attacks | Updates/run | Planning envelope | Interpretation |
| --- | ---: | ---: | --- |
| 20 classes x 5 training videos | 390 | $193.91 | Recommended narrow study, conditional on measurements |
| 20 classes x 10 training videos | 750 | $210.98 | Exceeds cap under these assumptions |
| 10 classes x 10 training videos | 390 | $148.37 | Lower-cost alternative with narrower class coverage |
| 101 classes x 10 training videos | 3,810 | $724.96 | Exceeds cap |

These alternatives are cost comparisons, not automatic fallbacks after
observing results. The original six-run, larger evaluation proposal remains
about $950 before missing work; it is a different schedule from this table.

## Efficiency, storage, and implementation order

Evaluation dominates this proposal. First investigate batching independent
clips/mask draws and removing avoidable Python/transfer overhead while keeping
all attack steps, per-example candidate selection, restarts, and independent
scoring. Compare exact mask banks, losses, input gradients, and perturbation
bounds against the present FP32 reference before crediting a measured gain.

Then consider BF16 autocast for model operations with FP32 perturbations,
projection, target reference, and loss reductions. Validate attack gradients
and cross-evaluate candidates in FP32; a weaker numerical attack is not a
speed improvement. Autocast and gradient accumulation require deliberate
handling, especially if scaled `autograd.grad` is used; see the
[PyTorch 2.8 AMP examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html).
Do not assume a twofold speedup or silently change precision across conditions.
Target caching and activation checkpointing are already in the timing baseline.

Retain weights at the six primary anchors plus two rolling resumable epoch
states. Off-anchor weights can be replaced only after the next resume state
passes atomic-save/hash verification. One encoder/predictor FP32 weight file
is about 1.21 GiB; all 48 anchor files across eight runs total about 58.29 GiB.
That exceeds a 50 GB remote volume. Export completed runs to verified local
storage before removing remote copies. Two optimizer-state slots need roughly
7.29 GiB, with additional temporary space for atomic writes. Keep the fixed
target in the pinned pretrained file rather than duplicating it per run.
Measure actual disk usage and transfer time; refuse a new run if space is
insufficient. A 50 GB rental volume is not an archival plan for the full study.

Next implementation order: local training/resume tests; proposed manifest
export and integrity checks; primary evaluation/diagnostic interfaces;
bounded cost measurements; freeze the revised protocol; exploration; then the
conditional confirmation phase. Aim for the first full pair within several
working days after admission, followed by repeated runs and report writing
within the remaining four-week window. This is a schedule target, not an
assurance of a positive result or publication acceptance.

No instance was launched, no training trajectory was run, and no test pixels
were accessed while preparing this plan.

## Reproduce this planning check

From the repository root, with Python 3.11+ and the scaffold requirements:

```bash
python scripts/estimate_staged_study.py --output /tmp/vjepa-staged-projection-new.json
python -m pytest -q tests/test_staged_study.py tests/test_compute_plan.py
```

Use a fresh output name. The calculator verifies input hashes and refuses to
overwrite reports. It reads metadata only, needs no PyTorch/GPU/network, keeps
unknown timings null, and always reports `budget_approved: false`.
