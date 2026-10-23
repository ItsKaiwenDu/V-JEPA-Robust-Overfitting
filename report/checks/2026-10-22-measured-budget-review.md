# Measured study budget review

October 22, 2026. **The unchanged staged proposal exceeds both reviewed caps
at the recorded RTX 5090 quote. No research run is admitted.**
[Reproducible calculation](2026-10-22-measured-budget-review.json).

The verified [shared-forward pilot](2026-10-22-shared-forward-cuda-workflow.md)
passes bounded CUDA recovery and representative numerical checks. Applying its
median update/evaluation observations to the existing 20-class, 100-training-clip,
eight-run proposal gives **$304.25**.
Using maximum-observed units gives **$307.39**.
These are short-sample projections, not measured complete-study totals or
statistical bounds. The old $193.91 envelope is superseded for this candidate.

## Accounting and workload

Keep 30 epochs, 390 updates per run, one exploratory pair and three confirmation
pairs; preserve all primary anchors, PGD strengths, validation panels and final
testing. The proposed precision policy is still unfrozen: these timings use
FP64 context-patch accumulation with FP32 learned parameters/tokens/tail, rather
than the original FP32 patch forward. No YAML or scientific threshold changes.

Counts remain 9,660 routine clip-checkpoints; 1,980 primary train/validation
clip-checkpoints; 600 primary final-test clip-checkpoints; 140 convergence clips
with two extra scoring banks; 140 FGSM and 140 random controls. Calibration
projects the measured three-candidate span to ten training monitors (30 attacks).
Checkpoint state/save is charged at 31 boundaries per trajectory, plus twelve
selected trained-anchor loads. Full probe/feature work remains a separate unknown.

| Timed operation | Median-unit projected hours |
| --- | ---: |
| Adversarial training updates | 17.920 |
| Ten training calibration clips, three candidates each | 1.209 |
| Checkpoint state creation and saves | 0.701 |
| Clean training updates | 3.049 |
| Convergence attacks and two scoring-bank audits | 24.575 |
| FGSM controls | 0.252 |
| Primary final-test evaluation | 22.872 |
| Primary training/validation evaluation | 75.478 |
| Random controls | 0.107 |
| Routine PGD-5 evaluation | 13.925 |
| Selected trained-anchor loading | 0.013 |

Timed-unit extrapolations total **160.10 hours**.
The historical eight-hour allowance (two exploration/six confirmation) remains
an **unverified allowance**, then 25% headroom is applied. At the user-reported
October 22 **$1.21/hour** including the selected running disks, add the entire
$10 shared pilot reserve once and $40 noncompute reserve:

```text
total = (timed hours + 8-hour allowance) × 1.25 × $1.21 + $10 + $40
```

The median projection is $264.25
against $160 compute and $304.25
against $200 total. Even omitting the unverified eight-hour allowance leaves
$292.15. The shared pilot reserve already
contains prior charges; **do not add the provisional $2.280 again**. The previous
confirmed $1.195 plus the full pending $3 session hold leaves $5.805 unreserved
in that pilot envelope. The user confirmed termination; billing is provisional.
The quote is historical for this review, not a new capacity or price check.

Clean native scoring is already inside each evaluated clip timing. The
convergence timing already includes both extra scoring banks. Neither is added
again. Disjoint `engine.state` and `checkpoint.save` spans are summed;
nested workflow/scheduled-evaluation/control totals are excluded.

## Existing alternatives

All retain the same eight runs, 30 epochs, attacks and confirmation design.
No new subset is selected or decoded by this comparison.

| Existing variant | Updates/run | Median units + allowances | Maximum-observed units + allowances |
| --- | ---: | ---: | ---: |
| 20 classes × 5 training clips | 390 | $304.25 | $307.39 |
| 20 classes × 10 training clips | 750 | $333.53 | $336.85 |
| 10 classes × 10 training clips | 390 | $219.35 | $221.12 |
| 101 classes × 10 training clips | 3,810 | $1270.13 | $1286.06 |

The ten-class alternative also exceeds the cap. Class-count variants are
planning comparisons; switching to one requires an explicit design revision
and new data review before observing research trajectories.

## What would need to improve

Primary train/validation/test evaluation is **61.4%**
of timed work. At unchanged measured throughput, the 20-class design needs
an all-in running rate of about **$0.714/hour or lower**
using median units, or **$0.705/hour**
using maximum-observed units, under the same overhead allowance. At $1.21/hour,
all timed work would need to improve by about
**1.76×**;
if only primary evaluation improves, it needs about
**3.34×**.
These are algebraic requirements, not predicted speedups or provider offers.
Do not assume a lower hourly price gives the same throughput.

The exploratory pair alone projects $59.13 in study compute using median units
and its two-hour allowance; that is not admission for the full research goal.
If confirmation remains required, its remaining cost must be affordable before
starting exploration. A discovery-only output would be an explicit smaller
study with more limited claims, not an automatic fallback after seeing a signal.

## Remaining uncertainty and next decision

Full-panel loading/identity/journal work, data review/decoding/targets, full-size
features/ridge/probe scoring, backup/export/temporary disk peaks, setup/idle and
failed work remain `null` in the ledger. Existing eight-clip features, cache
and I/O observations do not establish bounds for the complete study. Eight
allowed hours cannot certify them. Neither 80 GB storage sufficiency for all
retained anchors nor university 8 GB optimizer feasibility follows from this.

The primary timing summary pools seven 2/255 records and one each from
0.5/255 and 1/255 calibration candidates, all at the same attack strength.
The one-clip diagnostic chose 2/255 for primary PGD; the required ten-training-
monitor calibration has not frozen that choice. Convergence/controls were
timed on one pretrained monitor. Primary/update samples cover few inputs and
checkpoints. Numerical agreement and exact recovery remain required on any
new hardware; the earlier FP32 rejections remain unchanged.

The next choice is to preserve the 20-class design and prepare a cheaper-GPU
benchmark, or deliberately narrow the exploratory research scope. The former
is preferred before reducing coverage or attack strength. Use a fresh actual
quote, benchmark matching workloads/precision, and rebuild this same ledger;
an attractive GPU name or hourly listing alone does not pass the cost gate.
No provider is selected, no Pod is deployed, and no held-out pixels opened.

Reproduce offline from the repository root with a fresh output path:

```bash
.venv/bin/python scripts/estimate_measured_study.py \
  --output runs/pretrained/diagnostic/measured-budget-review-01.json
```

The calculator checks pinned sources, rejects changed workload/precision
evidence and preserves unknown costs. Targeted budget tests cover independent
schedule counts, no duplicated scoring or pilot charges, stale sources,
economic thresholds and rejection without automatic admission.
