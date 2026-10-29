# Measured EOT8 study-cost screen — October 28

The retained **20-class / 100-training-clip / eight-run** design screens at
**$243.71–$249.62** using the recorded RTX PRO 4500 SE
$0.74/hour running quote. This exceeds the unchanged **$200 project / $160 compute**
caps. Do not launch research calibration or trajectories on this basis. This is
an incomplete measured-unit extrapolation, not a full-study bound or an invoice.

| Screen from short-run observations | Median units | Maximum-observed units |
| --- | ---: | ---: |
| Timed workload, including disjoint panel wrappers | 201.42 h | 207.80 h |
| With historical 8 h allowance and 25% headroom | 261.77 h | 269.75 h |
| Study compute plus the single $10 pilot reserve | $203.71 | $209.62 |
| Total including $40 noncompute reserve | $243.71 | $249.62 |
| Total even with the historical 8 h allowance removed | $236.31 | $242.22 |
| Break-even all-in rate at identical sample throughput | $0.5730/h | $0.5561/h |
| Uniform timed-work speed factor needed at $0.74/h | 1.307x | 1.348x |

Primary native evaluation is 74.6–75.0% of the timed work. The
price ceilings assume the same measured throughput and the historical allowance;
they are necessary arithmetic conditions for this partial screen. They do not
predict A5000, RTX 3090, A40 or another GPU's speed, availability or price. Missing
work can tighten these limits. Even a future partial screen below $200 would not
establish complete budget admission.

## Workload and measurement accounting

The original schedule remains: four clean and four adversarial runs, 390 updates
per run, initialization, all six primary anchors, routine evaluation every epoch,
and conditional confirmation/test evaluation. Counts are 1,560 updates for each
condition; 9,660 routine clips; 1,980 train/validation and 600 final-test primary
clips; 140 clips each for convergence, FGSM and random controls; ten complete
three-candidate calibration bundles; 248 state/save boundaries; and 12 selected
anchor loads. Confirmation/test counts are metadata only; their pixels stay closed.
Full eight-clip update units are applied without discounting final partial batches.

Eight zero-audit PGD20/R2/EOT8/score4 primary timings include three pretrained,
two two-update trained-anchor and three diagnostic step-candidate records.
Repeated clips and steps are technical samples, not eight independent videos.
Primary settings and source/runtime provenance match. Ordinary primary timings
do not include the separate primary-with-two-audits control sample.

Clean and main scoring are already in native evaluation time. PGD50 native time
already includes both audit banks. Native times are not added to their enclosing
wall spans: only the positive wall-minus-native residual is applied to **634
panels** (483 routine, 109 primary, and 14 each for convergence/FGSM/random).
The observed residuals add 0.90–0.92 hours. The ten three-step
bundles already contain all thirty calibration evaluations and their wrappers.
State construction and checkpoint save spans are disjoint. Nested phase totals,
recovery comparisons and one-time diagnostic preflight are not charged again.

## Unknowns and retained gates

The measured one/three-clip wrapper residuals do not bound full-panel identity,
journal, transfer or data-loading overhead. Full fixed-target caching, 100-clip
features/ridge fit, held-out probe scoring, checkpoint retention, temporary disk,
verified exports, future setup/idle/failure costs and out-of-quote charges remain
unknown. Eight-clip feature-fit time is not scaled linearly to the full study.
Any additional primary-with-audit replays required by the eventual frozen protocol
remain unassigned costs, not invented counts or assumed free work. The historical
two-plus-six-hour allowance is a placeholder, and 25% headroom is not a missing-
cost measurement or a confidence interval. Both observation summaries are short-
run extrapolations; maximum-observed is not an upper timing bound.

The one-clip diagnostic choice of 2/255 is not research calibration. The original
ten-monitor / thirty-evaluation calibration, chosen research step, policy adoption,
protocol freeze and research admission stay pending. No study YAML, attack, seed,
epoch, dataset split, old numerical rejection or scientific source pin changes.
No tensor deserialization, GPU launch or cloud connection occurs in this task.

Final billing for the terminated benchmark is pending. Keep the $1.75 session
reservation and historical hold/gap: conservative diagnostic encumbrance stays
$9.22, leaving $0.78 unreserved inside the $10 pilot envelope. Do not add those
encumbrances or actual prior pilot charges again on top of the full $10 reserve.
No hold is released or created; recharge ability does not increase either cap.

Next reconcile the final charge, then prepare a sufficient-memory lower-cost GPU
qualification if the remaining diagnostic allowance permits. Obtain fresh total
quotes and usable RAM, and benchmark affected units on that device before full
calibration. Do not cut evaluation strength or the retained design to make the
budget screen pass.

## Reproduction and validation

Run `.venv/bin/python scripts/estimate_candidate_study.py --output NEW_PATH.json`
from the repository. Existing outputs are refused. The calculator verifies the
completed diagnostic archive, all 169 members, all 61 candidate source pins and
242 total input fingerprints before computing. It preserves all historical
reports and verifies the recorded quote; it does not check a new live quote.

All **32 focused tests pass**, with zero failures/errors/skips. They cover exact
independent Fraction arithmetic for both scenarios, all retained workload counts,
separate wrapper costs, calibration/clean-score/audit nonduplication, step/settings
and evidence drift rejection, unknown-cost preservation, source refusal and no-
admission behavior even under hypothetical favorable rates. The builder reproduces
identical output and the scaffold check passes. No broader test rerun was needed
because all existing scientific inputs remain byte-identical.

[Structured screen](2026-10-28-candidate-measured-budget-review.json) ·
[Validation record](2026-10-28-candidate-measured-budget-validation.json).
Raw test output and the exact estimator/test snapshot remain under
`runs/pretrained/diagnostic/candidate-measured-budget-20261028/`.
