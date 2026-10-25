# Broader training-only matched diagnostic plan

October 24, 2026. Predeclaration status at freezing: metadata-tested, before
runner implementation. The later [runner/handoff](matched_panel_handoff.md)
is implemented and CPU-tested; the CUDA panel remains unexecuted. The machine-readable
specification is `configs/matched_panel_diagnostic_v1.json`. It does not alter
`configs/study.yaml` or `configs/budget_study_plan.yaml`.

The [one-clip result](../report/checks/2026-10-24-university-matched.md) motivates
checking variability before choosing final attacks. The first ten existing
training monitors are the same prefix declared for research calibration;
the broader diagnostic itself is not the research calibration.

## Fixed rows and contrasts

| Index | Video ID | Original class ID | Source group |
| ---: | --- | ---: | --- |
| 0 | v_BreastStroke_g10_c02 | 18 | BreastStroke_g10 |
| 1 | v_JumpRope_g16_c02 | 47 | JumpRope_g16 |
| 2 | v_Swing_g20_c03 | 88 | Swing_g20 |
| 3 | v_TennisSwing_g14_c04 | 91 | TennisSwing_g14 |
| 4 | v_FloorGymnastics_g17_c01 | 29 | FloorGymnastics_g17 |
| 5 | v_Biking_g16_c04 | 10 | Biking_g16 |
| 6 | v_Archery_g19_c04 | 2 | Archery_g19 |
| 7 | v_YoYo_g14_c03 | 100 | YoYo_g14 |
| 8 | v_Bowling_g19_c04 | 15 | Bowling_g19 |
| 9 | v_HandStandPushups_g21_c01 | 36 | HandstandPushups_g21 |

For every row use bank0 then bank1, with A20/EOT4, B50/EOT4, C20/EOT8,
D50/EOT8 followed by a fresh-process A-only replay. Preserve actual video IDs;
do not launder bank labels through synthetic video identifiers or monkeypatch
the old runner's global seed. Four draws always mean the first half of the
same eight-draw attack bank. Both replicates share starts, targets and main/
audit scoring seeds within their clip. No score selects an iterate or restart.

The seed base is 20261024 in a new `matched_panel_v1_` namespace. All seed arrays,
starts and metadata identities are explicit in the specification and rederive
from the existing evaluation-seed algorithm. Global upstream-mask collisions
and historical seed reuse reject. Input metadata comes only from training;
no new video decode or pixel/weight loading occurs in the planner.

## Review and implementation contract

Integrity requires all 20 cases, four matched arms and exact A-only replay per
case, retained native CUDA evidence, finite nonzero input gradients, pixel
bounds and frozen source/runtime/data/weights/targets. Independent score losses
need not be monotone. No all-arm fresh replay or full optimizer validation is
claimed. All failures stop further dispatch and preserve partial evidence.

Calculate paired B-A/D-C/C-A/D-B contrasts separately for each scoring bank.
Average technical bank replicates within clip, then average ten clips equally.
Keep per-case/per-clip values and descriptive signs/ranges. Do not pool audits,
choose cases or arms using scores, claim independent sample size20, or publish
partial-panel means as the planned result. There is no newly invented numeric
convergence tolerance or automatic method selection. Hold canonical study
settings fixed pending broader evidence, research calibration and budget review.

The runner implementation must bind the frozen plan hash, explicit case ID/seeds, actual
runtime and its own qualified execution-source manifest. It must verify the
first case before dispatching the remainder, use one process group at a time,
and journal only independently verified whole cases atomically. Reuse on
resume requires exact identities and all artifact receipts; partial cases need
a fresh attempt with their failed evidence preserved. No silent skip/retry.

The target remains the released fixed clean encoder; context/predictor use
shared-fp64-native only. Step2/255 here is diagnostic. The existing research
step-size comparison 0.5/1/2 remains a separate uncompleted gate.

## Resources and next action

The [predeclaration report](../report/checks/2026-10-24-matched-panel-predeclaration.md)
records 67 passing focused checks (36 new) and the CPU seed audit. Its13.65h
repeated-reference/17.07h margin scenarios come from one observed case. They
are not predictions, calendar deadlines, memory guarantees or A40 observations.
Serial case deadlines,7GiB/90% allocator limits,256MiB evidence limits and a
12GiB free-space requirement are specified in JSON. At predeclaration no executable panel
scheduler or handoff existed; the later implementation is linked above. This
file authorizes no paid compute.

CPU mask sampling used local PyTorch2.14.1. Future execution keeps the recorded
university PyTorch2.8.0+cu128 stack, must regenerate masks in that runtime and
retain actual hashes; cross-runtime equivalence for all 280 draws is not assumed.
Do not promote the local dry-run hashes to a new acceptance gate.

The separate implementation now supplies the bounded runner, first-case review,
receipt journal, complete-panel aggregation and handoff. Read its preparation
report before use; a current university hardware check is still required before
any launch. Existing study/budget/held-out gates remain.
