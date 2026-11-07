# Native predictor boundary capture reviewed — November 6

**The saved attribution capture is complete and verified. The predictor-math
candidate remains numerically rejected.** Both workers exactly reproduce their
own preserved loss/input-gradient returns, confirming that these observations
correspond to the previously reported discrepancy. The original passes; the
candidate still violates the unchanged input-gradient bound at 3,724 of
3,145,728 elements. Both saved scalar losses are exactly 0.5617199540138245.

## What the actual output boundary shows

All sixteen actual prediction/cotangent pairs per arm were saved and checked.
The same targets and target indices select 22,601,728 feature values per arm.
Independent NumPy reconstruction verifies every actual incoming cotangent
against the native L1 tail, including its eight-draw / two-family averaging
and feature/token denominator. No model, surrogate loss, new oracle, extra
backward or CUDA operation was evaluated during this saved-data review.

Across the arms, **11 cotangent entries differ in nine of the sixteen pairs**.
They match exactly the changed signs of the FP32 prediction-minus-target
residuals: ten switch between opposite nonzero signs, and one changes from an
exact zero residual/cotangent in the original to positive in the candidate.
All eleven coordinates, actual predictions, clean targets, residuals and
cotangents are retained in the JSON report. For example, draw 0 / family 0,
target token 1335 / feature 605 has target -1.5121026039123535. Its original
prediction -1.5121002197265625 gives a positive residual; the candidate's
-1.5121060609817505 gives a negative residual. The actual cotangent changes
from +4.384709484384075e-8 to -4.384709484384075e-8.

Prediction values differ at 22,326,053 of 22,601,728 entries, with maximum
absolute difference 4.610419273376465e-5. Most differences leave the L1 sign
unchanged. The maximum cotangent difference is 1.2014794492642977e-7.
These are descriptive boundary observations, not a new prediction tolerance or
an adopted candidate. An identical reduced FP32 scalar loss does not require
identical elementwise residual signs.

| Draw | Family | Target tokens | Prediction entries differing | Cotangent entries differing |
| --- | --- | ---: | ---: | ---: |
| 0 | 0 | 1,392 | 1,407,403 | 1 |
| 0 | 1 | 1,648 | 1,665,320 | 1 |
| 1 | 0 | 1,232 | 1,245,145 | 2 |
| 1 | 1 | 1,760 | 1,783,129 | 2 |
| 2 | 0 | 1,104 | 1,115,067 | 0 |
| 2 | 1 | 1,760 | 1,783,113 | 0 |
| 3 | 0 | 1,168 | 1,180,117 | 0 |
| 3 | 1 | 1,536 | 1,555,407 | 0 |
| 4 | 0 | 856 | 863,176 | 1 |
| 4 | 1 | 1,552 | 1,572,922 | 1 |
| 5 | 0 | 1,016 | 1,024,942 | 1 |
| 5 | 1 | 1,560 | 1,578,104 | 1 |
| 6 | 0 | 1,192 | 1,205,254 | 0 |
| 6 | 1 | 1,792 | 1,816,468 | 1 |
| 7 | 0 | 1,048 | 1,057,907 | 0 |
| 7 | 1 | 1,456 | 1,472,579 | 0 |

This establishes an actual difference at the external loss boundary. It does
**not** quantify how much of the final input-gradient error comes from those
eleven changes, nor isolate pure backward arithmetic from internal forward
state changes. The observed native gradients are J0^T c0 and J1^T c1. The
common-cotangent cross terms have not been computed. A causal decomposition
would require a separately reviewed comparison; it cannot be inferred from
the entry counts, equal scalar losses, or backend-selection flags.

## Numerical checks and native execution

The original observed loss and input gradient are bitwise equal to their
passing saved anchors and to the prior native EOT8 derivatives. The candidate
observed loss/input gradient are bitwise equal to its preserved failed values;
that reproduction is attribution evidence only. Its input-gradient comparison
fails identically against the historical and fresh originals, with maximum
absolute error 1.4960765838623047e-5 and maximum threshold ratio
8.641704438152846 at the unchanged rtol 0.002 / atol 2e-7. Loss rtol 1e-5 /
atol 2e-6 also remains unchanged.

Two fresh processes (978427 original, 983182 candidate) each retain one complete
native GraphTask and 1,152 actual call rows: 576 forward calls and 576 backward
recomputations across 36 owners. All sixteen observations per arm have removed
hooks; graph ownership, masks, weights, target/pixel signatures, seeds and
runtime identities verify. Both observer and attention adapter are restored,
model state is unchanged, and no ownership refusals are recorded. The candidate
retains its existing study-guard rejection. Its final failure marker binds its
actual header and external records, and the passing-reference loader refuses it.

The candidate stops at the original input-gradient gate after saving the
observations and returns. Its recorded numerical-check list ends after the
passing loss check because the input check raises before appending a result.
This review independently computes and reports that failure; it does not edit
or fabricate a missing worker check. The parent reports complete attribution
data, but the numerical diagnostic remains failed/incomplete. Exit one is the
preserved numerical refusal. No CUDA profiling, kernel qualification,
five-phase replay, attack/scoring, parameter/optimizer backward or study
adoption follows from this one-phase observation.

## Evidence, resources and frozen provenance

The original response is 512,530,141 bytes; all 223 collected files comprise
512,303,338 logical bytes. Independent review verifies every raw-stream member
and local file against its hash, size, mode, exact nanosecond modification time
and total tar extent. It rechecks all 99 original packet members, five
value-only archive members and twenty approved overlays. Both input archives
and their extracted contents match the approved manifests. The original
candidate value capsule is still explicitly failed and not executable evidence.

All 10,424 sealed preparation inputs and authorization/source/public-host-key/
consumed-predecessor bindings reverify. A new 10,665-file manifest seals the
target evidence and review source. Raw review is saved at
`runs/pretrained/diagnostic/predictor-boundary-v1-20261106/independent-target-review/`.
Model/autograd/SDPA/CUDA and process/network execution are blocked during the
review. The 48 local preparation checks were not rerun or added to old counts.

The full 2.5 GiB write/fsync/readback proof succeeds in 9.538 seconds before
either input archive, with 1,926 retained allocation observations. The actual
readiness record identifies private local XFS on linux10603, separate from NFS
home; the proof is not a kernel quota or a reservation against other writers.
Whole work takes 121.019 seconds within the 1,200-second limit. Worker
supervisors take 77.689 / 29.788 seconds within their seven-minute limits.
Their recorded allocator peaks are 5,122,851,840 / 5,847,053,312 bytes, below
7 GiB. Peak supervised capture allocation is 512,835,584 bytes, below 2.25 GiB;
raw and metadata categories remain below 512 / 64 MiB. The sixteen original
and six candidate GPU samples show only each owned process when compute is
present. Sampling does not establish continuous exclusivity. No timeout, OOM,
storage refusal or foreign-process refusal is recorded.

The new attempt is **consumed**. Claim SHA256:
`701c196c878eea81f6b6810fc0c2bcde5468c257a111d224e6dd6d14e930fa7e`.
Control SHA256:
`7a586af8080447c0215884432d093db4cb1215c91fef04dfe24feefda72a7668`.
Authorization SHA256:
`43bcd65023f7f81f1de6977eaa6d086eb9890669f8a8a7d3511a91fa75497a78`.
Never rerun the helper, reset a claim/clock, edit captured sources/results, or
reuse the remaining time as another attempt. No standalone retrieval, cleanup,
parallel host, held-out work or paid compute was performed. linux10609 remains
unselected and unarmed; both university hosts may idle.

Next use these saved observations to prepare a minimal offline causal-comparison
plan that separates changed cotangents from changed propagation. No new target
calculation is authorized at this checkpoint. Original numerical gates and all
unresolved scientific qualifications remain. Financial caps, holds and pending
reconciliation are unchanged; known Runpod Pods remain terminated.
