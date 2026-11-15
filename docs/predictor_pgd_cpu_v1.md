# Tiny CPU PGD context prototype v1

This opt-in local prototype integrates the accepted original-forward/math-VJP
attention route into a two-step, two-restart EOT8 PGD loop. The bounded synthetic
CPU cohort and fresh-process replay passed on November 14. See
`report/checks/2026-11-14-predictor-pgd-context-cpu-v1-local-review.json` for the
measured results and `configs/predictor_pgd_cpu_v1.json` for the tiny contract.

`attacks/pgd_context_v1.py` retains the frozen PGD selection body. Its only body
changes are the tiny domain check and the explicit gradient-context bridge.
The AST regression test verifies that relationship. `attacks/pgd.py` is unchanged.

For each iterate, `IteratedBinding.context(current)` creates a new single-use
context. The initial paired run first checks an original gradient at a clone of
the same pixels, then binds the candidate's actual native scalar loss and exact
input leaf. PGD considers the per-example loss before `step.gradient()` runs the
native owned backward. A stale/reused/wrong input or loss token, higher derivative,
changed modes/parameters/targets/masks, foreign thread, invalid count or exhausted
allowance refuses. No global autograd hook or patch redirects other objectives.

Original no-grad clean/last-iterate attack scoring and independent clean/adversary
scoring run outside every custom context. The fixed attack bank selects over clean,
all random starts and all iterates using strict per-example `>` comparisons.
Independent scoring uses a separate fixed four-draw bank and never selects.
Original target features are computed once from clean pixels. No optimizer or
training code calls this experimental API.

Every objective records actual prediction values, native L1 incoming cotangents,
full input gradient, native root/checkpoint observations, candidate inner math
root/leaf observations and restoration checks. Private, exclusive output files
retain partial failures. The start-before-work ledger charges any attempted
objective and refuses reuse. The old native per-objective call caps are unchanged.

`models/predictor_pgd_records_v1.py` reads saved evidence using NumPy and the
standard library only. It independently checks native/L1/proof values, all-iterate
numerical agreement, projection, strict candidate selection and fresh replay.
`tests/test_predictor_pgd_context_v1.py` supplies 40 new cold checks; its analytic
simulation gradients do not count as native numerical qualification.

The one completed stage is
`runs/pretrained/diagnostic/predictor-pgd-context-cpu-v1-20261114/`. Both local claims
are consumed: eight initial objectives plus four conditional fresh-process
candidate objectives exhaust the declared 12. Do not rerun this stage, reset its
claims or repeat the successful cohort. The one initial cold reader failure is
preserved with all source snapshots; only the failed cold check was retried.

This entry refuses CUDA/non-CPU tensors, the two known university hosts, inference,
autocast, non-tiny geometry and counts other than PGD2/R2. The synthetic model has
random weights; this result does not qualify pretrained GPU iteration, real
kernels, performance, attack strength, PGD20/PGD50, full AdamW training or recovery.
The 1/255 step is predetermined for the tiny check, not research calibration.

The next stage must prepare a distinct exact pretrained GPU iterated-input and
backend scope, preserve these completed results and seek one scoped authorization
only after concrete preparation. No university helper or authorization is ready
at this checkpoint. All budget reservations and research adoption gates remain.
