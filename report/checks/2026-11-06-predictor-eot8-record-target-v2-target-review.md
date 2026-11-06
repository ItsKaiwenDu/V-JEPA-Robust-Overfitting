# EOT8 baseline verified; predictor gradient candidate rejected — November 6

**The storage correction worked in the target run. The original baseline passes;
the predictor-math candidate fails the original input-gradient bounds.** The
four-worker diagnostic is incomplete and failed. No replay workers started.
This is a numerical rejection, not a successful candidate or research result.

## Saved evidence and original baseline

The 238,107,286-byte same-SSH response contains all 159 files / 237,948,160 logical
bytes. Independent review verifies every stream-member and saved-file hash,
size, mode and exact nanosecond modification time, plus the full stream extent.
All 99 input packet members and 17 approved overlay files match the sealed
control. The 7,330 predecessor inputs remain byte-identical; 7,505 inputs are
now sealed. Model/autograd/SDPA/CUDA and process/network execution were blocked
during this local saved-file review. No model or derivative was rerun.

Original worker 0 completes all five phases and saves all ten actual FP32
returns with a final hash-bound completion marker. Both full-EOT8 loss/gradient
pairs match the saved historical reference exactly; six within-process control
comparisons are also exact. The initial checkpoint, pixels, targets, seeds,
eight draws, both mask families and runtime match the frozen specification.
The model/targets-unchanged and adapter-restored receipts are present.

All 2,736 call rows and five complete backward scopes are exported and verified
by the unchanged graph-ownership reader, including five observed scopes whose
root-hook/recomputation thread differs from the creator thread. Every hook was
removed; no ownership refusal is recorded. The unchanged raw-event reader was
independently rerun on 216,858 events / 81,250,156 bytes and reproduces the
supervisor's accepted receipt. Actual efficient-attention CUDA forward/backward
links are established for both families and all 36 owners in the **profiled
first draw only**. This does not qualify every geometry or speed/cost.

The new external call file is 2,255,474 bytes, while the worker header is only
19,491 bytes, scopes 1,933 bytes and completion marker 436 bytes. The prior 1 MiB
operational JSON bound is unchanged; full call evidence now fits its separately
bounded file. No record overflow or missing completion marker recurred.

## Candidate numerical rejection

Candidate worker 1 saves the full EOT8-before scalar loss and input gradient
before the numerical gate. The loss is **0.5617199540138245**, exactly equal to
both the historical reference and fresh passing original. Its input-gradient
comparison fails the original elementwise rule:

`abs(candidate - reference) <= 2e-7 + 0.002 * abs(reference)`

| Saved comparison | Result |
| --- | --- |
| Gradient elements | 3,145,728 |
| Elements beyond the original bound | 3,724 (0.11838%) |
| Largest error / allowed threshold | 8.641704438 |
| Largest absolute difference anywhere | 1.496076584e-5 |
| Relative L2 error, descriptive only | 4.683764242e-5 |

The largest absolute difference occurs at an element whose own threshold is
larger and is not the most severe normalized violation. At the worst normalized
element `[0, 0, 2, 233, 112]`, candidate is -1.805456122e-5, reference is
-1.604884164e-5, absolute error is 2.005719580e-6 and allowed error is
2.320976833e-7. Aggregate closeness does not override the original elementwise
gate. No tolerance was relaxed, rounded or waived.

The live rejection occurs at the earlier historical-reference gate; the fresh
original is exactly that same reference, so independent comparisons against
both give identical failures. Initial weights/pixels/targets/masks/seeds/runtime
match. These saved values establish the discrepancy but do not locate which
predictor forward/backward operation caused it or prove it harmless.

The failure record binds the saved candidate header, all 1,152 call rows and
one complete backward scope with its hook removed. The adapter was restored,
and no ownership refusal is recorded. The candidate header is 8,913 bytes;
calls are 950,698 bytes, scopes 798 bytes and final failure marker 441 bytes.
The strict loader refuses it as a complete passing reference. Its remaining
four phases and both replay workers never run. The candidate's profiling phase
was not reached, so its actual CUDA kernel backend is **not independently
qualified** by this run. Recorded math-selection flags are not kernel proof.

## Resource and authorization accounting

The full 2.5 GiB write/fsync/readback proof passes in 9.755 seconds, with all
1,926 allocation observations inside the unchanged setup bound. Archive
storage follows the proof. Total work is 154.975 seconds; worker supervisor
intervals are 95.277 and 32.604 seconds. Recorded allocator peaks are
5,128,348,672 bytes for the baseline and 5,847,053,312 bytes for the candidate,
both below 7 GiB. Storage, metadata, raw-capture, file and time bounds remain.
No timeout, OOM, foreign-process or storage refusal is recorded.

The 19 baseline and seven candidate GPU snapshots contain only their respective
owned compute PIDs (652681 and 658470). These are sampled observations, not proof
of continuous exclusivity or a storage reservation. No processes were killed.
All 12 actual FP32 returns, the original CUDA trace and the failed candidate
metadata remain preserved. Collection used the original SSH connection and
started no second model or retry.

The one approved attempt is consumed:
`4dd65b03ea11ff2f67db43008130dbc7a84e75c7b29c01e1c5b2f84e0bd859bd`.
Keep its helper, source, authorization, claim and all captured files immutable.
Do not rerun it, reset its clock or reuse unused worker slots. The old pending
linux10609 handoff stays unselected and unarmed. Both university GPUs may idle;
no further remote operation or cleanup is required for this review. No tests
were rerun and no paid compute or financial holds changed.

## Next step

Prepare an offline attribution plan to distinguish predictor-forward sensitivity
from backward-implementation differences while retaining the original
numerical bounds. Start from the saved values and previously verified fixed-input
controls; do not adopt this candidate or infer a root cause from the loss alone.
Any new target calculation needs its own concrete reviewed scope and approval.
Historical failures stay preserved, and full replay, attacks/scoring, parameter
and optimizer behavior, performance/cost and research admission remain pending.

[Machine-readable review](2026-11-06-predictor-eot8-record-target-v2-target-review.json).
