# Same-input predictor attention comparison: offline proposal

The [four-cell result](../report/checks/2026-11-06-common-cotangent-v1-target-review.md)
shows that eleven changed predictor-output cotangents dominate this sample's
input-gradient discrepancy in both ordered decompositions. The native candidate
still fails 3,724 original input-gradient entries. The fixed-original-cotangent
cross comparison has zero original-bound violations, but a nonzero path term
and interaction remain. Those cross gradients qualify no native loss or attack.

This checkpoint prepares an inert [proposal](../configs/predictor_sdpa_fixed_input_v1.json)
and tests a **small local CPU observation prototype**. There is no university
operator, helper, upload, authorization or new attempt. All previous sources,
captures, rejected values, numerical bounds, clocks and consumed claims remain
unchanged. Both university sessions may stay idle or close.

## The question this comparison can answer

The previous experiment fixed cotangents at the **predictor output**. It did
not fix the Q/K/V inputs or incoming gradient at an internal SDPA operation.
The saved attention rows contain layouts, flags and ownership, not those
tensor values. Predictor-output cotangents must not be relabelled as SDPA
output cotangents. Actual pretrained post-RoPE Q/K/V and native SDPA incoming
gradient must be captured before a target same-input comparison can run.

Holding Q/K/V and `grad_output` fixed removes upstream activation differences
and the external incoming-gradient difference for one attention operation.
It still lets each public SDPA backend run its own forward and save its own
rounded internal state. This is an **attention forward/VJP comparison at
identical inputs**, not isolation of backward arithmetic alone. Even equal
returned outputs would not prove equal saved probabilities, log-sum-exp or
other backend intermediates. Private backend backward APIs or fabricated saved
states are outside this proposal. Do not claim a pure backward bug from it.

## Fixed boundary selection

Select original cell `00`, draw 0 / family 0, predictor block 0, first forward
SDPA call: row 24 in the frozen native call record. This follows execution
order rather than ranking numerical errors or scanning for a bad layer.
The full ordered eight-draw/two-family call coverage is checked before deriving
the selection. No alternative boundary is selected after seeing its result.

| Operand | Recorded shape | Recorded stride |
| --- | --- | --- |
| Q | 1 × 12 × 2048 × 32 | 786432, 65536, 32, 1 |
| K | 1 × 12 × 2048 × 32 | 786432, 65536, 32, 1 |
| V | 1 × 12 × 2048 × 32 | 2359296, 32, 1152, 1 |

All are actual FP32 CUDA tensors requiring gradients. Arguments remain
noncausal, no attention mask, zero dropout and the upstream default scale.
The original row's entry, effective and restored flags match; this records
selection flags, not the original CUDA kernel family. Original full-model
kernel identity remains unqualified.

## Proposed capture and four standalone workers

Prepare and test the real observer/operator separately before requesting any
new university resource exception. The proposed serial order is:

1. One fresh **original-only native capture worker** runs the unchanged clean
   full-model EOT-8 forward/backward with the new observer. It captures the
   selected actual Q/K/V, SDPA output, incoming SDPA `grad_output`, and native
   dQ/dK/dV. Hooks observe values and return None; no cotangent is replaced.
   Loss and full input gradient must reproduce the passing original anchor
   exactly, with all original state/ownership/restoration checks and source
   bindings. Unexpected multiplicity, missing hooks, refusal or mismatch
   stops before any component worker. The rejected full-model candidate is
   not rerun or promoted.
2. A fresh **fixed-original-0** worker loads the captured bytes without a
   generator or recapture. Its standalone default SDPA output and dQ/dK/dV
   must exactly reproduce the captured native values. Otherwise the standalone
   execution is not an adequate value control and the plan stops. This exact
   replay is necessary but does not establish the full-model kernel identity.
3. A fresh **fixed-math-0** worker uses the same saved Q/K/V and the same native
   SDPA `grad_output`, changing only the scoped public SDPA backend selection.
   It must pass the unchanged analytical and passing-reference component bounds.
4. Fresh **fixed-original-1** and **fixed-math-1** workers verify bitwise replay
   within each arm, without regeneration or time/reset reuse. No old worker,
   cross cell or rejected candidate supplies a passing reference.

All four standalone workers hash the entire actual device Q/K/V and incoming
gradient, and preserve their recorded layouts. Logical bytes, stride, offset,
alias relationships and materialization choices must be explicit. Independent
clone storage must not be presented as identical physical allocation or
alignment. The exact native-baseline replay remains mandatory; a default
standalone kernel can differ from full-model dispatch even with the same values.

Each standalone worker saves untimed output and all Q/K/V derivatives before
its numerical gates. One separately profiled iteration follows only when the
untimed gates pass; its values must replay exactly. Preserve actual device-typed
forward/backward events. A default or math flag, task name or CPU result does
not prove which CUDA kernels ran. The default arm's actual family is observed,
not assumed to be efficient attention. There is no warmup, timing benchmark,
training, attack, scoring, optimizer or held-out evaluation.

One fixed CPU FP64 analytical reference, using the existing query-chunked
NumPy softmax/VJP oracle, is built from the actual captured inputs and native
incoming gradient. All workers use the same frozen reference bytes. Reference
publication, independent reconstruction and original bookkeeping bounds must
be tested before implementation is called ready. Historical CPU oracle
failures remain recorded; no original gate is widened or bypassed.

Output bounds stay rtol 1e-4 / atol 2e-5; dQ/dK/dV stay rtol 0.002 / atol 2e-7.
Oracle bookkeeping stays 1e-12 / 1e-14. Full-model loss stays 1e-5 / 2e-6 and
input stays 0.002 / 2e-7. Finite, nonzero, exact replay and unchanged runtime,
flags, model state, source and native ownership are still required. All actual
arrays and partial/error receipts must survive a refusal. Missing publication,
changed previous values or any guard failure stops the sequence with no retry.

## Interpretation decided before target results

| Outcome | Permitted conclusion |
| --- | --- |
| Capture changes the native loss/input, or standalone original fails exact native replay | Control is inadequate; stop. No component or candidate qualification. |
| Either arm fails independent FP64 or reference bounds | That fixed-input attention comparison is rejected. This is not by itself a backward-only cause. |
| Both arms pass all bounds but differ bitwise | Describe the measured component difference and retain it; native full-model candidate rejection remains. |
| Both arms match and fresh replay passes | That one selected component comparison passes. It does not establish other layers, kernel speed or full-model native acceptance. |
| Private forward state would be needed to distinguish forward from backward arithmetic | Prepare a separate version-bound design; do not silently substitute private backward calls here. |

Reusing original predictor-output cotangents during research would change the
candidate's native derivative. This proposal does not authorize that shortcut,
a smoother loss, tolerance changes, a new training objective or a backend
adoption. Both measured four-cell orders and their interaction remain evidence.

## Prospective resource envelope, not a grant

One new attended linux10603 attempt is proposed for a future reviewed operator:
25 minutes including at most 180 seconds setup/full 2.5 GiB proof, one original
capture worker at most 420 seconds, four component workers at most 120 seconds
each, and a continuing 120-second receipt reserve. Static allowance is 1,200
seconds within 1,500. Same-SSH collection may take at most ten additional minutes
and 3 GiB. Old unused time is not reused. No operator or authorization exists
at this checkpoint; this envelope grants no SSH, proof or target calculation.

The proposed private local-XFS `/var/tmp` ceilings remain setup 3 GiB / capture
2.25 GiB; raw 512 MiB, metadata 64 MiB, input/code 128 MiB each, cache 32 MiB,
emergency 16 MiB, individual tensor file 16 MiB and allocator 7 GiB. There is no
automatic retry, cleanup, standalone retrieval, fallback/parallel host, weight/
dependency installation, private-key upload or paid compute.

Each logical FP32 boundary array is 3,145,728 bytes. Eight capture arrays use
25,165,824 bytes; four FP64 references use another 25,165,824. Four workers with
untimed and profiled output/dQ/dK/dV use 100,663,296 bytes. Together with a
20 MiB + 4-byte full-model loss/input/target allowance, the steady tensor
projection is 171,966,468 bytes; a 16 MiB atomic-save allowance still fits the
512 MiB raw category. The non-dense V layout spans 9,434,112 bytes per independent
FP32 backing, versus 3,145,728 each for Q/K. These are projections, not observed
GPU allocator/temporary/file peaks or reservations. Actual guards must hold.

## Local evidence and next implementation

25 focused local checks pass with no skips. A small CPU native SDPA example
captures one boundary directly and through nonreentrant activation checkpointing.
Before/observed/after full toy loss, output and input gradients are bitwise
equal; hooks are removed. Actual captured output and dQ/dK/dV replay exactly
with the same Q/K/V and incoming gradient, preserving non-dense V strides and
passing the independent NumPy FP64 analytical bounds. Recomputed forward calls
do not become a second capture. Changed incoming gradient, missing backward,
primary exceptions and nonlocal/nonfinite/full-size domains refuse or preserve
partial data. The prototype is restricted to tiny local FP32 CPU tensors.
It contains no live model route, capture files, target operator or CUDA proof.

The first test run had 24 passes and one incorrect handwritten expected V
storage span. The production extent calculation and recorded layout were
correct. Only that test expectation changed; original and corrected sources,
logs and receipts are retained. The second run passes all 25 checks. Counts
are not added, and earlier 68-check preparation tests were not rerun.

All 11,998 sealed predecessor files are reverified, including the consumed
four-cell claim. The proposal is generated without scientific imports or
model/network/process execution. Next implement and review a separate native
one-boundary observer with exact ownership and local end-to-end controls;
then build the bounded fixed-input operator/transport. Only after that concrete
handoff is tested should one new scoped approval and a fresh GPU status be
requested. There is no target command or user SSH task now.

The $200 project / $160 compute / single $10 diagnostic caps, conservative
$9.8116 encumbrance, $0.1884 unreserved amount, old $2 hold / $0.0792 gap,
last reported $6.56 wallet and pending reconciliation remain unchanged.
Known Runpod Pods remain terminated.
