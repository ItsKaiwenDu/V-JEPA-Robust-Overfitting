# Component worker v7: validated numerical receipt bookkeeping

This separate local integration uses the frozen numerical-receipt policy and
profiler metadata policy. Original v6 workers, helpers, plans, results, consumed
claims and failure states remain unchanged. There is no university operator,
handoff, authorization, new target attempt or paid compute at this stage.

## Explicit identities

Plans require `native-SDPA-fixed-input-plan-v7`, the unchanged profiler policy and
`numerical_receipt_policy = fixed-SDPA-numerical-receipt-bookkeeping-v1`. The source
manifest includes the numerical policy and the new v2 bounded record module.
Both canonical references and worker headers explicitly bind the same numerical
policy. The new reference and worker kinds are respectively
`native-SDPA-fixed-FP64-reference-v2` and `native-SDPA-fixed-input-component-v2`.
Completion markers bind the new kind, plan hash and complete header bytes. Old
readers and old plans cannot reinterpret new results, and new readers refuse old
records, missing/wrong policies and rewritten completion claims.

The target wrapper uses `native-fixed-SDPA-component-resource-v7`, explicit
numerical policy and the completed worker's exact hash, index, PID and startup.
The resource reader checks the same policy against the worker. It retains all
original type/owner/mode/link/size, positive peak <=7 GiB, configure-once startup,
120-second work and caller-umask rules. Local wrapper cases are simulated, not
measured CUDA performance or target readiness.

## Unchanged scientific work

After normalizing only versions, record-module selection, explicit numerical
policy fields and the two receipt-comparison calls, every core function's AST
matches frozen v6. Reference/return record functions are AST-identical to v1; only
their distinct kinds change. Wrapper and CPU CLI ASTs preserve all other actions.
Original operands, shapes/strides/storage/aliases, native loss/input/value anchors,
FP64 oracle and single-reference count, pointwise error bounds, backend selection,
precision/runtime, CUDA initialization owner, profiled bytes, serial predecessor
hashes, fresh-process identity and exact replay remain.

The generator's pointwise comparisons stay unchanged. The reader recomputes checks
from verified saved arrays, then validates all recorded fields with the frozen
numerical policy. Acceptance fields and original tensor tolerances stay exact.
Only diagnostic relative L2 uses the existing bookkeeping bound, 1e-12 relative /
1e-14 absolute; exact zero-error receipts keep zero summaries. The same check is
used for the canonical FP64 and passing-original summaries. Unknown/missing fields,
nonfinite or wrong-type metrics, genuine numerical failures and diagnostics beyond
the bound refuse. No summary is ignored, normalized or rewritten by production.
[Receipt policy](predictor_sdpa_receipts_v1.md).

Raw profiler metadata is still bounded, saved and checkpointed before semantic
validation. The frozen overhead/GPU-range interpretation, genuine CUDA witness
requirements and exact proof comparison remain. Failures retain available arrays,
trace and an incomplete marker, without resource receipt or next-worker progress.
A missing raw API cannot fabricate a trace. All storage limits remain unchanged.

## Local verification scope

The focused new integration suite exercises a normal cohort of four fresh small
CPU workers sharing one canonical FP64 reference. A separate controlled cohort
injects a one-last-bit change solely into nonzero final diagnostic summaries at
serialization, then completes the same four fresh serial worker processes and
reference sharing. Its attention arithmetic is unchanged, and its injected
metadata is explicitly labeled. Both cohorts check native values, profile equality,
math comparison, serial receipts and exact fresh replay through the new reader.

Other cases exercise real incomplete CPU failures and rehashed policy/record/
resource/trace/proof mutations, original bound violations, mismatched numerical
fields, old-reader refusal, single CUDA startup ownership, private receipts,
full storage/collection round trips and raw failure retention. The CPU entry still
refuses CUDA and university hosts. Previously verified policy and historical suites
are not rerun or accumulated. Actual local scientific work uses Torch 2.14.1 CPU;
it does not qualify university Torch 2.8/CUDA compatibility, speed or capacity.

Independent review reads saved evidence with new model/autograd/SDPA/CUDA/profile/
process/network/reference execution blocked. It verifies source pins, normalized
ASTs, actual saved CPU arrays/references/profile/fresh replay, injected/failure
classification, collection hashes/modes/nanosecond timestamps and unchanged
historical artifacts. Source snapshots, first test outputs and reports are retained.
The previous retrospective reader remains separate from production admission.

Next prepare and test a separate university handoff from the accepted native-input
capsule and new worker, binding numerical policy and all v7/v2 identities, including
any cold metadata adapter. Only after exact payload/imports/unused authorization/
live guards/storage/serial-clock/same-SSH collection tests could a new bounded
university proposal be considered. No old grant, helper, claim or time allowance
may be reused. Candidate rejection, full-training gates and financial caps/holds/
pending reconciliation remain unchanged.
