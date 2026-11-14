# Component worker v6: explicit profiler metadata policy

This separately versioned core integrates the verified overhead/GPU-range metadata
policy without changing the original component arithmetic or scientific gates.
The current executable rehearsal remains local CPU only. There is no university
operator, handoff, control, authorization or attempt at this stage.

## Bound identities and unchanged science

`predictor_sdpa_component_v6` requires plan kind
`native-SDPA-fixed-input-plan-v6` and explicit metadata policy
`fixed-SDPA-profiler-overhead-signature-v1`. Its source manifest includes the new
metadata verifier as well as the frozen raw extractor and old verifier. The trace
descriptor must independently contain the same policy, raw format, encoding,
count, actual runtime and publication-before-validation marker. Old/missing/
modified policy fields refuse; recomputed file/header hashes do not bypass these
bindings. Observed proof is recomputed from saved bytes and compared exactly.

The raw envelope stays unchanged and bounded to 8 MiB/20,000 records. The extractor
still retains all serializable metadata before semantic acceptance. Overhead and
GPU phase ranges are represented separately from real CPU owners and positive
non-annotation CUDA kernels under the already verified narrow policy. See
[policy details and pinned sources](predictor_sdpa_overhead_v1.md).

After normalizing only version identities, the metadata-provider import, and the
explicit policy fields, every core function's AST matches v5. Phase label values,
operands/layouts/aliases/offsets, original native anchors, FP64 analytical reference,
pointwise numerical bounds, selected backend flags, runtime/precision, ownership
of CUDA initialization, input/gradient versions, reference binding, serial
predecessors and fresh-process replay checks stay unchanged.

The target resource wrapper and local CPU CLI are AST-identical to v5 after only
version normalization. The wrapper requires resource kind
`native-fixed-SDPA-component-resource-v6`, its exact saved worker hash, plan/index/
PID and valid startup receipt. Peak memory stays positive and at most 7 GiB,
post-entry work at most 120 seconds, private receipt mode 0600 with mask restored
on success/failure. These wrapper cases are simulated locally; no measured target
GPU peak or actual CUDA execution is inferred from them.

## Retention and failure behavior

The worker saves profiled output/gradient arrays, writes and hash-binds the entire
raw trace, checkpoints the worker descriptor, then applies the new semantic policy.
Original profile replay, backend evidence and final input/flag checks still follow.
Malformed metadata or later failures retain available arrays/trace and an
incomplete failure marker. Missing raw APIs retain arrays without fabricating a
trace. A failure never produces a complete worker or GPU resource receipt.

No old worker or claim is reopened. The same actual saved 106-record GPU trace can
be copied into an explicitly local metadata fixture and read by the new descriptor
and verifier. All raw bytes remain identical; overhead row 10 and GPU range row 20
are retained, with 12 real metadata kernel witnesses. The fixture has no completed
scientific worker or GPU receipt and does not supply missing historical checks.
The old validator still refuses, the old worker stays incomplete, and components
1/2/3 remain unrun. Runtime substitution refuses before metadata acceptance.

## Local verification

61 focused integration checks pass on the first run, without failures/errors/skips.
One cohort executes four fresh small CPU workers: original-0, math-0, original-1,
math-1. One canonical FP64 reference is built and reused; all numerical/native/
profile/input/flag checks and bitwise fresh replay pass. CPU results never claim a
CUDA receipt. Separate actual small CPU failure cases cover association rejection,
missing raw API and an injected unrecognized negative-index record. The latter is
synthetic; it does not replace the saved university overhead record.

Additional cases cover rehashed completion/trace/policy/proof mutations, old-plan
and old-reader refusal, single CUDA configuration ownership, exact resource kind,
invalid startup/PID/deadline/peak values, unsafe receipts, caller-mask restoration,
8 MiB partial retention, full local storage/collection round trip and copied
array/profile verification. The CLI refuses CUDA and university hosts. No old
metadata/raw-extraction suite was rerun; the 78-check policy result is preserved,
not added to the 61 integration checks.

Independent saved-only review rechecks prior hashes and target descriptors,
compares normalized core/wrapper/CLI ASTs, verifies all unique CPU return/reference
arrays, exact replay, failure markers, copied collection bytes/modes/nsmtime and
synthetic resource refusals. New model/autograd/SDPA/CUDA/profiling/process/network/
reference/extraction execution is blocked during that review. Raw test artifacts,
source snapshots and reports are retained and sealed.

The local runtime is PyTorch 2.14.1 CPU, not the university's PyTorch 2.8/CUDA build.
Actual target compatibility, full fresh component completion/replay, capacity and
speed still require a separately prepared and authorized target handoff. The full
model candidate remains rejected, research/full training remains unadmitted and
financial caps/holds/pending reconciliation stay unchanged. No university SSH,
upload, retrieval, cleanup, installation, paid compute or held-out pixels occur.

Next prepare a separate university handoff from the unchanged accepted native
capsule and this v6 worker. Bind the policy and v6 resource receipt explicitly,
including any cold standard-library receipt adapter. Verify exact payload/imports,
unused authorization, live guards, storage, serial deadlines and same-SSH collection
before considering one new scoped university attempt. Never reuse old helpers,
consumed grants/claims, retrieval connections or time allowances.
