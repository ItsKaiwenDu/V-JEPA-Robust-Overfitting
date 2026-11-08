# Native capture accepted; component startup refused — November 7

The replacement attempt is consumed. Its same-SSH collection succeeded even
though the experiment exited 1. Independent local review verifies the complete
108,778,481-byte response, all 171 regular archive members (108,613,250 logical
bytes), saved bytes/hashes/private modes/nanosecond timestamps, all 33 overlay
sources, all 99 original packet members and 25,370 preparation bindings.
SSH stderr is empty; the captured worker traceback is preserved in its own log.
The one-use helper must never be repeated or reset.

## Accepted native capture

The 2.5 GiB write/fsync/readback proof passes. The exact 146,562-byte mask file
now passes its reviewed 256 KiB allowance; both live storage and collection
complete without the previous typed-file refusal.

The original native capture completes in 52.57 seconds of worker wall time.
Its actual saved scalar loss and full input gradient replay the independently
reviewed original native values exactly, with zero reported maximum difference
under the unchanged bounds. Local deserialization independently verifies their
logical hashes, shape, dtype, finiteness and equality with the original packet
anchors. Clean targets also match the frozen target signature. Expected hashes
are not mistaken for newly computed returns: all actual return files are read.

All eight actual Q/K/V/output/incoming-gradient/dQ/dK/dV tensors verify against
their native capture record. Input layouts, offsets and alias-group metadata
reconstruct with matching logical values. The one completed graph scope has
576 forward and 576 checkpoint-recomputed calls over 36 owners; observer/route
hooks restore, baseline/runtime/weights/masks match and model state is unchanged.
Peak recorded allocation is 5,143,749,632 bytes, below the existing 7 GiB cap.
These capture observations do not qualify full AdamW capacity, throughput,
a standalone kernel backend or a new research policy.

## Component-0 failure

Only original-0 starts. Its complete process supervisor records exit 1; the
component worker itself remains incomplete with:
`RuntimeError: Configure deterministic execution before CUDA initialization.`
It saves no component returns, performs no FP64 reference build and reaches no
numerical comparison or profile. Math-0 and both fresh replay workers never run.
Session work ends after about 75.64 seconds, within the original 25-minute bound.

The source and saved traceback identify a wrapper sequencing defect. The target
wrapper configures deterministic execution, then initializes CUDA while setting
its allocator cap, then calls the component executor. That executor configures
deterministic execution again. The strict runtime guard rejects the second call
after initialization. This is a startup refusal, not a failed component numerical
tolerance. The CPU rehearsal and wrapper ordering test missed the complete
CUDA-specific two-layer startup sequence; they did not qualify it.

The failed worker's captured-input identity and native parent match the verified
capture. Failure marker, plan hash and empty outputs verify. The saved component
plan's server paths are translated only in memory for local review; its bytes
and identity stay unchanged. The first reviewer selected the saved overlay as
its source root and correctly hit the loader's source-root guard. That source/log
is retained. A second reviewer uses the loaded, byte-identical repository source
while mapping only input paths; the original guard remains active and review
passes. No test, reference, model or target work is rerun for this correction.

## Preserved state and next task

Review blocks model/autograd/SDPA/CUDA/process/network and reference execution.
Only saved arrays are loaded and checked. Native capture is accepted; the full
four-cell comparison remains incomplete. The prior candidate rejection at 3,724
entries, original tolerances, private-forward-state caveat and all financial
caps/holds/reconciliation remain unchanged. No study training or adoption is
admitted. No CUDA/backend or performance result is claimed for component-0.

Prepare a separately versioned component-startup correction and a regression
that exercises both wrapper and executor layers, including CUDA initialization
state. Keep the strict pre-initialization guard, exact runtime/precision identity,
7 GiB allocator cap and original numerical/backend/replay requirements.
Prefer a read-only capsule of this verified native capture for a bounded
component-only continuation, avoiding another full capture. That continuation
needs concrete local integration/review and a new exact scoped university grant;
this consumed attempt grants no retry, cleanup or extra connection. All current
sources, claims, captured evidence and clocks remain frozen.
