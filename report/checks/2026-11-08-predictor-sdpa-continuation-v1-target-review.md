# Startup and original values verified; backend association refused — November 8

The component-only attempt is consumed. It exits 1 but its same-SSH collection
completes: 152,863,158 response bytes, 179 regular members and 152,685,099 logical
bytes independently verify, including all file hashes, private modes and nanosecond
modification times. SSH stderr is empty. The 29 overlay files, all 118 capsule
inputs, 10,718 preparation bindings and 30,568 prior hashes remain unchanged.
No reconnect, retrieval or new target work is needed for this review.

## What ran and passed

The full 2.5 GiB allocation/write/fsync/readback proof passes. The original
accepted native capture is reused unchanged; there is no new full-model backward
or native capture. Session work ends after about 29.21 seconds within the
15-minute allowance. Only original component-0 starts; the other three do not run.

The startup correction works on the actual target. Its receipt records one
configuration before CUDA initialization, the original runtime/precision flags,
and the unchanged 7 GiB allocator cap. The duplicate-configuration refusal is
gone. This is not a full capacity or performance qualification: the final peak
allocator/resource receipt is absent because worker completion was not reached.

Component-0 builds one canonical FP64 reference. All four reference arrays and
all eight actual untimed/profiled FP32 arrays verify against their saved headers,
bytes, shape, finiteness and logical hashes. The original output/dQ/dK/dV match
the actual native boundary bitwise; the profiled iteration replays the untimed
values exactly. There are zero failed elements under every original bound.

| Return | Maximum absolute error against saved FP64 reference | Maximum tolerance ratio | Failed elements |
| --- | ---: | ---: | ---: |
| Output | 1.6001e-5 | 0.55418 | 0 |
| dQ | 1.0339e-11 | 0.00005160 | 0 |
| dK | 3.4827e-11 | 0.00016256 | 0 |
| dV | 1.5785e-11 | 0.00007776 | 0 |

These numerical observations do not complete the worker or the four-cell study.
Fresh-process original replay and both math workers remain unexecuted.

## Why the worker stops

The unchanged backend verifier rejects:
`Unrecognized/mixed or missing actual component backward backend; no fallback.`

All 106 saved profiler records verify. The efficient forward operator lies under
the thread-1 forward user label. The actual efficient backward operator instead
belongs to a thread-2 autograd subtree whose parent is not the thread-1 backward
user label. The current verifier searches only CPU-parent descendants of that
label, excludes the backward operator and refuses before qualifying its backend.

The trace contains the efficient forward/backward operator names and positive
linked device-0 CUTLASS forward/backward kernels. The autograd engine/node record
forward-thread 1 and sequence 1, matching the selected forward operator, and its
host subtree falls within the backward label's host interval. This is evidence
for a separately reviewed cross-thread association rule; the existing rejection
is preserved. No events are relabeled, no parser is bypassed, no fallback is
forced and no backend is newly accepted in this review. Asynchronous CUDA kernel
intervals are not assumed to fit inside CPU-label intervals.

The failed worker remains incomplete and has no backend qualification or final
resource receipt. Its final flags/input completion checks were not reached;
their false fields are not relabeled as successful. Profiling durations do not
qualify speed/cost, isolated backward arithmetic or a research policy.

## Saved review and next step

Review blocks model/autograd/SDPA/CUDA/process/network and reference execution.
It loads saved arrays and recomputes their original elementwise acceptance gates;
it does not rebuild the full native FP64 reference or rerun any scientific test.
Only an in-memory source/input path view is translated for local loaders; stored
plan bytes and identity stay unchanged.

The first review/source and refusal are retained. Exact equality of an aggregate
relative-L2 diagnostic differed by 8.47e-22 between CPU reductions. The second
review verifies every elementwise-gate field exactly and reports both diagnostic
summaries separately. Relative L2 is not an acceptance gate. No original
numerical bound or target output changes.

Next prepare a separate **offline** event-association correction using this saved
trace, with strict tests for ambiguous or mismatched thread/sequence identities,
foreign or unlinked kernels and invalid parent graphs. Keep the failure and all
original numerical/backend/fresh-replay requirements. Current scientific sources,
operators, helpers, authorization, consumed claim, evidence and clocks are frozen.
No new target attempt is authorized; never repeat/reset the consumed helper.
The full-model candidate's 3,724-entry rejection, training gates and all financial
caps/holds/reconciliation remain unchanged.

[Machine-readable target review](2026-11-08-predictor-sdpa-continuation-v1-target-review.json).
