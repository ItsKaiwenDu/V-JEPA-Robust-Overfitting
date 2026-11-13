# Component continuation v3: numerical values pass; raw profiler metadata refuses

The attended attempt exits 1 after the first original component. Collection is
complete: 152,884,429 response bytes, 183 regular members and 152,699,689 logical
bytes verify with exact SHA256, private modes and nanosecond modification times.
There are no send errors or saved SSH stderr bytes. The failure is in the raw
profiler validator, not the transport. The attempt is consumed; do not rerun it.

Independent saved-data review verifies all eight actual FP32 return arrays and
four canonical FP64 reference arrays. Original elementwise output/dQ/dK/dV bounds
pass with zero failed entries. Both untimed and profiled saved returns match the
accepted native logical bytes exactly. One canonical reference was built by the
first fresh component process. Recorded/local relative-L2 diagnostics are kept
separately across platforms and are not used to change the acceptance bounds.

Strict configure-once CUDA startup passes with the original precision identity
and 7 GiB allocator setting. Saved runtime and actual input metadata match the
frozen native layout/alias/offset/logical-byte requirements. The full 2.5 GiB
allocation/write/fsync/readback proof passes. Session work is 38.17 seconds;
the component's own recorded work is 8.29 seconds. GPU samples show the expected
host/device and only the owned component process; sampling is not continuous
exclusivity proof. No new full-model backward or native capture occurs.

After the profiled return arrays are saved, raw Kineto extraction reaches
`project(trace)`, which raises `Invalid raw event fields or legacy CUDA fallback.`
The frozen collector validates before returning the envelope. The frozen worker
publishes the envelope only after that return. Consequently there is no saved
raw trace, including no partial `events.jsonl`, and the available generic exception
cannot identify the exact offending row or field. No particular sentinel, API
behavior, correlation identity or backend diagnosis is inferred. The real refusal
cannot be replayed locally without its missing raw records; they are not invented.

The independent comparison of saved profiled arrays does not alter the worker's
completion flags. Its `profile_replay_exact`, `flags_restored` and `inputs_unchanged`
remain false because later checks were not reached. There is no backend proof,
completed worker marker or final measured GPU-peak resource receipt. Components
1, 2 and 3 never start; no fresh-process replay, candidate component acceptance,
backend/capacity/speed qualification or full-training admission follows.

The exact 34 overlays and 118 capsule files, 13,951 preparation files, 15 armed
bindings and 47,042 prior hashes all reverify. The final control, private
authorization, old consumed claims and their modification times remain unchanged.
The current consumed claim SHA256 is
`95a15a08e35721a7840e26915544b5599d3656635b2527162b3ec595236770b0`.
Current plan paths are translated only in memory for saved-data verification;
stored plan bytes and timestamps remain unchanged. All target input descriptors
and current source/evidence hashes are sealed.

The review blocks model, autograd, SDPA, CUDA, profiling, process, network and
reference execution. No tests or experiment repeat, no frozen source correction,
no SSH/retrieval/cleanup, and no paid compute occurs. The candidate's 3,724-entry
rejection, original numerical/backend/replay gates, financial caps/holds and pending
billing reconciliation remain unchanged. No files were staged, committed or pushed.

Next prepare and test a separately versioned bounded capture-before-validation
path offline, so a future refused trace retains its actual metadata. Keep the
current validator and scientific/backend rules strict. Preserve this failed run
and all earlier claims/clocks. No target retry, retrieval, cleanup, fallback or
new university attempt is authorized.
