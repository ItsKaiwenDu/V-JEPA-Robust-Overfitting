# Component tensors pass; exact summary comparison stops the run

The approved v5 attempt is consumed. Its same-SSH download is complete: all 189
files, 152,793,107 logical bytes, hashes, private modes and nanosecond modification
times verify against the 152,977,694-byte response. Transport exit is 1 with no
send errors or SSH stderr. The claim SHA256 is
`8e5614d40fd19cf2fd35861e19a86eb90251cd113ef23b144db9a0682d070662`.
The full 2.5 GiB write/fsync/readback proof passes. The session takes 56.98 seconds
within its original 15-minute work allowance; no retry or paid compute occurs.

The first original component worker exits 0 and publishes its complete marker,
startup, final resource and restored-flags/unchanged-input receipts. The parent
reader then rejects it with `Actual saved component/reference numerical bound
rejected.` Its task list remains empty, comparison incomplete and overall exit 1.
The math worker and both fresh-process replays do not run. Producer completion is
preserved separately from parent admission; the whole attempt remains failed.

Independent saved-only review verifies all eight actual FP32 return arrays and
four canonical FP64 reference arrays. Every original elementwise bound passes
with zero failed elements. Both untimed and profiled logical bytes match the
accepted native component exactly. The largest output tolerance ratio is 0.55418,
below 1; original output/gradient tolerances and bookkeeping bounds stay unchanged.
All recorded acceptance fields, shapes, maximum errors and tolerance ratios match
the recomputed values exactly.

Only the diagnostic `relative_l2_error` fields differ in the local reproduction:
each saved recomputation differs from its recorded summary by about 8.47e-22.
For example, the recorded output value is `1.452763659319288e-06`; saved-array
recomputation yields `1.452763659319289e-06`. The frozen reader requires exact
whole-dictionary equality as well as passing elementwise bounds, so it reproduces
the same recorded error. No metric or tolerance is rewritten to admit this run.
The remote parent's recomputed dictionary was not serialized; its exact scalar
difference and any thread/library reduction mechanism cannot be proven from the
saved records. The isolated local mismatch is evidence for the next reader fix,
not a retroactive passing result.

The profiler metadata correction works on this actual saved trace. All 106 raw
records remain intact, including overhead row 10 and GPU annotation row 20.
The metadata proof recomputes exactly with 12 distinct positive CUDA kernel
witnesses and the observed efficient backend family. Annotations do not count
as kernels. The proof stays metadata-only, without adopting the complete cohort,
qualifying speed or inferring a fresh replay. The final first-worker resource
receipt verifies a 50,534,400-byte allocator peak within the unchanged 7 GiB cap.
GPU samples show the expected device and only its owned worker, while retaining
the explicit limitation that sampling does not prove continuous exclusivity.

Review blocks new model/autograd/SDPA/CUDA/profiling/process/network/reference
execution. It reverifies 7,327 sealed preparation files, 16 armed inputs, 67,341
prior hashes, all 38 overlay files, 118 capsule members and five older consumed
claims. All 195 current target-file descriptors are preserved. Scientific sources,
saved plans, original records, modes and timestamps remain unchanged. No tests or
scientific runs repeat, and no source correction occurs during this review.
Earlier incomplete workers and all old claims remain untouched. Candidate rejection,
full-training gates, financial caps, holds and pending reconciliation remain.

Next design and test a separately versioned receipt comparison offline. It should
preserve strict elementwise acceptance, source/runtime/input/backend/replay gates
and explicit validation of diagnostic bookkeeping without relying on bitwise
whole-dictionary equality. No new university attempt, retrieval connection,
cleanup or installation is authorized. Never rerun this consumed helper.
