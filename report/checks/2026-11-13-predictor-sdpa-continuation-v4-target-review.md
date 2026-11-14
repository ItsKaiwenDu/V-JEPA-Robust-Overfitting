# Retained university trace identifies the first raw-field rejection

The attended attempt exits 1 after the first original component. The same-SSH
collection is complete: 152,936,174 response bytes contain 186 regular files and
152,749,518 logical bytes. SHA256 hashes, private modes and nanosecond modification
times all verify against both the collection header and tar contents. There are
no send errors or SSH stderr bytes. This is a semantic validation failure; the
attempt is consumed and must not be repeated.

Independent saved-data review verifies eight actual FP32 output/gradient arrays
and four canonical FP64 reference arrays. Original elementwise output/dQ/dK/dV
bounds pass with zero failed entries. Both untimed and profiled returns match the
accepted native logical bytes exactly. Relative-L2 diagnostics are retained
separately across platforms; no acceptance bound changes. One canonical reference
was built. Strict configure-once CUDA startup, original precision/runtime identity,
actual input layouts/aliases/offsets/logical hashes and the 7 GiB allocator setting
verify. The full 2.5 GiB allocation/write/fsync/readback proof passes; it is not a
reservation against other writers or a kernel-enforced aggregate quota. Session
work is 35.81 seconds; recorded component work is 5.03 seconds. GPU samples match
the expected device and only the owned process, with no claim of continuous
exclusivity. No new full-model backward or native input capture occurs.

The capture-before-validation change works on the actual target. All 106 raw
Kineto records are saved and hash-bound before the frozen semantic check, in the
39,241-byte `events.jsonl` envelope (SHA256
`d1b2006c583c6cfc4d8dc0bb0b12e823293222079c740d6d78acb78a72382686`).
The exact saved envelope reproduces the original exception. A separate read-only
scan of every basic raw-field predicate identifies one violation: record index 10,
`Activity Buffer Request`, has `device = cpu` and `device_index = -1`. The frozen
validator requires every device index to be nonnegative. Other basic field/type/
range/name/interval checks pass. Later CPU ownership, correlation and backend
checks were not evaluated after editing or filtering this record; their outcome
is unknown. All original rows remain unchanged.

The official PyTorch v2.8.0 tag pins Kineto commit
`5e7501833f1021ce6f618572d3baf657b6319658`. Its CUDA-12.7-or-later overhead-name
mapping includes this event, and its overhead wrapper explicitly returns device
ID -1. PyTorch carries that activity device ID into the raw event device index.
This supports an overhead interpretation of the observed pair, rather than
proving the exact installed wheel source revision or the native activity kind,
which the current envelope does not record.
[PyTorch gitlink](https://github.com/pytorch/pytorch/tree/v2.8.0/third_party/kineto),
[overhead-name mapping](https://github.com/pytorch/kineto/blob/5e7501833f1021ce6f618572d3baf657b6319658/libkineto/src/cupti_strings.cpp#L67),
[overhead device representation](https://github.com/pytorch/kineto/blob/5e7501833f1021ce6f618572d3baf657b6319658/libkineto/src/CuptiActivity.h#L157),
[activity conversion](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/profiler/collection.cpp#L1021),
[raw device-index accessor](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/profiler_kineto.cpp#L985).
Saved primary-source receipts and a separate standard-library review preserve
this source finding. Four full source blobs also match their upstream Git blob
hashes; the name-mapping receipt is explicitly a bounded line excerpt.

The worker remains incomplete. `profile_replay_exact`, `flags_restored` and
`inputs_unchanged` stay false because the later checks were not reached, despite
independent equality of the saved profiled arrays. There is no accepted backend
proof, completed marker or final measured GPU-peak receipt. Components 1/2/3 do
not run. No fresh-process replay, candidate/backend/capacity/speed qualification
or full-training admission follows. The prior attempt without a saved trace
still has no identifiable offending row; this new observation does not recover it.

All 36 overlays, 118 capsule inputs, 4,915 sealed preparation files, 15 armed
bindings and 61,648 prior hashes reverify. Current consumed claim SHA256 is
`3829d1b634b32053f73560610f2e78d5a835e7193eaea885b55211e6c2536c32`.
All four older lineage claims, byte identities and modification times remain
unchanged. Stored plans are unchanged; review translates paths only in memory.
The 192 target input descriptors and all review/source receipts are sealed.

Saved-data review blocks model/autograd/SDPA/CUDA/profiling/process/network/
reference/extraction execution. No tests, model work, SSH/retrieval, cleanup,
installation, frozen source correction or paid compute occurs. The preceding
79- and 82-test results are preserved, not rerun or accumulated. The 3,724-entry
candidate rejection, original numerical/backend/replay gates, financial caps,
holds and pending reconciliation remain. Nothing is staged, committed or pushed.

Next design and test a separately versioned metadata contract offline that can
represent profiler overhead separately from frontend CPU ownership and CUDA
launches, while retaining every original raw row. Do not blanket-allow negative
IDs, discard rows by name, or infer later backend success. Keep strict ownership,
correlation, startup, numerical and replay checks. This review authorizes no
new university attempt, retry, retrieval or cleanup.
