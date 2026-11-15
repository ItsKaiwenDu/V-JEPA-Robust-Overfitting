The metadata read completed and confirms an incomplete technical attempt. The
storage proof passed, the original input gradient was saved, no candidate gradient
started, and the saved worker PID was absent when inspected. Both the original
experiment claim and the metadata-read claim are consumed. Do not repeat either
helper, reset either clock/claim, or start another connection under those grants.

The first original objective recorded all 37 array descriptors: pixels, clean
targets, masks, loss, 16 predictions, 16 native-L1 cotangents and the input gradient.
Its 12,583,040-byte input.npy was saved before the supervisor timeout. Source order
writes that file only after the native gradient call returns. This corrects the
initial preliminary interpretation that backward itself timed out.

No native-call/scope proof, success/failure marker or backend aggregate was
published. The last persisted events remain forward/backward, but that does NOT
pinpoint a hang inside backward or ExitStack.__exit__: exit is appended only in
memory and the next publication occurs after cleanup, restoration checks, native
proof validation and profiler event processing. The exact later subcall is
unknown. A possible post-save check/error path, hook removal, native route exit,
profiler stop, model/target checks, event materialization/aggregation or final
publication cannot be separated from these existing records.

The worker supervisor recorded exit -14 (SIGALRM) and a phase-deadline timeout;
its wall record is 1,942.58 seconds against the 1,920-second scientific-worker
allowance. The parent recorded 1,954.31 seconds and incomplete/failed status. The
metadata's saved-file timestamps place the input/last objective record well before
that final supervisor record. These are diagnostic chronology, not GPU timing or
performance units. The reason the SSH connection lost responsiveness is still
unknown. No out-of-memory, competing-compute or disk-limit diagnosis is established.

Storage evidence is positive: full 2.5 GiB write/fsync/readback proof passed and
removed its own file in about 9.72 seconds. The last worker storage sample reports
275,878,254 logical bytes and 276,353,024 allocated bytes, below the 2.25 GiB capture
ceiling; none of 179 observed file entries exceeded its consumed typed limit.
This does not establish where CPU/GPU/profiler processing stalled. Peak GPU/RSS
and final restoration/native/backend results were not published.

The new read returned a 99,646-byte JSON response, 39,428 bytes of selected metadata,
zero SSH stderr and a missing saved worker PID 3891454. Its original scope/control,
claim, source/key bindings, base64 payload hashes and transmitted file metadata
were checked locally. No numerical array, archive, binary proof or cache contents
were read or downloaded; the arrays' existence/descriptors are metadata evidence,
not independently verified numerical/ownership qualification. No candidate
rejection or pass occurred. Prior CPU and clean-EOT8 results remain unchanged.

Local source review found an observability gap: profiler stop and later processing
precede durable proof/completion publication. The event-count ceiling is checked
after profile.events() is materialized, so it does not bound materialization
itself. Also, the older detailed full-model profiler observed one two-family draw;
this new attempt profiled the complete EOT8 pair. These are concrete design risks
and next investigation points, not proof of the root cause.

Next LOCAL ONLY: use a separate artifact version with bounded durable progress
checkpoints before/after hook removal, native exit, profiler stop, restoration
checks, event materialization and publication. Preserve all native/numerical/
restoration gates; do not move a passing marker ahead of them. Review separation
of numerical evidence preservation from a smaller representative detailed-profile
scope. Test sequencing with simulated callbacks only. Do not repeat a successful
CPU/clean cohort, silently remove observers/guards for speed, infer performance,
extend the consumed timeout, or prepare a new GPU/retrieval approval before that
local design is concrete and reviewed. No new remote request is ready here.

All existing source/evidence and financial caps, holds and pending reconciliation
remain unchanged. No new model, autograd/SDPA/profiler execution, GPU, proof, SSH,
cleanup or paid work occurred during this saved-metadata review. Full training,
research calibration, iterated-candidate numerical/backend/replay/performance and
full-study cost/adoption gates remain open.
