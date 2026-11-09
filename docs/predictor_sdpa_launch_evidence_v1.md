# Raw profiler launch-correlation evidence v1

This separate metadata collector/verifier preserves raw Kineto correlation fields
rather than identifying CUDA rows by kernel name/device/duration. It reads an
already-completed profiler result. It never starts a profiler, model, derivative,
SDPA call, CUDA operation, process or university connection. The frozen v1/v2
collectors, association rules, scientific workers, refusals, claims and clocks
are unchanged. No new worker, target handoff or authorization is prepared here.

## Recorded identity and provenance

The adapter reads `profile.profiler.kineto_results`. Each complete raw record
retains its index, name, device/resource, integer start/end nanoseconds, start/end
threads, forward thread, sequence, scope, annotation/async flags, correlation ID,
linked correlation ID and legacy-fallback indicator. It records the trace origin
and caller-supplied PyTorch version. Unknown fields/APIs/devices, legacy CUDA
fallback, malformed types, duplicate records and oversize input refuse; there
is no processed-FunctionEvent fallback or truncation. Limits are 20,000 records
and 8 MiB serialized, matching the existing trace-file ceiling.

PyTorch 2.8 uses a device event's linked correlation ID to associate it with a
frontend CPU event's correlation ID. These fields and integer nanosecond clocks
are exposed by its bindings.
[Versioned association source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/profiler.py#L558-L633),
[versioned raw-event bindings](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/init.cpp#L227-L279).
The adapter uses those exact read APIs. They are internal APIs: a future target
must expose and validate them under the existing exact runtime guard. Passing
local tests does not establish target compatibility.

A CPU owner must have one unique positive frontend correlation ID in one recorded
process. A positive CUDA row must carry an explicit link to such an owner on a
stable CPU thread. Device 0, non-annotation activity and positive duration remain
required. Each device row is identified within the complete capture by its raw
record index and retained provenance. A device correlation ID is **not assumed
to be globally unique for each individual kernel**. Multiple kernels may belong
to one CPU operation; the verifier keeps their separate rows and links.

## CPU ancestry and backend evidence

A separate deterministic view derives synchronous CPU containment from integer
host intervals. Linked runtime-event threads follow the recorded CPU owner, as
in the pinned PyTorch association code; the original raw thread remains stored.
Indistinguishable CPU intervals refuse instead of choosing an arbitrary parent.
GPU timing is never used to choose its CPU owner, and GPU execution may outlive
CPU labels. No nearest-time or equal-duration matching is used.

Raw duplicate-name CPU nodes remain present with their own IDs. This avoids losing
an inner kernel owner when processed FunctionEvents remove duplicate nodes and
lift kernels to their parent.
[Versioned duplicate-node processing](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/profiler_util.py#L44-L57).

The frozen v2 thread/sequence/autograd producer-chain rule is reused unchanged on
the derived CPU view. Supported families, phase membership, required efficient
or math forward/backward operations and kernel families, device checks and the
no-fallback policy remain. Every qualifying positive kernel link must identify
its actual raw CUDA row and recorded CPU owner. The returned proof is metadata
only: it cannot infer worker completion, actual CUDA execution admission, final
resource/flags/input checks, fresh replay, full-model adoption, speed/cost or
research admission. Hash-bound provenance is still necessary; structurally valid
fabricated metadata cannot be authenticated by a parser alone.

## Validation and scope

54 focused checks pass. They cover equal-name/device/duration launches with
separate explicit owners; parent-contained and cross-thread efficient/math/CPU
flash cases; duplicate CPU wrappers and raw ordering; async/foreign/missing/
ambiguous IDs; malformed clocks and fields; unchanged family/selection refusals;
bounded APIs; and cold standard-library assessment with scientific imports,
processes and network blocked. CUDA fixtures are handwritten metadata, not new
CUDA measurements or IDs recovered from historical files.

One tiny actual local CPU probe records an eight-element allocation/addition,
without SDPA, autograd/backward or a model. It checks the adapter against the
installed PyTorch 2.14.1 API. This is not the university's PyTorch 2.8/CUDA runtime
and does not qualify it. No old scientific suite or university work repeats.

The first run's 46 passes and two failed assertions remain with its sources/logs.
One asserted exact floating microsecond subtraction instead of the existing
rounded nanosecond representation. The other expected a raw-schema error from
old JSONL, which is rejected earlier as multiple JSON documents. Only those test
assertions changed; six further metadata regressions produce the final 54 passes.
Counts are not accumulated and no scientific tolerance changed.

The real 106-row university collision trace remains byte-identical. The frozen
v2 verifier reproduces its ambiguity refusal; the new raw format refuses it for
missing IDs. The consumed claim and incomplete worker are never relabeled.
Independent saved-only review rechecks preserved hashes, actual collision rows,
synthetic ID witnesses, the CPU adapter trace, and cold assessment provenance
without importing scientific libraries or running profiles/processes/network.

Next integrate this separate collector/verifier into a new component worker and
test the complete startup, numerical, profiling, storage and receipt paths before
preparing any new university scope. Full training, the rejected candidate,
financial caps/holds and pending reconciliation remain unchanged.
