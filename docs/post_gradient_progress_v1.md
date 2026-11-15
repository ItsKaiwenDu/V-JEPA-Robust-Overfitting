# Local post-gradient progress prototype

The preceding pretrained PGD attempt saved its first original input gradient,
then exceeded its worker deadline before final proof/backend publication. The
native backward returned. The specific later call and the SSH failure cause
remain unknown. Partial array descriptors have no scientific admission.

This version adds diagnostic boundaries without executing another model or
modifying the consumed worker. Its only runnable checks use opaque objects,
fake callbacks and a synthetic process exit. No tensors, actual profiler, GPU,
SSH, storage proof or paid work run here. It supplies no target launcher.

## Implementation

`models/post_gradient_progress_v1.py` writes one new private, exclusive-create
JSONL journal per objective. Each before/after row carries the control/objective/
creator identity, sequence, monotonic interval and chained checksum. A before row
is fsynced before its callback. The whole file is at most 64 KiB, 64 rows and
1,024 bytes per row. Directories require mode 0700; files require 0600 and one
link. The writer detects changed inodes, modes, links and committed prefixes.
It preserves incomplete writes and refuses reuse, repair and automatic retries.
Fsync is a durability request to the filesystem, not proof against every storage
or host failure. A reader checks stable bounded bytes and preserves a partial
tail. It cannot determine whether a final visible line's fsync returned.

The stages are gradient checks, provisional metadata snapshot, prediction-hook
removal, native route exit, profiler stop, restoration checks, input restoration,
native proof, event materialization, aggregation, backend save/validation and
final publication. A before record without its end identifies an observed
interval; it does not establish a deadlock, its cause or a performance result.
All progress records have `scientific_qualified: false`, including a fully closed
journal. They never replace the original numerical/native/backend receipts.

The named ExitStack keeps original LIFO cleanup and exception/suppression behavior.
A logging error cannot skip mandatory registered cleanup or failed-result
publication. Cleanup and publication are single use. Passing publication has
no logging-failure fallback and requires every relevant gate to return without
an observed error. Callback return values are passed through without inspection,
conversion, device transfer or copying.

After the original post-save gradient checks, the first profiled original/
candidate pair can preserve actual native calls, VJP calls and scope metadata
before profiler teardown. Distinct `provisional-*` names, a new kind and explicit
incomplete/restoration-unchecked fields prevent these copies from serving as
passing native proofs. Existing 4,096/384 row and 2,048/8,192-byte row limits remain;
scope/header limits are 64/16 KiB. They contain metadata only, never new arrays.
Partial snapshots stay partial if writing fails; no old file is replaced.

`models/predictor_pgd_progress_adapter_v1.py` builds an **unbound class AST** from
the exact frozen worker and records hashes. It does not import or run either.
Reversing its eleven declared observation edits reconstructs the original
Capture class exactly. Original gradient calls, comparisons, selection order,
model/parameter/target/mask checks, flags/modes/private methods/hooks/`.grad`,
native root/checkpoint/inner-task proofs and success placement remain. Splitting
profiler event materialization from aggregation reconstructs the exact original
profiler body and its ancestry/kernel predicates. No complete worker, CLI,
operator, authorization or transport is produced.

The existing one-million-event cap applies **after** `list(profile.events())`.
The prototype retains it; the cap cannot bound the preceding parse/materialization
allocation or stop latency. No old deadline is increased.

## Profiling follow-up review

| Option | What it resolves | Limitation / work required |
| --- | --- | --- |
| Keep full EOT8 profiling with the new checkpoints | Separates cleanup, stop, parse, aggregation and publication intervals if a later attempt reaches them | Keeps the unbounded preceding materialization risk; this is the prototype's unchanged scope, not a new launch decision |
| Separate numerical capture from a representative detailed profile | Avoids making a large detailed trace a prerequisite for retaining provisional numerical/native observations | Needs a new explicit profile identity, selected scope, event/count predicates, failure handling and resource accounting; no observer is silently disabled here |

The preferred design to review next is the second option. Keep all original EOT8
numerical and ownership checks. Define a separate representative profile at the
original geometry, with actual efficient-forward/backward and public-math CUDA
kernel ancestry. An explicitly selected two-family draw or saved-input component
scope could reduce tracing, but neither establishes full-EOT8 backend coverage
or end-to-end performance. Its expected counts must be derived for the selected
scope; the old 1,152-forward/576-or-384-backward/192-math full-EOT8 validator cannot
be applied to a smaller trace or relaxed to make one pass. No extra objectives,
replays or warmups are authorized by this discussion.

Before preparing a target attempt, review whether a numerical capture and a
separate representative backend gate can satisfy the intended claim. Then define
new typed inventory entries and a combined storage/time budget, simulate the
actual integration and review one exact scope. This local prototype alone does
not resolve that design or make a target grant ready.

## Bounds and verification

Eight maximum journals plus two maximum provisional snapshots add at most
23,756,800 logical bytes (22.65625 MiB). This is **additional** to the old files.
It is not proof that the combined capture fits the shared 64 MiB metadata or
2.25 GiB capture allowance. Existing target inventory allowlists do not admit
these new filenames; a future new version must review them explicitly. No old
inventory, claim, source, capture ceiling or timeout is widened or reset.

69 new checks passed in one run with zero failures/skips. They exercise actual
transformed gradient/exit methods using fake callbacks, every named failure
boundary, all publication gates, LIFO/suppression, cleanup after journal errors,
short writes/fsync errors, identity/link/prefix changes, malformed chains,
provisional bounds, opaque returns, source reversibility and the unchanged
post-materialization cap. A local standard-library child exits intentionally to
check preservation of an unreturned phase; it starts no scientific worker.
Existing passing tests and scientific cohorts were not rerun.

The raw stage is `runs/pretrained/diagnostic/predictor-finalization-observer-v1-20261115/`.
It retains initial sources, the single JUnit/log result, labeled simulation
journals and the unbound source artifact. The small tracked report and local-only
contract summarize the review. The earlier GPU timeout, partial metadata,
consumed grants, numerical rejections, closed successful cohorts, financial
caps/holds/reconciliation and training gates remain unchanged.
