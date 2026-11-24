# Snapshot ownership, lifetime and byte requirements

**This stage defines requirements and a validation plan. It implements no
snapshot adapter and runs none of the planned fake-port cases.** Keep the
original `.tobytes()` copies, every fresh hash and the closed execution gate.

The [contract](../configs/predictor_pgd_snapshot_contract_v1.json) and
[validation plan](../configs/predictor_pgd_snapshot_contract_validation_v1.json)
derive from the frozen runtime graph and the
[completed upstream-source review](predictor_pgd_target_source_review_v1.md).
See the [saved review](../report/checks/2026-11-24-pgd-snapshot-contract-v1-local-review.md)
for preservation and audit results. No new source fetch or local inventory is
needed for this task.

## Source coverage and existing observation boundaries

The source mapper parses all48 members of the frozen481518-byte runtime manifest
from its saved source payload:47 overlay members and one supervisor member.
Only source text is base64 decoded; no packet, checkpoint or scientific array
is decoded. The map records27 lexical hash calls, two hash definitions and27
matching symbol references. Ten calls are in unit preparation and the unit
module. The other17 are state-digest calls in the frozen evaluation, diagnostic
or training sources. Their presence supplies no reachability, dispatch,
execution-count or training-admission proof. Dynamic aliases outside these
matching name/attribute spellings are not resolved.

All29 input pins, the13 inherited upstream source claims and four original AST
anchors are retained. The
[complete call-site map](../runs/pretrained/diagnostic/pgd-snapshot-contract-v1-20261124/HASH-CALLSITES-v1.json)
includes source hashes, line/column, containing scope, expression and call AST
hash. Each of the27 calls has a recipe and requirements mapping.

| Call | Existing boundary to preserve |
| --- | --- |
| CALL-001 | Ordered model state digest during preparation |
| CALL-002 | Clean-pixel baseline during preparation |
| CALL-015 | Fixed-value registration for parameters, buffers, clean input, targets and masks |
| CALL-016 | Fresh fixed-value observation whenever the original hash expression is reached |
| CALL-017 | Score pixel hash after score computation, before score-record append |
| CALL-018 | New input-leaf hash before durable step-start publication |
| CALL-019 | Context-entry input baseline before the local cleanup try/objective timer |
| CALL-020 | Repeated exact input-token checks around consider/backward |
| CALL-021 | Checked gradient hash before successful gradient-row completion |
| CALL-022 | Final adversarial hash after evaluation/restoration checks and before terminal publication |

Earlier metadata/ownership refusals can short-circuit a `Snapshot.check` before
its hash expression. Preserve that ordering. Freshness means every reached
original hash performs a fresh observation; it does not add hashing after an
earlier refusal. Equal object/version metadata never replaces a reached hash.

## Two independent byte contracts

`tensor_hash` detaches, transfers to CPU, obtains logical contiguous layout,
flattens with reshape(-1), reinterprets the bytes as uint8, exports to NumPy,
materializes default C-order bytes, and returns the SHA256 hex digest. A future
proposal must preserve logical extent/order, offsets, dtype bits and all
original accepted/refused cases. Hashing storage capacity/padding or numerically
converting values to uint8 changes the contract.

`state_digest` instead walks `state_dict()` in its original order. It updates
SHA256 first with each name's encoded bytes, then with that tensor's detached,
CPU, contiguous NumPy bytes; it returns the final hex digest. Add no delimiter,
name sorting or metadata. Do not replace this path with `tensor_hash` or assume
their dtype/refusal domains coincide. Actual domain equivalence remains a
separate proof obligation, including empty/scalar/offset/noncontiguous and
unsupported dtype/layout/device/subclass/conjugate/negative cases. No such
tensor case is constructed or executed here.

## Ownership and lifetime obligations

Fourteen requirements cover source/build provenance; original domain/refusals;
fresh observations; source-writer ordering; private storage; transfer completion;
the two byte contracts; exporter containment; bounded lifetime; primary-error
preservation; original guards/clocks; complete cost/memory accounting; and
separate operational admission. Each remains a requirement, without target
ownership or performance qualification.

An immutable byte string owns its result after copying. This does not prove
that concurrent source writers were excluded during capture. The baseline is
unchanged and is not newly certified as an atomic snapshot. Any future proposal
needs both capture ordering and independence of the consumed bytes. CPU/GPU
labels, copy requests, readonly flags, object inequality and version counters
cannot establish those facts by themselves.

A future capture must keep storage/exporter/view dependencies inside trusted
consumption, retain them through all confirmed byte uses, and exclude surviving
writable aliases or reentrant mutation/release paths. Completion must match the
owner, allocation generation and stream/domain. Completing one current stream
does not establish ordering for unrelated writers. Unknown completion remains
unknown; dropping a reference is not completion evidence.

| Conceptual state | Required condition |
| --- | --- |
| Unqualified → capture pending | Provenance/domain and the original call boundary are established; capture has explicit ownership/order obligations |
| Captured → export held | Ordered capture is complete; logical bytes are private; the full exporter dependency chain is retained |
| Consuming → consumed | Trusted consumption has no writer/escape/release path; all reads finish before eligibility for success |
| Released | Only owned, confirmed-unused observation resources are released; cleanup succeeds before returning a success digest |
| Failed held → failed released | No success result; retain resources with unconfirmed uses, preserve the primary error, release only after quiescence; no retry |

These nine conceptual states are requirements terminology, not an implemented
monitor or a selected constructor/API. Caller-owned model/tensor resources are
never released by a proposed hashing helper. No cache, retained model, extra
timer allowance or unbounded buffer lifetime is introduced.

## Failure boundaries that require explicit treatment

The frozen `OwnedStep.__enter__` sets its active owner and computes CALL-019's
input hash at line316, before its local `try` at317 and objective timer creation
at319. A failure there cannot be assumed to enter that handler. This is a
source-order finding; no failure, leak or timer behavior is reproduced or proven.
A future capture must define ownership from its first allocation under the
original continuing unit/context/parent clocks. Moving the hash, extending
time or adding a source bridge is not adopted by these requirements.

CALL-018 computes its hash before publishing that step's durable start; a
failure cannot invent a start record or make the already consumed unit reusable.
For CALL-017 and CALL-021, score/gradient work can already have occurred before
hashing fails. A missing hash receipt does not prove no scientific work occurred.
CALL-022 must not produce a successful terminal receipt after failure.
The [boundary notes](../runs/pretrained/diagnostic/pgd-snapshot-contract-v1-20261124/FAILURE-BOUNDARY-NOTES-v1.json)
retain the exact source and AST anchors.

All failures, cancellation, deadline and cleanup paths must preserve the primary
exception and failed/partial/consumed status. Required buffers stay held when
use is unconfirmed; original bounded shutdown/idle gates remain. Cleanup notes
cannot replace the primary failure, authorize retry or promote incomplete work.

## Explicit fake-port plan — not run

The plan declares14 abstract port responsibilities and32 cases. Ports cover
authority/domain labels, fresh call boundaries, writer ordering, private-capture
generations, completion, opaque bytes, ordered state streams, exporter scope,
retention, injected failure/cleanup, supplied clock/guard labels, live-byte
accounting and blocked real effects. These names select no tensor API.

Cases cover valid symbolic lifetimes, missing authority/ownership/order,
mismatched completion, escaping writers/exporters, premature release, failure
and cancellation at different states, primary-error preservation, unchanged
deadlines/journal boundaries, overlapping allocation lifetimes, repeated fresh
calls and separation of the two recipes. Every case is `NOT_RUN` with no saved
result. Every port is unimplemented. There are zero case executions, fake
workers and actual tensor lifecycles in this stage.

Only a separately authored local stdlib symbolic harness may be considered
next. Its plan is bounded to32 cases,4096 opaque payload bytes/case,64 symbolic
events/case,eight simultaneous symbolic allocations and1MiB total saved local
evidence. These are future harness bounds, not changes to workload caps. No
real process/thread/clock/timer/signal/PIDFD/network/GPU/proof/scientific port,
FileSink/ledger or installed budget function is invoked. Prior bytearray
witnesses, passing tests and closed scientific cohorts are not repeated.

Such a harness could check only the state model and supplied opaque byte
streams. Fake labels do not prove physical alias exclusion, actual operator
refusals, buffer-consumer behavior, stream ordering, atomicity or blocking-I/O
enforcement. Its original-domain equivalence checks remain pending actual
versioned evidence; it cannot manufacture that evidence from metadata.

## Accounting and admission

Any later proposal must include every simultaneously live source, private copy,
temporary, bytes materialization and retained exporter/base allocation. Account
for full retained storage capacity as well as logical spans; count an alias's
retained allocation once, while including each distinct copy. Fake live-byte
totals are not RSS/allocator measurements. Private copies may increase cost.

Keep original phase inclusions/exclusions, own journal/timer exit, scoring,
restoration/publication and parent wall accounting. Do not subtract overhead,
add nested intervals twice, reset clocks, extend caps or use partial gradients
as throughput. This design selects no performance solution.

Actual target Torch/build/NumPy/CPython/dispatch provenance, physical ownership,
exact bytes/refusals, complete units/fresh replay/timing/memory/recovery and
economics/calibration/data/protocol admission remain unresolved. The runtime
graph56uploads936711B/runtime48files481518B/cache47/nine supervisors/270paths,
all original scientific/ownership/hash/durability/shutdown guards and caps,
training settings and financial caps/holds remain fixed. All real grants stay
consumed. No scope/package/grant/approval request/launch command or target work
follows. Budget remains `REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE` before protocol
clock, ATTEMPT or connection. Mechanism closed; research/training false.
No training percentage or time estimate is supported.
