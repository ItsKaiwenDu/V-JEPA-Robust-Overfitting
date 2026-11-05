# Proposal: complete the remaining original attention replays

**Offline proposal only.** No new operator, handoff archive, launch helper,
authorization, claim, SSH connection or scientific computation is prepared or
started. Both university hosts can stay idle. The [accepted original GPU pair](../report/checks/2026-11-04-original-attention-pair-v1-target-review.md),
its sources, complete captures, failed first offline audit and corrected review
remain immutable. Its grant is consumed.

## Question and original reference chain

Do two new original replay-1 workers reproduce the accepted replay-0 results
under the same fixed 400-token encoder-1 inputs and original execution contract?

The unchanged original parent orders four workers and selects these references:

| Worker | Execution status | Original reference |
| --- | --- | --- |
| efficient replay 0 | Accepted previous attempt; import evidence only | None |
| math replay 0 | Accepted previous attempt; import evidence only | efficient replay 0 |
| efficient replay 1 | Proposed new fresh CUDA process | Accepted efficient replay 0; exact result replay |
| math replay 1 | Proposed new fresh CUDA process | New complete, passing efficient replay 1 |

Run only the two missing CUDA workers, serially on linux10603's pinned RTX 3070.
Math replay 1 must not use efficient replay 0 as a shortcut. A failed efficient
replay 1, incomplete capture, changed imported anchor, source mismatch, foreign
compute or resource refusal stops the attempt without math, verification or retry.
The previous pair must not be rerun or counted as newly executed work.

Preserve all original worker gates: CPU FP64 bookkeeping at `1e-12/1e-14`,
output at `1e-4/2e-5`, dQ/dK/dV at `2e-3/2e-7`, nonzero/finite values, original
live Q/K/V strides, runtime/precision/flags, actual CUDA forward/backward backend
evidence, and exact six-phase within-process replay. Efficient replay 1's
original reference check is bitwise; math replay 1's original reference check
uses the unchanged component tolerances.

## Import the accepted evidence without renewing its grant

The proposed reference subset has **16 existing files / 59,132,357 bytes**:
six files per accepted worker (`authorization.json`, `worker-plan.json`,
`worker.json`, `capture.json`, `saved-audit.json`, `values.pt`), plus the accepted
raw review, pair receipt, operator result and transport-verification record.
Both value files are 29,487,103 bytes. The design manifest pins every source
path, size, hash and original modification time.

These are copies of already accepted records, not new workers or renewed
authorizations. Old worker deadlines, original errors/completion fields, hashes,
timestamps and process IDs stay unchanged. The new resource authorization must
be a separately named top-level grant, with its own single attempt claim.

The future client must first reverify the complete original 199-file download,
all captures and the independent accepted review. A bounded reference archive
must then be created and independently checked against the exact subset manifest;
it is **not created in this proposal**. Limit its archive and unpacked extent to
64 MiB each. Reject extra files, path aliases, links, existing targets and any
changed hash, size or time.

Extract once into fresh sibling `results/workers/encoder-1-{efficient,math}-0`
directories plus separately labelled provenance. Do not duplicate the PT files
or introduce hardlinks. Make imported files read-only, preserve their bytes and
mtime, and hash them before/after each use. New replay-1 workers occupy distinct
directories. The new receipt must explicitly label the old workers as imported
reference subsets: their original input and NPZ captures remain in the full
verified Mac evidence, rather than being falsely declared present remotely.
Final independent review must use that complete local evidence too.
Keep imported workers, newly executed workers and the CPU verifier in separate
receipt fields; never seed a new attempt's executed-task count with old workers.

The old resource grant, financial ledger, credentials, private keys, weights and
research pixels are excluded. Small old *worker* authorization descriptors are
included solely to verify provenance and expired reference edges; they confer
no new execution rights. The accepted raw review contains no financial ledger.

## Preserve return evidence and the final reconstruction gate

A separately named observer must save each new worker's actual CPU FP64 return
before its gate and all six actual GPU FP32 returns before their checks:
untimed, profiled, warmup and timed-1 through timed-3. Retain source dtype,
device, shape and stride without silent casting. Preserve nonfinite/mismatching
values for the original gate to reject; capture-write errors fail closed.
Detailed introspection follows the whole original worker. Restore hooks and the
remaining alarm without restarting elapsed time.

The frozen final summarizer also calls the original NumPy FP64 oracle once per
case. Omitting that work would omit an original scientific check. Propose **one
additional fresh CPU-only verifier process**, only after both CUDA workers and
their captures pass. It has at most 90 seconds inside the same 600-second clock,
hidden CUDA visibility, CPU-only tensor loading and no CUDA initialization or
SDPA calls. It may invoke the unchanged source-bound NumPy oracle exactly once
on the fixed encoder-1 operands. Save all four actual FP64 results, at most
16 MiB, before checking them. Require observed float64 before writing rather
than silently converting another dtype.

The verifier must reproduce the original reconstruction check with its original
argument order: fixed saved `oracle_*` values first, newly computed NumPy values
second, `rtol=1e-12, atol=1e-14`. A failed reconstruction is preserved and rejects
the cohort. No replacement reference, extra oracle warmup or tolerance change
is allowed. Thus the new attempt permits **two Torch CPU oracle calls plus one
NumPy CPU oracle call**, not the preceding pair's two-call allowance.

The same bounded verifier must check all four workers' original scientific data
conditions: fixed operands and references, matching common/hardware/runtime
contracts, both live layouts, per-worker numerical/backend/replay evidence,
finite/consistent timing records and allocator bounds. Across replay 0/1 require
exact equality of each arm's complete original artifact tensor signatures, as
the original summarizer does. Check both math/efficient comparison pairs and
their recorded reference decisions. Check actual capture bytes and phase
signatures separately from arithmetic comparisons.
The remote verifier checks imported PT data and the pinned accepted attestations
for old captures; it must not claim the omitted old NPZ files were re-read there.
Final independent Mac review re-reads those original NPZ files together with all
new captures before any combined-cohort acceptance.

Keep the recent audit correction: a CPU-exported V tensor can be packed at
`[409600,64,1024,1]` while its recorded live layout remains
`[1228800,64,3072,1]`. Check the original **live** stride record against the
fixture and check exported logical bytes independently. Do not impose a new
export-layout condition or waive the original live-layout gate.

## Honest cohort provenance and qualification

The two attempts are separated in time. A new, explicitly **cross-attempt cohort
record** must link their unchanged claims, grants, workers, source hashes and
per-attempt clocks. Never fabricate `session.json`, rewrite old completion or
timing fields, sum disconnected work into a contiguous session, invoke the old
parent against imported records, or make the old summarizer accept an invented
parent receipt.

Do not invoke or modify the original parent/summarizer. Implement a separate
combined-cohort verifier with a source-reviewed coverage map for every scientific
check in the frozen summarizer. The parent/session-envelope conditions remain
distinct and unclaimed, rather than silently removed. New tests must show that
the scientific coverage map includes its FP64 reconstruction, exact full-artifact
replay, per-replay references, backend, timing-record validity and allocator
checks. Missing evidence must refuse.

A future successful result may qualify the observed four-worker **cross-attempt
component cohort** and fresh-process replay for this fixture. It must still set
`original_parent_session_qualified=false`: this is not one uninterrupted original
parent run. Other fixtures, generator replay, full-model correctness, speed,
study cost, optimization adoption and research admission remain pending.
Every profiler row and timing observation stays; instrumented copies/I/O and
the gap between attempts prevent speed or cost claims. The old approximately
1e-10 refusal and missing historical computed arrays remain unresolved.

## Proposed new resource bounds

| Resource | Proposed bound |
| --- | --- |
| Target | linux10603 only; existing RTX 3070 UUID `GPU-e54fae90-feeb-8106-d8c0-1974fe48555f`, driver 580.178.04 |
| Environment | Existing Python 3.11.17 / Torch 2.8.0+cu128 / NumPy 2.4.6; no installation or fallback |
| Additional attempts / retries | One / zero; no prior grant or claim reset |
| Concurrent scientific processes | One; two fresh serial CUDA workers followed by one fresh CPU-only verifier |
| Work clock | At most 600 seconds including proof, uploads/imports, workers, verification and final records |
| Setup / proof | Setup freshness at most 110 seconds; full 2.5 GiB write/fsync/readback at most 60 seconds within setup |
| New workers / verifier | Each original worker at most 120 seconds; CPU verifier at most 90 seconds, all inside the continuing clock |
| New scientific calls | At most two original Torch CPU calls, twelve GPU attention iterations and one original NumPy CPU reconstruction |
| Scratch / capacity | Fresh private 0700 local-XFS `/var/tmp`; 8 GiB initial free / 1 GiB runtime floor plus inode checks |
| Observed setup / capture ceilings | Unchanged 3 GiB / 2.25 GiB |
| Raw category | **400 MiB**, explicitly increased from the preceding 320 MiB |
| Input category | **256 MiB**, explicitly increased from the preceding 128 MiB |
| Original/reference archive | Unchanged original 85,622,679-byte packet; new reference archive at most 64 MiB |
| Code / metadata / overlays | Unchanged 128 MiB / 64 MiB / 2 MiB; at most 1,024 files |
| Tensor files / original allocator | Unchanged 32 MiB per file; CPU return at most 16 MiB / GPU return at most 8 MiB; 4 GiB Torch allocator per CUDA worker |
| Collection | Same attended SSH connection, at most 600 additional seconds / 3 GiB of existing files, including every NPZ and imported reference |
| Paid work / financial changes / cleanup | None; retain remote scratch |

Raw arithmetic is 240 MiB for two new workers, 64 MiB for two imported values,
16 MiB for the new NumPy return, 32 MiB for one transient atomic-save name pair,
and 48 MiB margin: 400 MiB total. Input accounting reserves space for both
archives and a transient receive name pair; it must not depend on compression
being favorable. The input worst case is two copies of the 85,622,679-byte
archive plus a 64 MiB reference archive, below 256 MiB. Extraction must not
duplicate imported tensor files.

All capture categories sum to 864 MiB, below the unchanged 2.25 GiB ceiling.
The full proof must precede archive receipt/extraction so it does not overlap
their footprint. Static work planning reserves 110 seconds for setup, 240 for
workers, 90 for the CPU verifier and 60 for final checks: 500 seconds within
600, with no clock reset. These are refusal bounds, not a prediction or a capacity
reservation; oversized future results/imports must refuse rather than expand.

Keep the strict foreign-compute guard and five-second observations, leaving
foreign processes alone and stopping only owned children. Sampling is not
continuous exclusivity, and accounting is not a kernel quota. Recheck source,
imports, GPU and storage between phases. Preserve CUDA visibility for original
GPU workers and hide it only for the CPU verifier/proof. Keep temporary/cache
paths private and disclose them after each worker.

## Implementation and review gate

Next implement a new unarmed replay-1 operator, reference-subset importer and
combined-cohort verifier. The consumed v1 live observer/runtime permits replay 0
only. Its saved audit also expects per-array reference comparisons, whereas an
efficient replay-1 worker records `exact_replay=true`. Do not edit those frozen
modules, mutate their policies or bypass their guards to reuse the consumed
helper. Reuse only source-bound pure helpers where the new scope permits it.

Before any resource request, test exact subset coverage/hash/mtime preservation,
expired-grant separation, tamper/alias/overwrite refusal, efficient-1 exact anchor
and math-1 fresh-reference ordering, full-artifact cross-process replay, the
additional NumPy return captured before a failed gate, CPU-only verifier scope,
FP32 preservation, live-versus-export layout handling, hooks/alarms, all original
phase evidence, cold full imports with dispatch blocked, bounded storage/time,
owned-child handling, complete same-connection transport and unarmed refusal
before SSH/claim. Real tiny CPU math rehearsals must be labelled as CPU analogues;
they cannot qualify the live efficient CUDA branch. Negative saved-cohort tests
must reject altered canonical references, missing reconstruction or replay fields,
wrong reference edges and attempts to claim an original parent session.

Independently review the exact new source list, reference archive, control and
unarmed helper after those tests. Then seek a separate explicit resource grant
covering the new uploads, third CPU process/call and larger category bounds.
This proposal creates none of them and asks for no GPU check or approval now.

Finances stay unchanged: $200 project / $160 compute / single $10 diagnostic
cap; $9.8116 conservative encumbrance, $0.1884 unreserved, old $2 hold and $0.0792
gap retained, last wallet $6.56 and reconciliation incomplete. All known Runpod
Pods remain terminated. No paid benchmark is implied by these university checks.
