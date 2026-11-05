# Remaining original attention replays: offline proposal checked

The [proposal](../../docs/original_attention_replay1_proposal.md) stages the two
missing original workers on linux10603: efficient replay 1 must exactly match
the accepted efficient replay 0, then math replay 1 must use the new complete
passing efficient replay 1 as its reference. No accepted worker is rerun.
Original sources, numerical bounds, actual backend and replay gates stay intact.

The proposed import has 16 existing files / 59,132,357 bytes, including two
29,487,103-byte `values.pt` snapshots. Every path, size, hash and mtime is pinned.
A future archive is capped at 64 MiB; none is prepared now. Old worker
permissions/deadlines remain provenance, not new execution grants. Original
full captures remain authenticated locally; the subset is never described as a
complete new remote copy of those captures. Imported and newly executed workers
must have separate receipt fields.

Source inspection also finds the original final check's one NumPy FP64 oracle
reconstruction per case. The proposal explicitly includes one additional fresh
CPU-only verifier, at most 90 seconds, after both new CUDA workers pass. It may
call that unchanged oracle once, saves its actual FP64 return before the gate,
and checks the original argument order and 1e-12/1e-14 bounds. Together with the
two original workers, the future attempt permits two Torch CPU calls, one NumPy
CPU call and twelve GPU attention iterations. It adds no SDPA to the CPU verifier.

All 22 explicit refusal conditions in the frozen summarizer are mapped for
future implementation. The combined verifier must preserve its scientific
checks, including full-artifact exact replay, fixed-reference reconstruction,
per-replay references, backend, timing validity and allocator checks. The
original single-parent/session envelope remains unclaimed: no fabricated
`session.json`, restarted clock or original-parent completion flag is permitted.
Any eventual result is explicitly a cross-attempt component cohort, requiring
final independent review of all old and new captures before acceptance.

Static resource arithmetic proposes one new 600-second attended attempt,
including the full 2.5 GiB proof and setup, two CUDA workers at most 120 seconds
each, the CPU verifier at most 90 seconds, and final checks. Setup + workers +
verifier + final-check reservations total 500 seconds, leaving 100 seconds inside
the same clock. Collection remains at most 600 additional seconds / 3 GiB through
the same SSH connection; no retry or old allowance is reset.

The raw category is explicitly proposed at 400 MiB rather than the preceding
320 MiB: 240 MiB new-worker allowance + 64 MiB imported values + 16 MiB NumPy
return + 32 MiB atomic-save allowance + 48 MiB margin. Input is proposed at
256 MiB rather than 128 MiB, covering both archives and a transient receive pair
without relying on compression. Capture categories total 864 MiB, below the
unchanged 2.25 GiB aggregate capture ceiling. The 3 GiB setup ceiling also stays;
the full proof precedes both archives. Per-file, return, allocator, inode and
free-space safeguards remain. These are caps, not measured future resource use
or a reservation. A new scope review must explicitly cover the additional CPU
process/call, imported records and category increases before any execution grant.

The design check reauthenticates all 2,236 accepted sealed inputs, all 199 latest
saved files with exact mtimes, and the accepted independent review. It seals
2,252 proposal inputs. A second standard-library-only check confirms the
reference graph, exact subset, source coverage and arithmetic with process and
network creation denied. No Torch/NumPy module, worker, oracle or CUDA is
imported or evaluated. No new unit tests are added; the unchanged consumed
operator's 163 prepared checks remain its historical validation baseline.

This step creates only proposal/provenance files. No new operator, reference
archive, launch helper, authorization, claim, SSH, paid work or cleanup exists.
Next implement and meaningfully test a separate unarmed importer, replay-1
operator and combined-cohort verifier, then independently review the exact
payload/control before requesting one new explicit scoped resource exception.
The consumed v1 observer/runtime and audit remain unchanged and must not be
patched to admit replay 1 or its `exact_replay` reference receipt.

Both hosts can stay idle. The historical approximately 1e-10 refusal and missing
old computed arrays remain unresolved. Original parent-session, other-fixture,
full-model, speed/cost, adoption and research qualification remain pending.
Instrumented timing cannot support study-budget or speed claims. No new financial
hold or release occurs: $200 project / $160 compute / single $10 diagnostic cap,
$9.8116 conservative encumbrance, $0.1884 unreserved, old $2 hold and $0.0792 gap,
last wallet $6.56 and reconciliation incomplete. All known Runpod Pods remain
terminated.
