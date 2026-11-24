# Complete-unit budget options under fixed limits

The source review finds **no qualified complete-unit budget solution**. The
installed gate remains `REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE`. Seven concrete
options were examined against the bound phase graph; one is suitable only for a
separate local byte-equivalence prototype. No runtime or source graph changes
are implemented in this stage.

See the [local review](../report/checks/2026-11-24-pgd-budget-options-v1-local-review.md)
and [machine-readable options](../configs/predictor_pgd_budget_options_v1.json).

## What the existing deadlines actually reserve

`Coordinator.run_work` computes `later = sum(...)` but never loads that variable.
Its role deadline is still
`min(start + boot.SECONDS[role], self.work_deadline - 120)`.
It reserves the original final allowance, not all later roles' full windows.
The binding did not change this behavior and never promised every later unit720s.

Three full720s windows use2160s. The soft2280s boundary leaves120s for setup,
proof and gaps; the setup/proof design ceilings sum240s. All ceilings therefore
cannot be promised together. This does **not** establish that every actual run
is impossible: shorter actual durations could fit, but none are qualified for
the complete original/candidate/replay route.

| Role | Symbolic start offset | Deadline offset | Available cap | Role ceiling |
| --- | ---: | ---: | ---: | ---: |
| Proof | 180 | 240 | 60 | 60 |
| Original | 240 | 960 | 720 | 720 |
| Candidate | 960 | 1680 | 720 | 720 |
| Fresh replay | 1680 | 2280 | 600 | 720 |

This is limiting ceiling arithmetic with zero gaps, not a runnable schedule,
measurement, forecast or lower bound. Exact-deadline completion would refuse.
Adding real gaps cannot create a larger last window. Simply using the unused
`later` total would reject or redistribute time; it would not accelerate work.
The hard2400s boundary/final120s reserve and collection600s/connection3000s are
not extra compute allowances.

## Matched timing remains necessary

The old31-of40 context endpoints omit their own journal publication and timer
exit. They do not establish whole-unit cost or predict the remaining contexts,
candidate or replay. The saved actual proof duration is historical, not a bound
for another proof. No old gradient/proof/cohort is rerun or recalculated here.

The new phase definitions include original completion publication/timer exit,
but their tested durations are synthetic. Child caller envelopes exclude their
own summary publication/close/return; parent role spans include worker exit and
receipt checks. Parent/child stamps or old/new clock epochs must never be
subtracted. Nested spans overlap and must not be summed. Use the original
parent envelopes for admission and child phases for attribution, retaining their
explicit exclusions and the original absolute deadlines.

Actual full units, complete phase coverage, final publication/exit, parent
observations, fresh replay and memory remain missing. A metadata dictionary,
passing fake lifecycle or this options review cannot qualify those measurements.

## Reviewed source-level options

| Option | Disposition | Source reason |
| --- | --- | --- |
| Cache fixed-tensor hashes or rely on versions | Reject | `Snapshot.check` requires a fresh byte observation as well as identity/version checks. |
| Batch fsync or move timing endpoints | Reject | Changes durable ordering or excludes work that the full context must include. |
| Skip the journal's full-prefix readback | Reject | Before/after append protect different observation windows; a cached prior digest does not replace either check. |
| Reuse a worker/model or replay state | Reject | Changes fresh child/preparation/RNG/replay ownership semantics. |
| Replace tensor hashing with a direct mutable view | Defer | Removing the original `tobytes` snapshot lacks a buffer-lifetime/coherency equivalence proof. |
| Reserve later windows or extend caps | Reject as a speed solution | Reservation cannot make all ceilings fit; widening caps is outside the fixed constraints. |
| Reuse immutable canonical phase-row encoding | Local prototype only | The source serializes a row before hashing it and again after adding the row hash. Exact byte reuse might remove that duplicate serialization. |

The eligible item is ranked by its narrow semantic surface, **not expected time
savings**. A new parser, copying or bookkeeping could erase any benefit or make
it slower. No dominant cost has been measured, and no claim says this can close
the unit budget. It preserves every hash input/check and creates no lean route.

`FileSink.append` performs two full-prefix checks per row; final receipt adds
one. For row-prefix sizes `S[i]` the read/hash volume is
`sum(S[i-1] + S[i]) + S[R]`. The existing384-row/512KiB caps give a loose
403177472-byte bound per ledger. This is source-level byte arithmetic, not
elapsed overhead, filesystem throughput or an actual workload measurement.
Encoding reuse must not remove these checks, fsync or any retained observation.

## One next local prototype contract

The candidate is **requirements only**, not implemented, emitted, bound or
adopted. A separate version may investigate reusing the first immutable canonical
row encoding to construct the final encoding after inserting `row_sha256`.
It must preserve exactly:

1. The original prehash bytes including their newline, SHA256, final canonical
   row bytes and all row/file/count/depth bounds.
2. Nested values, sorting, Unicode/escaping, role-like text in errors, and the
   existing nonfinite/invalid-input refusals. Unchecked substring replacement
   cannot establish structural equivalence.
3. Both full-prefix readbacks, anchored identities, fsync/durable ordering,
   timing boundaries, original clocks, mandatory cleanup and primary errors.

Use stdlib synthetic metadata only. Retain baseline/candidate sources and full
byte inverses; compare against an independent canonical-byte oracle. Do not run
a worker, import Torch/NumPy, repeat scientific controls, emit a new operational
graph or change the budget gate. Any parser/copying cost remains real work.
Byte equivalence alone would still establish no target performance or fit.

## Preservation and remaining gates

This step preserves the prior uncommitted binding. HEAD remains `df9e61b`; user
alone stages, commits and pushes. Eighteen whole-file pins and14 function AST
anchors bind the review to the exact existing graph. All runtime/client/budget/
scientific/config/guard sources remain unchanged; the new options config is a
review artifact, not execution admission.

Raw evidence: `runs/pretrained/diagnostic/pgd-budget-options-v1-20261124`.
The standalone stdlib audit passes first run on69854 prior hashes and76570 prior fixture
entries, source witnesses and arithmetic without project/test callbacks, tensor
or array decoding, scientific recalculation, worker/process/signal/timer/GPU/
proof/network ports or new fixtures. Ordinary metadata filesystem/hash work is
real. No fake lifecycle or project test is run; no performance is measured.

Current device16777229 uses the frozen preceding observation; no new probe,
waiver or remap is introduced. Old device records, target66307 and four narrow
directory exceptions remain exact. The historical FIFO is LSTAT only, never
opened. All sources/tests/auditors/failures/results/statuses/consumed grants,
including actual v3 proof/31-of40 partial failure, remain unchanged. Seal:
`sealed-budget-options-v1.json`; mutable indexes have exact before/after snapshots
and remain excluded from the seal.

No actual scope/package/grant/approval request/launch command, SSH/query/upload/
retry/resume/retrieval/remote cleanup/target/GPU/proof/paid work follows. Original
scientific work, resource caps, trainingPGD5/EOT1, routineEOT1, optimizer backward,
registry/study YAML and financial200/160/40/single10 caps/holds remain fixed.
Mechanism closed; full units/replay/performance/overhead/memory/recovery/economics/
calibration/data/protocol/training remain gated. Historical financial snapshots
are not current quotes or balances. No training percentage or time estimate.
