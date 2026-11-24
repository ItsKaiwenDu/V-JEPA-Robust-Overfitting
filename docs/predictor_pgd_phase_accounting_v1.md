# Guarded whole-PGD phase accounting

A separate, unbound cold derivation adds phase journals to the reviewed shutdown
runtime. It records the work omitted from the old context endpoints without
changing scientific bodies or extending a deadline. The
[local review](../report/checks/2026-11-23-pgd-workload-phase-accounting-v1-local-review.md)
records 24 first-run passing checks and a first-run independent saved audit.
The installed budget gate remains closed; actual performance is unqualified.

## Boundaries and clock domains

All intervals use one process-owned monotonic clock. Parent and child endpoints,
or the old performance-clock endpoints and new monotonic endpoints, are never
subtracted. These are host caller intervals, not CUDA kernel or device-event
times. No additional device synchronization is introduced.

| Record | Included | Explicit exclusions |
| --- | --- | --- |
| Outer setup | Original archive/source/cache/storage installation and checks in `prepare_tree` | Earlier handshake/root claim and its own record publication |
| Child startup | Live bootstrap sample before cache installation through child dispatch/ledger startup | Its journal row is published retrospectively, not durable before startup |
| Model preparation | Original provider construction, early device transfer and strict checkpoint load | Enclosing caller checks are recorded separately |
| Owned context | New start publication, original context construction/entry, gradient work, original completion journal and objective-timer exit | Its own end-row publication and return |
| Evaluation | Original setup, selection, no-grad scores, evaluation checks, mode/RNG restoration and unit publication in nested spans | Nested intervals overlap; they must not be added |
| Worker body | Entire original `UnitTimers` context, including its exit, validation and existing completion publication | Final phase summary is outside this span |
| Child caller envelope | Live child entry sample through the final sample after `UnitTimers` exit and earlier phase-row publications | Its own summary publication, descriptor close and return |
| Parent role | Original role plan, proof/unit invocation, worker publication/exit, verification, receipt and state writes | Its own end-row publication and later roles |
| Parent work envelope | Same-parent preparation start through final accounting sample | Earlier outer setup and its own summary publication/close/return |

The parent role includes the worker summary publication and exit that the child
envelope excludes. The parent envelope includes earlier parent end-row
publications. No instrumentation overhead is subtracted. The original context
records remain unchanged and retain their narrower definition. If startup fails
before a child ledger can open, a child summary need not exist; parent evidence
does not invent missing child intervals.

## Guarded metadata and mandatory restoration

`models/predictor_pgd_phase_accounting_v1.py` provides a source/plan/role/deadline
bound `PhaseLedger` and an anchored `FileSink`. The sink creates one private,
single-link file exclusively, retains its descriptors, checks named and held
identity/extent, verifies all previous bytes, and fsyncs each canonical row.
Journals carry a hash chain, exact LIFO span tokens, owner PID/thread and monotonic
domain. Limits are 512KiB/384 rows/192 spans/16 nesting levels per journal,
4096 bytes per row and 64KiB per summary. A clock reversal, foreign owner,
deadline, byte drift or exhausted limit refuses completion.

The outer context proxy returns the exact original `OwnedStep`; its exit always
runs before the accounting observer. A failed start observation cannot skip the
original mode/RNG restoration. Accounting failures preserve an existing primary
exception and add a note, poison timing completion, and retain partial evidence.
Summary publication is attempted once, with no overwrite or retry. A missing or
incomplete phase summary cannot satisfy the new parent completion reader.

A live exact `PhaseLedger` is required before scientific imports. A dictionary
or saved summary cannot bind execution. Saved metadata validation checks bytes,
counts, endpoints, source/plan/owner joins and explicit exclusions; it does not
authenticate physical execution, numerical agreement or measured performance.
Even a complete fake measurement retains scientific/performance/training FALSE.

## Separate source derivation

`models/predictor_pgd_phase_accounting_build_v1.py` uses the active eleven pins in
`configs/predictor_pgd_phase_accounting_inputs_v2.json`. It emits definitions,
control, source payload and manifest only. `LIVE_EXECUTION_BINDING` remains NULL;
there is no client, scope, package, grant or launch API. Input v1 and both
successful cold builds are preserved. Before checks, v2 strengthened saved-row
validation of kind, retrospective starts, exclusions and caller envelopes; all
v1 sources and outputs remain. Production did not change after checks began.

Ten modified generated files have full-byte inverses. Removing only the explicit
phase bridges restores the provider, target unit, evaluator and PGD module ASTs.
Forty-one old cached files are byte-exact, with one new observer module. Original
`OwnedStep`, target factory, snapshot and native route ASTs remain exact. Original
unit timers, alarms, publication functions, unit file definitions, frozen proof,
monitor, scanner, assessment and Guard remain unchanged.

The new graph contains 56 uploaded sources/936711 bytes, 48 runtime files/481518
bytes, 47 cached modules, nine supervisors and 270 primary result paths. Six new
worker sidecars and three parent/root sidecars have bounded declarations. The
metadata reservation, including all permitted partials, is 27705344 bytes, below
the existing 32MiB reservation ceiling and 64MiB metadata cap including 8MiB trace.
The original packet and all previous path descriptors remain exact.

Objective120s/unit720s/setup180s/proof60s/soft2280s/hard2400s/final120s,
collection600s/connection3000s, RSS4GiB/allocator7GiB and all storage/process/
thread/sample/cadence caps remain unchanged. Accounting consumes the original
continuing windows; it creates no extra allowance. Bounded owned shutdown and
confirmed-idle gates remain in force.

## Validation and evidence limits

All 24 new checks passed on their first execution. They cover full context
boundaries, original exit and restoration after observer failures, continuing
clock/owner/deadline refusal, journal drift/caps, six saved metadata mutations,
unbound/dictionary entry refusal, source inverses and the closed installed budget.
No passing or closed scientific case was repeated.

Four new fake lifecycles exercise the emitted runtime: one complete four-role
flow, and gradient, original timer-exit and phase-summary publication failures.
The successful three worker journals each have 200 rows, including 40 complete
contexts, 42 selection calls and six no-grad score calls. All process, clock,
timer, signal, PIDFD, scientific compute, GPU and proof ports are explicit FAKE.
Original context-exit and evaluator-restoration bodies run with fake operands;
there is no Torch/NumPy/model/gradient calculation or target/network work.
Four 16MiB reserves and opaque archive/source/pixel copies are real filesystem
work. Synthetic clock increments are not measurements of that filesystem work.

The raw transports contain 442/178/179/260 files. The independent stdlib audit
parses each through its final EOF, checking payload/reference bytes, hashes,
mode/mtime, journal chains, prefixes and terminal records. Failed sessions and
collection/partial/journal flags remain unchanged. In the summary-publication
failure, the old worker `complete.json` was already written TRUE; it is retained
exactly. The missing phase summary and failed parent prevent completion. No
status is rewritten to make those records agree. Transport TRUE is separate.
Fake proof remains FALSE/zero bytes; the old real-reader refusal is inherited.

The audit passes first run on 59365 prior hashes, 65294 prior fixture entries,
2492 new entries, eleven pins, ten inverses and saved source/test/transport joins.
It imports no project/test/scientific callbacks and performs no tensor decoding,
scientific recalculation, subprocess, signals, timers, proof, GPU or network work.
The historical FIFO is LSTAT only, never opened. Four prior directory lifecycle
exceptions remain exact. Current device16777229 uses the preceding frozen local
observation; no new device waiver, probe or old/target remapping is introduced.

Raw evidence is under
`runs/pretrained/diagnostic/pgd-workload-phase-accounting-v1-20261123`.
`sealed-workload-phase-accounting-v1.json` inherits the prior seal and includes
new sources, evidence and index snapshots; live mutable indexes remain excluded.
User alone stages, commits and pushes.

## Remaining gate

The installed v4 client still binds the older graph and unconditionally refuses
through the unchanged budget version. It is incompatible with these new source
identities and paths. The next separate local step is to bind new client/outer/
bootstrap/scratch/collection readers to this exact graph and review phase-aware
completion locally, with explicit fake ports and the budget gate still closed.

This instrumentation does not solve the rejected budget. Three full720s windows
plus setup/proof ceilings still exceed the soft2280s window. Actual complete
units/fresh replay, matched timing/overhead, memory, exact recovery, economics,
calibration, data/protocol adoption and training remain gated. No scope, approval
request or launch command follows. All real grants stay consumed; actual v3
remains failed/31-of40 partial. Mechanism closed; scientific policies, study and
financial caps/holds remain fixed. No training percentage or time estimate.
