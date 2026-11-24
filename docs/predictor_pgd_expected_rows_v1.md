# Expected-row outer-list prototype: local synthetic proof

**All 101 new checks pass on their first run. This is a local metadata result;
cost, full original guard-domain equivalence and complete-unit fit remain
unqualified.** No original `Snapshot`, tensor operation or worker is executed.

See the [decision](../configs/predictor_pgd_expected_rows_v1.json),
[frozen cases](../configs/predictor_pgd_expected_rows_cases_v1.json),
[unbound helper](../models/predictor_pgd_expected_rows_v1.py),
[standalone proof source](../tests/test_predictor_pgd_expected_rows_v1.py) and
[saved review](../report/checks/2026-11-24-pgd-expected-rows-v1-local-review.md).
The [preceding reassessment](predictor_pgd_local_budget_reassessment_v1.md)
remains preserved. Its expressly requested commit/push completed as
`a3883af109e3a963f64c476d21b0233d12c7e474`, with the supplied title and numbered
description. HEAD matched origin/main and was clean before this separate step.
This new work remains for review; no further unrequested commit/push occurs.

## Narrowed hypothesis

The prototype reuses only a private outer list object. Every reached tuple row
is constructed anew and every synthetic identity-port call is made in the
original row order, including unchanged records and repeated tokens. No previous
identity result, tuple, hash or model observation is reused. The list is cleared
before filling and in `finally` after comparison, on success and BaseException.

The baseline uses the original-shaped parameter/buffer/module comprehensions,
with synthetic records replacing saved records and an explicit fake identity
port replacing `id`. The candidate uses the same row shapes and fresh calls in
loops that append new tuples to its private list. There is no original-source
factory, bridge, guard patch, target API/constructor/adapter selection or runtime
binding. The new module is outside the frozen worker/upload graph.

Reusing an outer list object does not prove storage capacity is retained or any
allocation is saved in practice. Tuple allocation remains. Builtin type/shape
inspection, method calls, clear/refill, branches, cleanup and bookkeeping add
real work; they can erase any benefit or make the candidate slower. No cost,
dominant-cost, allocation-byte, peak-memory or speed claim is made.

## Typed path and fallback

The reusable path inspects exact builtin lists/tuples, exact string names,
integer opaque tokens and boolean parameter flags; observed row values have
matching primitive domains. It inspects no foreign iterator, comparator or
attribute protocol. Unsupported outer/container/value types use the baseline
before any identity call or foreign iterable consumption. These extra builtin
reads are real work, not a concurrent-mutation guarantee.

The declared new recipes exercise outer list subclasses, generators, stateful
row iterators, string subclasses, numeric flags, custom comparisons and malformed
rows. Their saved iteration/lookup/comparison traces match the baseline without
duplicate consumption. Comparison capture receives a fresh baseline list, never
the reusable list, on those fallback paths. Closed or reentrant owners also use
the baseline; the late reentrancy check preserves an already built outer prefix.

Each synthetic owner has separate storage. Ordinary exercised methods return a
comparison result, None or copied primitive diagnostics. Privileged tests inspect
the backing attribute to inject dirty contents/wrong storage and to retain tuple
witnesses. Python attributes remain reflectable: these tests establish no hard
privacy, thread safety, writer exclusion, native ownership or security boundary.
No additional runtime fault interface is introduced.

## Frozen evidence and coverage

Eighteen prior source/review pins stay fixed. Five new source/oracle files have
exact frozen copies and hashes before the first helper check: prototype, runner,
case corpus, independent invariant oracles and corpus-builder source. The builder
does not execute either helper to derive its expected results or traces.
Membership is frozen at 95 cases and six owner invariants, with no warmup or
earlier helper check. Sources and oracles remain unchanged after the first run.

The 95 cases cover all three row kinds, unchanged repeated checks, fresh identity
results, changed names/tokens/flags, reordered/grown/shrunk/empty records, duplicate
and Unicode names, ignored module fields, mismatched observed rows, cache
corruption, closed/reentrant fallback, stateful recipes, malformed inputs,
identity/iterator/comparison exceptions, BaseException cleanup and later use.
Five cases use a new pure metadata OR driver to check parameter, buffer,
gradient-scan and module short-circuit order; it calls no model enumeration.

The six additional invariants check late reentrant-prefix preservation, distinct
fresh tuple rows with one retained outer list, owner isolation, copied diagnostics,
no input/observer/row references in idle owner storage and ordinary method results.
The largest literal record list in the declared corpus has three rows. This is a
finite synthetic domain, not exhaustive real-model or high-cardinality coverage.

All 101 checks pass first run. Both arms' saved comparison results, exact error
types/messages and lookup/iteration/comparison traces join the independent
declarative oracles. Idle cache row count is zero after every exercised call and
exception. This describes owner storage, not traceback/caller reference lifetime,
garbage collection, allocator capacity or physical peak-memory behavior.

The runner executes only this new synthetic module, denies workload process,
PIDFD, signal, thread, timer/clock and network ports, records zero blocked-port
attempts and has no stopwatch. Standard-library imports and local metadata
filesystem/JSON/hash operations are real. No Torch/NumPy/model/tensor/gradient,
original guard/test/scientific callback, FileSink/ledger/worker, target/GPU/proof
or paid work occurs. No old passing witness, symbolic harness, encoding benchmark
or scientific cohort is repeated.

## Review and next eligible task

The saved-only stdlib audit verifies the 71544 prior hashes, 77941 prior fixture
entries, 102 new entries and 78043 total, eighteen old pins, five frozen sources/
oracles, source ASTs and every saved result/trace/error join. It executes no new
prototype/test callback and no old project/scientific callback or scientific-value
decoder/recalculation. Historical FIFO stays LSTAT-only. Device16777229 uses the
prior frozen observation with no probe/remap/waiver; old target66307 and four exact
directory lifecycle exceptions remain. Further changes need a new record/review.
The seal excludes live indexes and includes exact before/after snapshots.

The [next local cost requirements](../runs/pretrained/diagnostic/pgd-expected-rows-v1-20261124/NEXT-LOCAL-EXPECTED-ROW-COST-v1.json)
permit at most a separate bounded matched stdlib metadata cost comparison, with
new frozen accepted membership and independent correctness oracles. Whole local
paths include input construction, eligibility, fresh fake identity calls, every
tuple, clear/refill, comparison, cleanup and bookkeeping, without assumed overhead
subtraction. A future dedicated stopwatch may measure outside bounded batches,
at most16 batches per arm, with a cooperative5-second total cap checked between
batches; that is not OS enforcement. No benchmark runs in this proof step, and
passing checks must not be rerun for timing.

Any local cost result still cannot qualify full original-domain refusal/error/
lifetime equivalence, physical private capture, target/backend behavior, whole-unit
timing/memory/fresh replay or training. Separate full source/inverse/domain/
ownership review and actual matched complete-unit evidence plus all other gates
remain. Copy-elision's provenance gap and the old encoding rejection are unchanged.

## Fixed historical status and admission

Genuine old v3 proof and 31 original gradients remain genuine without rerun,
demotion or promotion. Whole units remain zero, candidate/replay did not start,
and all old failed/partial/pending/consumed and transport flags remain. The current
phase runtime48/481518B remains distinct from historical v3 47/450548B; no new
actual-target execution, current remote-state observation or primary-cause claim.

Active graph remains56 uploads936711B/runtime48 files481518B/cache47/nine
supervisors/270paths, control
`d997bf9ed74c8ebf4504a5e6214a35e28365f85e9da3d7738a5b58f5edaa9f29`.
Original immutable `tobytes()` snapshots, EVERY fresh hash, ordered `state_digest`,
science, guards, durability, shutdown, continuing clocks and all caps stay fixed.
The v5 gate remains unconditionally closed before protocolclock/ATTEMPT/connection;
budget decision remains `REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE`.

Objective120/unit720/setup180/proof60/soft2280-hard2400/final120/collection600/
connection3000, RSS4GiB/allocator7GiB and every other cap remain. All real grants
are consumed; no operational scope/handoff/package/grant/approval request/launch,
university query/SSH/upload/retry/resume/retrieval/cleanup, target/GPU/proof or paid
work follows. Mechanism closed/research-training false. Recovery/economics/
calibration/data/protocol/training gates, PGD/EOT/optimizer/registry/study YAML and
financial caps/holds remain. Historical snapshots are not current quotes/balance.
No training percentage/time estimate. Commit messages remain automatic; Git
mutations need an express request for the work being committed.
