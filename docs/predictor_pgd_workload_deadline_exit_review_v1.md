# V3 whole-unit deadline and exit-confirmation analysis

The [saved v3 target review](../report/checks/2026-11-23-pgd-workload-handoff-v3-target-review.md)
records actual preparation and31 original gradient completions before a failure
at the720-second unit boundary. The proof and237-file collection succeeded, but
no full PGD unit, candidate or replay completed. The grant is consumed. This
source analysis changes no code, cap, claim or scientific method.

## Recorded facts and inferred failure path

The original launch/deadline pair differs by720 seconds. Collection begins
0.132956048 seconds after that deadline; the final GPU sample is just before it.
The saved failure is a RuntimeError in coordinator assert_idle, with child poll
status None. The child log has no traceback. No original primary exception,
terminal child status, original RSS/allocator peak or cleanup receipt was saved.

The saved source orders failure handling as follows:

1. The owned monitor enforces absolute phase/work time and sampled RSS, then
   attempts held-lease termination and bounded reaping during its finally block.
   It closes all lease handles even when the original error path remains active.
2. The outer launch wrapper performs an extra child wait only if monitoring did
   not run or a lease still needs cleanup. A monitored, already-closed lease
   therefore skips that additional wait.
3. Its exception handler calls assert_idle before re-raising the original error.
   A child poll still returning None can replace that error with the observed
   exit-unconfirmed RuntimeError.
4. The diagnostic writer preserves only the current exception's message/frames/
   notes. The initial exception chain and any monitor-local observations are lost.
5. The remote outer calls assert_idle again before collection. The completed
   collection is consistent with a later successful check; it does not change
   the failed session or prove terminal status, timing or exhaustive cleanup.

This is a plausible deadline/shutdown race from source and timing, not a saved
primary-timeout diagnosis. Deadline, resource or other initial errors cannot be
distinguished from the retained record alone. A missing RSS refusal is not proof
that no RSS breach happened. No current remote process state is queried or claimed.

## Requirements for the next independent local adapter

1. Preserve these exact shipped sources, failure records and all31 completed
   gradient journals. Derive a separate OS/coordinator variant with exact source
   identity and inverses; leave scientific/native/math/PGD/evaluator entries fixed.
2. Retain bounded primary-error and cleanup observations without allowing a later
   idle assertion to erase the first cause. Distinguish observed trigger, cleanup
   attempts, handle ownership, exit/reap confirmation and final refusal. Keep all
   metadata bounds, FALSE science/complete fields and immutable evidence rules.
3. Keep held ownership valid throughout cleanup and require confirmed idle before
   clock handoff/collection. Never signal an already-closed/reused handle, suppress
   the idle check or infer death merely from a closed descriptor. Bound every
   wait by the original remaining work/connection clocks and explicit cleanup
   allowances; do not extend unit/objective compute time or create a fresh clock.
4. Add explicit fake delayed-exit, poll/reap/handle/signal-failure, deadline and
   interruption controls. Test both successful late confirmation and conservative
   refusal when confirmation cannot be obtained. Integrate source/cache/parent/
   timer/collector/client behavior locally before any fresh exact scope. No real
   process/GPU/proof or old scientific cohort is needed to repair these OS paths.
5. Review a concrete complete-unit time/overhead plan under the existing outer
   budget before a new attempt. The recorded17.745–17.895-second contexts cover
   only31 completed observations and exclude their own journal publication and
   timer exit. They do not provide complete original/candidate/replay timing or
   independently estimated overhead. Keep full hashes, ownership/numerical guards,
   scoring and exact PGD20/R2/EOT8; no shortened/tiny/lean substitute can qualify.

No implementation, alternate timeout or new scope is selected or adopted here.
All real grants are consumed: no retry, automatic resume, separate retrieval,
old remote cleanup, new approval request or repeated manual status paste follows
from this review. Partial hash journals are not restart/recovery checkpoints or
completed-unit accuracy evidence. Actual full units/replay/performance, memory/
exact recovery/economics/calibration/data/protocol gates remain. The mechanism
diagnostic stays closed; training policies, study and financial caps/holds stay fixed.
