# Local PGD repair integration

The live November 16 byte probe reproduced a real XFS allocation change at owned
file close. The frozen writer recorded the pre-close allocation and later refused
its own unchanged data. The reviewed Store candidate closes the owned fd before
pinning metadata and retains all later allocation/identity/size/mtime checks.
This step integrates that exact **full module**, with its original 419-result
Inventory, into a fresh PGD source candidate. It does not use the probe's one-file
inventory or change scientific workers.

The saved parent diagnostic also reported ProcessLookupError from process
sampling, followed by a note that its owned numerical child exited 1. That trace
is truncated and does not establish whether the disappearing entry was the root
or a descendant. A deterministic fake-child reproduction shows the ordering hole:
metadata may disappear while Popen.poll still reports no visible exit status.

## Owned-child monitor repair

A new cold recipe accepts only the consumed monitor's exact bytes. It adds one
stdlib import and changes only the missing-process sampling branch in watch.
After the existing poll, it may call **one wait on the already-owned Popen child**,
with timeout `min(0.1, phase_deadline-now, work_deadline-now)`. It checks the
original finite, monotonic clock before and after the wait. There is no resampling
retry, new process discovery, clock reset or numeric-PID signal.

A confirmed nonzero exit still raises the original child-failure RuntimeError.
An unconfirmed exit retains the original metadata error and stops the attempt.
Other ordinary wait errors add a bounded note to that metadata error. Cancellation
(KeyboardInterrupt/SystemExit) propagates through the existing owned-handle
cleanup. Exit zero still requires at least one prior RSS sample, no surviving
owned tasks, final guards and time remaining; it cannot fabricate a successful
observation from missing metadata. ProcReader, PidfdLease and the entire cleanup
finally block are unchanged. Component RSS/time gates remain; the coordinator's
existing proof/numerical adapter keeps its original role clocks and RSS policy.
This repair has only fake-process qualification; no native process race is run.

## Complete source and reader bindings

`models/predictor_pgd_close_integration_v1.py` reads only the pinned consumed
10609 source/control metadata. It returns 69 candidate source files, a supervised
control, runtime manifest and three local reader modules. Exactly five target
files change: Store, monitor, coordinator's monitor hash, scratch's contract and
monitor hashes, and bootstrap's policy hash. The other 64 uploaded files are
byte-identical, including every scientific worker, IO facade, serializer,
forward-preserving attention implementation, seed/value gate, numerical bounds,
original collection implementation and allocation-proof implementation.

The payload totals **878,631 bytes**, with 63 runtime files and six supervisors.
The current manifest is bound by `runtime_repair.runtime_manifest_text` and the
scratch contract; the older host-change manifest stays explicit historical
provenance. The full scientific/resource/host scope and embedded historical
base/design/financial bytes stay exact. Contract source pins, policy fixed support,
coordinator watch hash and local reader hashes all bind the new monitor. No caps,
gradient counts, proof requirement, ownership or scientific gates are relaxed.

Local bootstrap/scratch modules are exact server bytes. The local collector
changes only its two bootstrap/scratch import aliases; its functions stay exact.
The real local reader verifies a synthetic full-catalogue collection header and
rejects stale Store/monitor/control pins. The actual pinned importer reads all 69
private source fixtures while keeping them unexecuted, and refuses tampered
Store/monitor bytes. The catalogue remains 172 assets / 419 result files.

The exact current source hashes and controls are recorded in
`configs/predictor_pgd_close_integration_v1.json` and raw
`runs/pretrained/diagnostic/pgd-close-integration-v1-20261117/CANDIDATE.json`.
The Store is the already reviewed 9,915-byte candidate with SHA256
`7bcbd863eab7c23940fa1d32b194448b88353202c2100b66a9a30f59e4ef6022`.
The repaired monitor SHA256 is
`07237d77806aba1c129e0e93520a2ceedd3ff2b06001f391d1cdaa46e55fae06`.

## Validation and limits

35 distinct new checks pass over two runs (25 and 10), with no failures or passing,
historical or scientific cohort repeats. The first 25 pass on the first candidate.
Review then narrows the wait-error handler from BaseException to Exception so
cancellation can propagate; the first source/control/payload/readers are retained.
Only ten new cancellation, complete-header and pinned-source cases run afterward.
All other monitor code and behavior are unchanged by that review correction.

The checks use real source/metadata files, small inert result writes, fsync/link/
close/byte hash and owned local identities. Children, process handles, signals,
RSS, clocks, pauses, target identity, collection inventory and grants are explicit
fakes or inert metadata. No actual child process, process signal, timer, network,
SSH, GPU query, storage proof, NumPy/Torch array, model or gradient runs. The old
frozen process-error path and repaired path are compared using new fake fixtures;
no old scientific results are repeated or relabelled.

This is a **local source candidate**, not a runnable outer/client/target handoff.
No authorization, attempt or remote scope is created. Existing consumed launchers
must not be pointed at these new sources. All university grants are consumed.
Next locally assemble the new outer/client/CLI and source envelope, bind their
bootstrap/collector/contract hashes, retain the existing refusal/clock/collection
behavior, and finish focused checks/review before requesting one new model scope.
No repeated manual GPU/key paste is needed. Full-model/gradient, training,
performance and cost qualification remain false; financial caps, holds and
reconciliation stay unchanged. The user stages, commits and pushes.
