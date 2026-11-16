# V3 setup GPU refusal and secondary collection error

The saved refusal identifies another CUDA compute process at attempted startup:
PID 3418810, /usr/NX/bin/nxnode.bin, using 191 MiB on linux10603. Total recorded
GPU memory use was 293 MiB. The strict original guard refused the other process
regardless of its small memory use. This is a saved observation from that attempt,
not a fresh statement about the workstation's current state. Existing processes
were left alone; no manual GPU check or additional connection was needed.

The visible NoneType.layout exception is secondary. The setup journal checkpoint
runs before SystemDriver.prepare assigns its guard. After that checkpoint refuses,
Coordinator.run_work has not created its parent session or entered the proof/worker
role loop. The outer handler retains the work error but then constructs Collector
with owner.driver.guard still None. Collector.__init__ dereferences guard.layout,
replacing the useful GPU message. The small same-SSH refusal frame still contains
the original setup guard message and both GPU query texts/hashes.

The saved metadata and exact pinned source ordering show that proof/model worker
dispatch was not reached. The corrected packet import path was not exercised.
Earlier setup did create the new root and install inputs/source; the statement
about no proof/model dispatch does not imply that no setup writes occurred.
Remote directory contents were not inspected, and no results archive was collected.

Five local records are pinned by bytes, extent, private mode and modification time:
ATTEMPT, ROOT-RECEIPT, REFUSAL, CLIENT-REFUSAL and empty ssh-stderr.log. Outer control,
authorization, consumed claim and root identities join. There is no saved collection
body or collection receipt. The transport's final SSH exit status was not retained.
Both GPU query lengths/hashes verify against their complete saved text. All
scientific/training flags remain false.

A new cold source correction inserts one barrier after the original owned-idle
check and before finalization/collection. If the parent session is unclaimed or its
guard is absent, it raises the original work error. An inconsistent success without
an initialized parent also refuses. The existing bounded same-connection refusal
handler then closes the journal and returns the useful error. Claimed sessions
with a valid guard keep their existing success/failure collection path. No worker,
clock, ownership, cleanup, numerical or GPU rule is relaxed.

Nine distinct new checks have passing latest results over seventeen executions.
The first eight shared-harness cases lacked MiB and failed before their intended
path; only that test binding was corrected and those eight cases repeated. The
one passing inverse-source check did not repeat. All first snapshots/logs/negative
fixtures remain. The tests reproduce the exact saved masking message, verify
primary refusal preservation, cover absent guard/session combinations, unchanged
claimed-session routes and refusal when owned cleanup is unconfirmed. They execute
only the standard-library outer function with fake clock/alarms/root/host/parent/
collector/import bindings. No real server file, timer, process, signal, GPU, model,
objective, storage proof or network operation executes.

The independent saved-file/AST audit passes all 13,140 previous hashes and 10,728
previous fixture entries. The old FIFO receives lstat only, never a read or hash.
The auditor initially imported base64 after its exec-blocking hook; only that
stdlib import order was corrected, with both scripts/logs kept. No production or
test source changed for the audit. The source inverse restores the entire original
remote_session function exactly after removal of the one new barrier.

The correction is a cold function candidate, not a new runnable handoff. The
consumed v3 helper, authorization, source, claim and refusal evidence remain frozen.
Next integrate it into a fresh complete version and review source/scope/ownership
and same-SSH refusal behavior before seeking one additional exact scoped approval.
Do not rerun/rearm v3, reset its claim, retry collection or inspect the remote root
under the consumed grant. No new connection or authorization is ready now. Original
scientific/resource and financial caps, holds and reconciliation remain unchanged.

See [the local correction plan](../../docs/predictor_pgd_split_preclaim_refusal_v4.md).
Machine report: 2026-11-16-pgd-v3-GPU-preclaim-refusal-review.json.
Raw records: runs/pretrained/diagnostic/predictor-pgd-startup-incident-v3-20261116/.
