# Preserve setup refusals before parent session creation

This local-only cold function candidate addresses the secondary v3 collection
error. It is not an armed or runnable target handoff. All prior grants are consumed.

The saved original guard refused another CUDA process during the setup checkpoint.
The parent driver had no guard/session claim yet, but the outer error path tried
to initialize Collector. Its guard.layout access obscured the primary GPU error.

The cold builder requires the exact 34,015-byte consumed v3 source by SHA256 and
returns only a corrected remote_session function definition. It inserts one branch
immediately after owner.driver.assert_idle and before finalization. An unclaimed
parent or missing driver.guard cannot enter collection; the original work error
is raised, or an inconsistent no-error state gets an explicit refusal. The existing
same-SSH bounded refusal handler preserves last_guard_refusal. Claimed sessions
with a valid guard retain their original path. Removing this branch restores the
whole original function AST. No scientific, source/pixel, ownership/stop, timer,
metadata, GPU or financial bound changes.

Nine meaningful local regressions have passing latest results. The complete outer
function executes under explicit fake globals only. Tests reproduce the saved
NoneType.layout message, then verify the original GPU refusal, all three missing-
guard/session combinations, inconsistent success refusal, unchanged claimed
success/failure and the owned-idle barrier. They invoke no actual target, process,
timer, proof, scientific objective, signal or GPU. Failed harness runs and all
snapshots are retained; passing old tests/scientific cohorts were not repeated.

Next integrate the checked function into a fresh complete outer version. Preserve
the packet-aware v3 bootstrap, scientific/runtime sources and original policy bytes.
Bind fresh source/control/client/CLI identities and a new exact handoff path, keep
all old claims/results untouched, test the whole source/setup/refusal integration,
and independently review/seal that complete package. Only then ask one new exact
university approval. Do not request a repeated manual GPU status check; a future
approved helper checks live. Current GPU state is unknown beyond the saved refusal.

Never rerun the consumed v3 command, weaken the other-process guard, kill another
process, change server settings, reuse a proof/result/root, force raw collection,
open an extra retrieval connection, install dependencies or start paid compute.
The latest attempt did not reach the storage-proof or model worker, so it supplies
no new packet-import, numerical, backend, performance/cost or training qualification.
