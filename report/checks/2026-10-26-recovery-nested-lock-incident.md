# Recovery startup stopped at a nested registry lock — October 26, 2026

The university launcher passed all 139 CPU checks (zero failures, errors or
skips), then exited 1 during preflight. The recovery wrapper already held
`.registry.lock` when `Registry.__init__` tried to acquire it again through a
different descriptor. This is a wrapper bug, not a second-SSH-session policy
or a CUDA-process rejection. That call precedes the hardware gate, grant
consumption and numerical dispatch in this invocation. The deployed packet,
failed transcript, original case evidence, STOP and authorization are retained.

A real-file-lock regression reproduces the failure before the fix. The local
fix constructs/preflights Registry while holding only the host lease, then
rechecks preserved files and the exact status inside the short claim lock.
All 67 focused CPU checks pass, including genuine competing-lock rejection,
changed-state rejection, and release on failure. A combined full `run` plus
unmocked `preflight` rehearsal with real Registry/file locks rechecks ten real
saved cases and preserves 56 coordinator hashes. Only the neural body/export,
host and GPU are mocked in that local rehearsal; it qualifies no CUDA case.
The previous rehearsal exercised preflight separately and missed this combined
lock path. Its saved reports remain historical; they are not relabelled.

The fix is local only. There is no corrected GPU helper to run yet, no new
grant, and no reset of the approved 80-minute budget. The read-only capture
will copy 31 existing startup/source/ledger files, record current absence or
presence of the grant and attempt02, observe all 56 retained file hashes and
collect a current GPU snapshot. It writes no university files and starts no
model/tests. Byte fixtures pass collection/receiving, wrong host/source,
presence/absence reporting and corrupt-copy rejection.

Run from Mac Terminal:

```bash
bash /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting/runs/pretrained/diagnostic/recovery-lock-incident-20261026/copy-existing-startup.sh
```

After readback, independently verify the authentic 139 JUnit results, exact
deployed sources, launcher start/end/exit, preserved-file hashes and unused
grant/attempt boundary. Charge failed startup time before preparing a separate
corrected packet. Preserve the same single-use case grant and remaining bound;
never patch the old deployed source in place, delete locks, clear STOP or repeat
the old model helper. Ten cases remain independently verified and nine recorded
remotely; Floor remains incomplete and no partial aggregate is admitted.
