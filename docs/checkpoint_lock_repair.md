# Checkpoint locks on the university filesystem

**Verified follow-up:** [the university retry](../report/checks/2026-10-23-university-cpu-recovery.md)
passes all 165 selected CPU checks with zero skips and exit 0. All saved
recovery comparisons and the actual NFS4 lock probe verify independently.
The instructions and rejected outcomes below are retained as history; no
additional university rerun is needed. GPU performance/research gates remain.

October 23, 2026. The source-only university check completed with **99 passed,
29 failed, zero skips/errors** in 169.18 seconds. Its rejected outcome remains
preserved in the [follow-up report](../report/checks/2026-10-23-checkpoint-lock-repair.md).
All 70 export members and the original 263 source/bundle pins verify.

The generated initial-state repair works in the retained Linux evidence:
20 loader cases, 15 native diagnostic cases and 31 precision cases pass.
Independent rechecking of eight fresh workers reproduces the saved gradient,
attack and precision comparisons. All twelve recovery initializations load
the pinned state despite three deliberately different construction seeds.
The four resume processes then fail before loading their saved checkpoint;
full recovery has not passed this Linux attempt.

## Failure and repair

All 29 failures are in shared-execution recovery/anchor/evaluation checks and
report `OSError: [Errno 9] Bad file descriptor`. Twenty-eight traces identify
the shared lock in `CheckpointStore._lock`; one interruption-expectation
assertion wraps the same error. The lock file is opened with `a`, which is
write-only, before requesting a shared read lock.

Linux [flock documentation](https://man7.org/linux/man-pages/man2/flock.2.html)
describes NFS emulation using whole-file record locks. The corresponding
[record-lock rules](https://man7.org/linux/man-pages/man2/fcntl_locking.2.html)
require a readable descriptor for shared locks and a writable one for
exclusive locks. This explains the observed error on the shared university
home; the retry additionally records the actual filesystem and old/new access
probes rather than assuming a mount's configuration.

The fix changes that descriptor to `a+`: read/write append. It creates the
same lock file when absent, preserves existing contents and retains the
inode. Shared reads, exclusive writes, nonblocking contention failures and
unlocking remain enforced. No lock error is ignored or retried without a lock.
Checkpoint transactions, hashes, generations, weights and optimizer/RNG
recovery contracts remain unchanged.

A new local regression control uses real POSIX record locking to reproduce
the old `EBADF` before the fix. Independent processes check all four
reader/writer contention combinations and successful acquisition after release.
Additional checks require propagated lock errors without checkpoint mutation
and release when protected work raises. These are filesystem checks; they do
not claim an actual NFS mount on the Mac.

## Fresh CPU retry

The new source-only package is separate from the rejected handoff:
`runs/pretrained/diagnostic/cpu-lock-repair-20261023/vjepa-cpu-locks.tar.gz`.
Its extraction directory is `cpu-locks-20261023`. Keep the prior
`cpu-fixtures-20261023` directory and failed evidence intact.

The launcher uses the existing university `~/.venvs/vjepa` Python environment,
hides all CUDA devices, records `df -PT` for the run filesystem, and checks
write-only and read/write shared/exclusive locks before the test run. It runs
165 selected checks, including all previous 128 cases plus checkpoint-lock,
training recovery and anchor checks. Zero skips/errors/failures are required.
It installs nothing and uses no paid GPU. The test cap remains 25 minutes,
with a 28-minute outer cap including failure export.

On the Mac:

```bash
cd /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting
lock_handoff="runs/pretrained/diagnostic/cpu-lock-repair-20261023"
scp "$lock_handoff/vjepa-cpu-locks.tar.gz" \
  "$lock_handoff/vjepa-cpu-locks.tar.gz.sha256" \
  kdu@linux10609.dc.engr.scu.edu:Desktop/VJEPA-Robust-Overfitting/
```

Inside university SSH:

```bash
(
  set -e
  cd "$HOME/Desktop/VJEPA-Robust-Overfitting"
  sha256sum -c vjepa-cpu-locks.tar.gz.sha256
  test ! -e cpu-locks-20261023
  tar -xzf vjepa-cpu-locks.tar.gz
  bash cpu-locks-20261023/scripts/university_cpu_fixtures.sh
)
```

Use the new printed log path to monitor the detached job. It is safe to exit
SSH while it runs. Copy the new evidence archive and its manifest for independent
verification. A reported pass still needs that verification before another
rental; no new GPU timing, whole-model memory fit, optimizer recovery on A40,
study affordability or research admission follows from this preparation.

The full local suite is recorded separately in the report. Historical scientific
thresholds, study YAML, benchmark results and budget pins remain unchanged.
The historical budget builder continues to reject changed diagnostic source.
The displayed $3.930 aggregate is not a per-Pod bill; retain both existing $3
holds and the conservative $2.805 unreserved pilot balance until reconciliation.
