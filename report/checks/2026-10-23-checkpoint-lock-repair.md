# University CPU follow-up and checkpoint lock repair

October 23, 2026. Diagnostic evidence only; no research admission or GPU rental.
The [machine-readable report](2026-10-23-checkpoint-lock-repair.json) records
source/evidence hashes and verification details. The [retry guide](../../docs/checkpoint_lock_repair.md)
contains the fresh source-only university commands.

## Verified rejected attempt

The copied `cpu-fixtures.jXYphp` export is 69,221,414 bytes with SHA-256
`b6d7f0786bce72f3330e864187f2f064d34ccc576bdcc0ba903501963ff97d4b`.
All 70 regular members and 263 original handoff source/bundle pins verify.
Linux x86_64 uses Python 3.11.17, PyTorch 2.8.0+cu128, NumPy 2.4.6;
`CUDA_VISIBLE_DEVICES` is empty. The 128 CPU tests completed in 169.18 seconds:
99 passed, 29 failed, zero skips/errors. The overall attempt remains rejected.

All generated checkpoint-loader cases and both four-worker gradient/attack
comparisons pass. Saved tensor hashes, oracle/reconstruction gates and exact
replay are independently recomputed. All twelve fresh recovery initializations
also load their pinned state after three deliberately different construction
seeds. The four resume processes fail before loading their saved checkpoint;
there is no successful full Linux recovery result from this attempt.

Every failed test reports the checkpoint shared-read `EBADF`: 28 traces name
`CheckpointStore._lock` directly and one interruption assertion wraps it.
The shared university home uses a network filesystem, while the lock descriptor
was write-only (`a`). Linux NFS [emulates flock with record locks](https://man7.org/linux/man-pages/man2/flock.2.html),
whose [read locks require a readable descriptor](https://man7.org/linux/man-pages/man2/fcntl_locking.2.html).
This mechanism explains the observed error; actual mount/access probes are
also recorded by the fresh retry so its environment is checked explicitly.
The original initialization primitive discrepancy remains unattributed.

## Narrow repair and local validation

The only production AST change opens the same checkpoint lock file with `a+`.
Read/write append preserves its contents and inode. Shared/exclusive
nonblocking locks, errors and release remain enforced. Transactions, checkpoint
hashes/generations and optimizer/RNG contracts are unchanged. No numerical gate,
attack, training policy, study YAML or historical budget pin changed.

A real POSIX record-lock control reproduces the previous `EBADF` before the
fix (one failed/seven passed), then passes after it. Four independent-process
reader/writer combinations verify contention and release; unavailable locks
must abort reads/writes without mutation or an unlocked fallback. A raised
protected operation also releases its lock. This control does not claim a
mounted NFS filesystem on the Mac.

All 165 affected checks pass locally with zero skips. The complete documented
project suite passes **503 tests, with two actual-CUDA skips**, zero failures
or errors. An initial unscoped pytest command collected upstream and ignored
archived tests and had 31 collection errors; scoping to the documented `tests/`
directory runs the full project suite. Both invocation logs are retained.

## Next gate and spending

A distinct `cpu-locks-20261023` snapshot retains the failed handoff and raw
exports. It hides GPUs, uses the existing university environment, records the
actual run filesystem, probes old/new lock access and requires all 165 CPU
checks with zero skips. Dependency installation is unnecessary. The 25-minute
test/28-minute outer caps and bounded failure export remain.

The new archive is 5,313,508 bytes, SHA-256
`41926b451a2727fd77609bb4c650761d882d27ccc1dec88a25136a52d3779e6b`.
All 270 members and 269 source/bundle pins verify after extraction. Three
extracted checks pass: record-lock checkpoint round-trip, native four-worker
replay and fresh-process adversarial shared-native full AdamW/RNG recovery.
Source/runtime tampering rejection, inline access probes with real POSIX
record-lock controls, and partial failure export including probe metadata
also verify. This final preparation record follows the frozen source snapshot;
compare that archive against its own manifest rather than later status text.

The new Linux retry is not yet run. Its archive/manifest must be copied and
independently verified before another paid deployment. Neither this fix nor
CPU validation measures A40 pretrained throughput, full-model memory fit,
GPU recovery or whole-study affordability. Historical budget-source rejection
remains active. No new cloud charge was incurred. The latest displayed
aggregate remains $3.930 with unresolved per-Pod attribution; retain both $3
holds and the conservative $2.805 unreserved pilot balance.
