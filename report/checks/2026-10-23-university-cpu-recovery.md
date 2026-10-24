# Verified university CPU fixture and recovery check

October 23, 2026. The source-only `cpu-fixtures.q8YbFO` run passes all
**165 selected checks, zero failures/errors/skips**, in 352.306 seconds.
The saved exit status is **0**; the user's truncated terminal line did not
indicate a missing or failing status. The [machine-readable report](2026-10-23-university-cpu-recovery.json)
records independent verification, source identities and detailed comparisons.

## Archive and environment

The 92,848,738-byte export has SHA-256
`a14319354225614ed1a794ee0052276b07aae69acd081bf191377bf40bb34820`.
All 75 regular members and the original 269 handoff source/bundle pins verify.
The packaged source manifest exactly matches the frozen local restored copy.
Linux x86_64 uses Python 3.11.17, PyTorch 2.8.0+cu128 and NumPy 2.4.6.
CUDA devices are hidden; no GPU benchmark or dependency installation occurred.

The log records the actual university home mount as **NFS4**. Its unchanged
write-only control (`a`) reproduces shared-lock `EBADF`, while exclusive
write-only and both read/write (`a+`) locks succeed. Lock-file contents remain
unchanged. All eight locking regression cases run, including independent
reader/writer contention and release, with no bypass of lock failures.
This confirms the lock-access explanation for the earlier 29 rejected cases.

## Independent replay and recovery verification

Native and shared-forward CPU comparisons each complete four fresh workers.
Worker/result/tensor hashes, original upstream source, initial checkpoint
identity, numerical/oracle/reconstruction limits and exact replay are checked
again from saved tensors; their comparisons reproduce exactly.

All four fresh-process recovery combinations pass: clean and adversarial
training, each with shared-fp64-cudnn and shared-fp64-native. In each case,
three deliberately different constructions load one verified initial state.
The saved interrupted checkpoint is at update 2; uninterrupted and resumed
states reach update 6. Full saved state, model/optimizer tensors, RNG/history,
contracts and continuation draws match exactly. Each comparison covers
247 tensor leaves (1,583,550 elements) and 1,527 other leaves. The loader,
numerical and recovery checks retain their original strict limits.

JUnit also records successful scheduled evaluation, anchor loading, malformed
checkpoint/contract rejection and budget-provenance cases. These are test
outcomes; independent retained-tensor reconstruction specifically covers the
eight diagnostic workers and four fresh-process recovery comparisons above.

## Scope and next gate

This clears the selected **Linux CPU fixture and filesystem recovery check**.
The earlier 99-pass/29-failure attempt and the paid A40 setup rejection remain
preserved. This is not an A40 performance, full pretrained memory-fit, GPU
optimizer recovery, held-out evaluation or whole-study affordability result.
Research is not admitted. No scientific tolerance, study YAML or historical
budget source pin was changed, and historical budget-source rejection remains.

The next offline step is a new lower-cost GPU handoff containing the verified
fixture and lock changes. The old October 22 package must not be reused for
that retry. Refresh its proposed reservation to fit the conservative $2.805
unreserved pilot balance while retaining both earlier $3 holds; do not treat
the displayed $3.930 aggregate as a settled per-Pod bill. After package checks,
obtain a fresh all-in quote and availability. Nothing here starts paid compute.
