# Offline corrections for scratch allocation and CPU audit receipts

Integrated follow-up (November 4): the [new capture operator](cpu_oracle_capture_v2.md)
now passes 121 combined checks and the full saved-data integration audit. Its
actual helper is verified unarmed. The integration-pending statements below
record the earlier component checkpoint; a separate new target exception and
live XFS qualification remain pending.

The separate v2 components pass **69 regression tests**. The complete saved
November 4 target capture also passes the new offline reader and two fresh local
verification processes. This verifies saved-array evidence under the original
numerical bounds; the original parent and storage failures remain rejected.
See the [review](../report/checks/2026-11-04-cpu-oracle-corrections-v2.md) and
[machine-readable evidence](../report/checks/2026-11-04-cpu-oracle-corrections-v2.json).

## Physical allocation accounting

`models/scratch_allocation_v2.py` is a library for a future operator. It has no
remote launch command. It requests one fixed 2.5 GiB extent with
`os.posix_fallocate`, then writes only within that extent. Unsupported
preallocation, allocation errors and changed lengths refuse before reference
work; there is no fallback to an extending write loop.

The library counts both logical bytes and `st_blocks * 512` across the private
setup directory, including other setup files, directories and its event log.
It checks the unchanged **3 GiB physical and logical ceiling** before and after
allocation, each write, fsync and close, and around removal of its own successful
proof file. It also checks free space, inodes, deadline and path identity. A
refusal retains its partial proof and failure receipt when storage permits.
Existing files are preserved.

The default prerequisite is now 8 GiB available space, with a 1 GiB operating
floor. That prerequisite does not reserve disk space. The write request remains
2.5 GiB; the setup ceiling remains 3 GiB. The recorded v1 value of
**4,294,901,760 allocated bytes** is explicitly rejected under that ceiling.

The fixed-range allocation request does not promise a physical upper bound.
Python documents [space allocation for the requested range](https://docs.python.org/3.11/library/os.html#os.posix_fallocate)
and [512-byte allocated-block accounting](https://docs.python.org/3.11/library/os.html#os.stat_result.st_blocks).
XFS also has [allocation and preallocation behavior](https://docs.kernel.org/admin-guide/xfs.html).
Observed ceiling checks are still needed. These components install no aggregate
kernel quota and provide no exclusive reservation or zero-overshoot guarantee.
An enclosing operator must supervise blocking allocation and I/O calls within
the original deadline and account for the complete setup/capture lifecycle.

The tests exercise real small writes, fsync, readback and own-file cleanup on
the Mac using an explicitly injected allocator. They also inject quota/full-disk/
unsupported-allocation errors, over-allocation at allocation/write/close,
corruption, short writes, other setup growth, aliasing and deadline failures.
**The real Linux/XFS allocator and full 2.5 GiB proof have not been tested with
v2.** The tests do not qualify a university host or replace that live check.

## Context-sensitive audit statistics

`models/cpu_oracle_reassessment_v2.py` authenticates the unchanged rejected
target review, response and all 435 saved members before reassessing arrays.
It uses the original numerical implementation, source bindings, fixture and
`rtol=1e-12 / atol=1e-14` bounds. No saved tensor, source packet, original receipt
or consumed claim is rewritten.

The new comparison contract requires exact identity, coverage, shape, numerical
bounds, failed-element counts and pass/refusal decisions. It validates finite
diagnostic statistics and their consistency with the decision. It retains each
auditor's error statistics and runtime context separately, rather than requiring
floating diagnostic summaries from different contexts to be identical. Every
saved stage and final reference comparison is independently recomputed; matching
failure labels never become a numerical pass.

This is a separately named offline interpretation, not a waiver of the frozen
v1 protocol. The stored one-thread/four-thread reports differ in **354 diagnostic
statistics** with matching decisions. The unchanged original reader still raises
`Stage comparisons do not reaggregate.` The new reader verifies 2,048 stage and
48 fixed-final comparisons, and records all local/producer statistic differences.
Two fresh verification processes repeat the saved-value checks successfully.
Their environment requests one and four threads; the actual Mac Accelerate
thread count was not measured, so this is not a controlled BLAS thread study.

The producer context remains linux10603, Python 3.11.17, PyTorch 2.8.0+cu128 and
NumPy 2.4.6. The separate local audit uses arm64 Python 3.11.16, PyTorch 2.14.1
and NumPy 2.4.6. Original per-worker CPU/thread/CUDA-hidden conditions are checked;
the audit does not reproduce the original GPU execution context. The older
approximately 1e-10 discrepancy and exact failed parent report pair remain
unresolved.

## Offline verification interface

`scripts/reassess_cpu_oracle_v2.py` can create a new local report only outside
the original captured tree. It requires `--review`, the exact trusted
`--review-sha256`, and a fresh `--output` directory. Verification adds
`--verify-output` and the externally pinned `--summary-sha256`. It verifies source
hashes and recomputes saved-array decisions in the current process. It has no
SSH, allocation-test, model-training or remote-launch mode.

The reviewed output is under
`runs/pretrained/diagnostic/cpu-oracle-corrections-v2-20261104/reassessment/`.
Its summary SHA256 is
`ea88a94a402a521b38ae51b828822c8dde08eb7f374d165b3632c271f0f30beb`.
The adjacent driver, test XML and stdout/stderr preserve how the checks ran.
These raw local artifacts remain outside Git.

## Next execution gate

Next integrate the tested components into a separately versioned **unarmed**
capture operator and review its complete storage/runtime envelope. Only after
that concrete operator is ready should an additional university attempt be
considered under a separate resource exception. The consumed linux10603 target
grant does not extend to a retry. Linux10609 has no authorization, attempt or
accepted target-review bridge and must stay idle. The original helper must not
be reused.

No new university attempt, full-size allocation proof, cleanup, paid resource,
financial release, optimization adoption or research admission occurs here.
The single $10 diagnostic envelope, $160 compute and $200 project caps, existing
holds and incomplete billing reconciliation remain unchanged.
