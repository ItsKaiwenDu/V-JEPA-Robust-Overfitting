# Correct the read-only Archery evidence capture

The first copy returned SSH exit1 and zero archive bytes. It stopped at
`fcntl.flock(own, LOCK_EX | LOCK_NB)` on a descriptor opened `rb`. This was a
collector bug. The original source and the failed download folder are kept
unchanged; the failed capture started no model or remote test job.

The [Linux manual](https://man7.org/linux/man-pages/man2/flock.2.html)
documents whole-file record-lock emulation over NFS and the requirement for
write access when requesting an exclusive lock. Native Mac flock accepted
that descriptor in the earlier local check, which missed the NFS access rule.
The correction uses a shared read lock on the same existing host lease.
That still excludes the original workers' exclusive lock. The registry lock
remains shared; no lock file is created or written and10609's lease is untouched.

The corrected source is separately identified as
`read-only-archery-guard-refusal-v2`, SHA256
`3729c61162d215ed22e3287c6e3fbbf27e55d267244df2ce895d691b255c2792`.
It copies the same42 existing files plus one transport manifest, observes all
286 fixed historical hashes and records claim/attempt presence and current
GPU status. Its180+15second bound and strict checksum/size/path checks remain.
The original scientific packet, numerical/GPU guard and case time limits do
not change.

A real POSIX record-lock regression reproduces the original EBADF and the
corrected full collector/stream/receiver passes through that same primitive.
Independent processes check reader compatibility, exclusive-worker rejection
and release for both native flock and POSIX record locking. Existing lock
contents/inodes and all captured bytes remain unchanged. Wrong host/source,
present/absent case states and corrupt-copy rejection also pass. A held10609
lease does not block the capture. Only the GPU query is mocked; these local
checks are not a claim of actual mounted-NFS validation.

Actual corrected university copying remains pending. Run the new copy helper
once from a separate Mac Terminal and leave linux10609's running job open.
The last observed YoYo transcript progressed to armD. The old Archery model
helper must not repeat; no numerical retry or additional allowance is prepared.
After copying, independently review the actual155 JUnit results, timing,
source pins,286 preservation hashes and unclaimed case boundary. Current
acceptance stays13/20 with six complete clip pairs; no partial scientific
aggregate or research adoption occurs.

[Structured report](2026-10-27-archery-guard-copyfix.json).
