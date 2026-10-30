# Interrupted CPU reference captures: storage-quota evidence

Both read-only status downloads are independently verified: 18 metadata files,
eight checksummed JSON records and 256 source-file hash visits. The two packets,
61 original scientific pins, 33 upstream source bindings and all earlier/current
claims remain unchanged. No probe process was listed at either saved status
check. Both first workers are incomplete; neither has an accepted numerical
result, complete reference coverage or a valid result export. No retry started.

Linux10603's saved worker traceback explicitly reports
`OSError: [Errno 122] Disk quota exceeded` while syncing an observed chunk.
Subsequent index/worker-record persistence also fails with the same error. Its
last saved record has only the unobserved before-call output; 58 of 64 chunk
receipts and a chunk-058 partial file remain. The saved index's incomplete hook
field predates the failed final write; it is not proof that a live trace hook
was left installed. The parent stopped after about 9.60 seconds before export.

Linux10609's saved record reaches the after phase, after recording before and
observed outputs and all 64 observed chunks. Its after.npz.partial remains, and
the parent records an incomplete-worker error at about 10.61 seconds. Its worker
and parent logs are empty, so this host has no explicit quota traceback. The
shared-home quota problem is consistent with its interrupted write, but the
exact exception on this host is not independently established. Its trace index
records a complete observed call and hook restoration; the worker's overall
three-call capture is incomplete. No later one-thread or replay worker started.

The before-call output digests recorded on both hosts are identical, as are the
before/observed output digests recorded on 10609. Those are metadata observations
only: calculated tensor files have not yet been downloaded or independently
compared. In total, 976 intermediate-array descriptors are preserved across
122 chunks; none is counted as a verified numerical comparison yet. No result
resolves, reproduces or waives the earlier CPU-reference discrepancy. Closing
Terminal is not established as a cause of these failures.

The user's read-only storage check reports 19G in the research directory,
including 7.7G in matched-panel-transfer, 4.9G in the checkout, and several older
transfer directories. The existing Python environment uses 6.6G and the Python
installation 109M. Both quota commands print nothing, so the actual byte/inode
quota and remaining allowance are unknown. The filesystem's 2.1T free space is
shared capacity and does not override an account/group quota. These usage values
are user-provided command output, not a fresh remote measurement by the review.

Preserve the partial captures before planning cleanup. The separate
`cpu-oracle-salvage-20261030/copy-existing-files.sh` Mac helper is prepared and
unrun. It reads exactly 145 existing files (298,518,026 bytes) via streaming tar,
one SSH/password connection per host, and creates no remote archive. It writes
only to fresh Mac folders. Of the files, 141 have previously recorded SHA256
digests; two unfinished fragments and two empty archive files have no prior
digest and remain explicitly unqualified. The receiver checks sizes, exact file
coverage and known hashes before verified extraction. It refuses changed files,
duplicates, traversal, symlinks or missing members, preserving failed copies.
Local copy checks pass, including six refusal variants and a network-free SSH
stub round trip. They start no reference calculation or remote write/deletion.

Next use that copy-only helper and independently inspect the saved fragments.
Then identify exact redundant files whose Mac backups are verified before any
cleanup. No file deletion, environment removal, claim reset, new CPU/GPU attempt,
server-setting change, numerical threshold change or paid compute starts here.
A future probe needs a separately identified bounded plan and sufficient verified
shared-home quota headroom; the failed one-time helpers remain consumed.

All financial state is unchanged: $9.8116 conservatively encumbered / $0.1884
unreserved in the single $10 diagnostic envelope, inside $160 compute / $200
project / $40 noncompute reserve. Historical $2 hold and $0.0792 gap remain;
last wallet $6.56, full account reconciliation false. No optimization adoption
or research admission follows from this status review.
