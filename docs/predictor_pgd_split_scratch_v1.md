# Whole-scratch guard for the split diagnostic

The new standard-library guard covers the complete proposed scratch tree around
its separate `capture/` result/log namespace. It has 108 distinct passing local
checks across 109 executions. It creates no directories, writes no files, allocates
no proof and launches no process. It is not a university helper or authorization.

## Exact namespaces

The pinned contract retains the 17,192,095-byte original packet and its 99 extracted
members, the 63 previously assembled runtime files, the original base/design JSON,
and the frozen 419-result table. A future control-bound installation descriptor
must name six exact supervisor sources; it cannot add arbitrary files or callbacks.
The old PID-handle monitor and allocation-proof source hashes are fixed. The other
supervisor sources, trusted bootstrap and control binding remain unfinished.
The installation descriptor's hash must ultimately be pinned by that reviewed
outer control; these checks alone do not establish its authority.

The proposed tree is:

```text
scratch/
  original-packet.tar.gz[.partial]
  code/                 # 99 unchanged packet members
  overlay/              # 63 source-pinned runtime files
  supervisor/           # six control-bound supervisor sources
  configs/              # frozen base.json and design.json
  capture/results/      # existing split result paths, unchanged
  capture/logs/         # existing five bounded log paths
  cache/, tmp/          # at most 48 files together; bounded depth/names
  emergency-space.bin
  space-proof-v2.bin.partial
  <17 exact operational metadata names and their permitted partials>
```

The result table stays independent of setup storage. Unknown source/result/root
paths, private-mode changes, wrong user/device, symlinks, special files, unrelated
hardlinks, changed root/parent identity and excessive traversal refuse. Parent and
child directories are opened with held no-follow descriptors. Reads compare held
and named file identities and their ancestor bindings. These are guarded observations,
not a claim of a fully atomic or adversarial-race-proof filesystem snapshot.

## Accounting and publication

Original limits remain: 3 GiB setup/proof, 2.25 GiB capture/collection, 1,792 MiB
raw, 128 MiB input, 128 MiB shared code, 32 MiB combined cache/tmp, 16 MiB emergency,
64 MiB shared metadata including the two 4 MiB traces, and one full 2.5 GiB proof.
Logical bytes and allocated blocks are checked independently; directory blocks are
included. Free space must be at least 8 GiB initially and 1 GiB during work, with
2,048 and 1,024 available inodes respectively. These observations reserve nothing.

All 17 operational metadata caps total 3,031,040 bytes and share the 64 MiB limit
with result metadata and traces. The frozen completed-results-plus-partials budget
and these root caps fit together. There are at most 1,024 files, 256 directories,
1,280 visited entries and 256 non-result files. Cache/tmp gets 48 of those file
slots, depth at most eight including its leaf, and no extra storage category.
A second on-disk evidence archive has no admitted path; collection must stream.

The unchanged result writer publishes an immutable name by linking its partial and
then removing the partial name. A live scan accepts only that exact same-directory,
same-inode, identical-stat final/partial pair in the declared result or root-metadata
namespace. No external link or third name is allowed. Both names remain in the
observation. The pending logical entity and its blocks count once; the separate
`named_bytes` diagnostic counts both names. The frozen result budget sees the
pending partial, never a completed artifact. Partial counts and metadata limits
remain shared; a settled scan refuses every such pair and unfinished publication.
No raw record, value, trace row or scientific predicate is changed by this
filesystem-alias accounting. Ambiguous concurrent transitions may refuse; no scan
retry is hidden behind that refusal. Filesystem sampling is not a kernel quota.

## Proof and immutable-source checks

Proof readiness requires a fresh namespace with no previous proof records, the
full allocated 16 MiB emergency extent, initial capacity floors, and enough observed
logical/allocated space for the full proof plus the frozen 32 MiB allocation reserve.
A sparse emergency file's logical length is insufficient. This helper authorizes
no allocation or launch; the future coordinator owns its claim, clock and ordering.

The saved-proof reader retains the original full extent/write/fsync/readback,
non-injected allocator, removal, setup ceiling and 60-second predicates. It adds
strict JSON duplicate/nonfinite/type checks, a parent-bound receipt hash, a complete
digest and an observed peak at least the requested extent. Leftover proof files
refuse. No proof function is imported or called. A matching metadata record cannot
authenticate that a fresh proof executed; the returned execution-admission flag
stays false. The future parent must join the actual fresh proof process/claim.

The immutable-source gate requires all 172 declared assets, exact sizes/hashes,
full emergency allocation and stable before/after scans. Later scans refuse any
changed, missing or replaced pinned asset. This does not rehash every input on
rapid polls; explicit source verification and later stable metadata checks have
different jobs. Installed source bytes, scope authority and fresh-child imports
still require the future bootstrap and combined package review.

## Local evidence and remaining work

Initial run: 103 passes, no failures/errors/skips. Review added pre-proof projected
allocation fit: its one affected positive check and two new scalar refusals passed.
Three additional checks exercised changed/missing/replaced source pins against
tiny private filesystem fixtures; only those new cases ran. Total: 108 distinct
checks, 109 executions, one justified passing repeat, no old test/cohort repeats.
Every source snapshot, JUnit and log is preserved under
`runs/pretrained/diagnostic/predictor-pgd-split-scratch-v1-20261115`.

GiB/block-capacity cases are synthetic stat dictionaries, not large or sparse files.
Proof JSON explicitly labels itself synthetic; the leftover proof file is a tiny
text label. The full source-loop harness deliberately substitutes tiny descriptors
and scan/read callbacks only in its test double; it cannot qualify a real 172-asset
installation. Separate real descriptor tests read a pinned source leaf without
importing it. No model, arrays, gradient, profiler, proc/PID/signal, child, GPU, SSH,
actual proof, retrieval, cleanup or paid compute runs. Ordinary small fixture I/O
is the only filesystem work.

Next implement trusted source installation and exact control/argument binding,
then coordinate the proof, numerical child and native-before-candidate component
children under the original clocks. Finish combined receipts and bounded same-SSH
streamed collection. Review that complete package before one new exact university
scope. Live PIDFD/stop/resource/fit and scientific validation remain pending.
Numerical bounds, eight full gradients plus two component VJPs, frozen outcomes,
consumed old grants, financial caps/holds/reconciliation and training gates stay
unchanged. No repeated manual GPU status is needed during this local preparation.
