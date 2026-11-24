# Local inventory of pinned-target source evidence

**No Torch2.8.0+cu128 source/package artifact was found within the declared local
search.** The snapshot copy-elision proposal remains deferred. The existing
immutable bytes copy, every fresh hash, runtime/client graph and unconditionally
closed budget gate remain unchanged.

See the [inventory review](../report/checks/2026-11-24-pgd-target-source-inventory-v1-local-review.md)
and [machine-readable decision](../configs/predictor_pgd_target_source_inventory_v1.json).
This is a scoped, sequential inventory, not proof that no copy exists anywhere
on the computer or could appear later.

## Search coverage

Filename inventories cover the repository, including hidden environments and
saved diagnostic evidence, for wheels, Torch-named source archives, version
files, Tensor documentation and selected native copy/export source names.
Repository filename expansion adds previously uninspected paths only. Saved
scientific payloads, generic checkpoint archives and Git object history are
excluded; no tensor payload is decoded.

Nine standard cache/checkout paths were probed. Only
`/Users/kaiwendu/Library/Caches/pip` exists among them; the other pip/uv/Torch-hub
and conventional PyTorch checkout paths were absent. Named absolute pip/uv/XDG
cache overrides were considered. The pip name list contains33 files;16 package
body files were inspected. Cache envelope/self-check files are not treated as
library payloads.

Top-level Torch-named artifacts/checkouts in Downloads, Desktop and
Documents/GitHub were also checked. None were present. Unrelated personal files
and unrelated home directories were not read. The user did not know another
local source path. These limits remain explicit; there is no whole-machine
absence claim or automatic expansion to network sources.

## What was found

| Observation | Result |
| --- | --- |
| Initial/incremental inspected paths | 136/15,151 distinct total |
| Package ZIP metadata observations | 120;115 saved offline dependency wheels and five cache wheels |
| Distinct wheel metadata snapshots | 28; none named Torch |
| Compressed cache bodies | Six complete package-index JSON catalogs |
| Plain cache bodies | Five distribution metadata records |
| Library source snapshots | Six current/archived source or metadata files |
| Version declarations | Two version.py records declare2.14.1/CUDA None; installed distribution metadata says2.14.1 |
| Version header | A forwarding include, without an independent version literal |
| Exact2.8.0+cu128 candidates /2.8 release references | Zero/zero |
| Inspection errors or reached caps | Zero |

All120 archives contain distribution metadata, and none has a root
`torch/version.py` or a Torch distribution identity. The archived packages are
other dependencies; their metadata does not constitute Torch source evidence.
The six compressed cache catalogs concern PyObjC components and pip. Catalog
links describe available packages, not payloads known to be stored locally;
no link was fetched. The five plain cache records are also other distribution
metadata. Complete bounded decoding resolves the initial gzip-prefix ambiguity.

The current environment's version source and its previous archived observation
both declare2.14.1 without CUDA. Corresponding documentation, distribution
metadata and the forwarding version header are retained as observations only.
No module or version file is executed. These files verify neither a loaded local
binary nor the pinned target's copy/export/synchronization semantics.

## Read-only handling and bounds

Inspected regular files have before/after identity, owner/mode/link/extent/
mtime/ctime observations. No source or cache file is changed. Symlinks and
special files are not followed/opened. Source snapshots and package metadata
are copied only into the new local evidence stage. Archive members are read
by name without filesystem extraction or executing package code.

Limits include50000 planned paths,4MiB ZIP central directory/20000 members,
2MiB selected text/metadata members, and bounded gzip catalog decoding up to8MiB
per catalog. No limit or inspection refusal was reached. No Torch archive needed
a whole-carrier hash because none was identified. Package metadata identities
are not binary integrity or target-installation authentication.

The inventory and its provenance pass use stdlib file, ZIP/gzip, JSON, AST and
SHA256 operations only. Ordinary source/metadata reads, decompression and hashing
are real local work. No Torch/NumPy import/install, package execution, model/
tensor/gradient/worker, FileSink/ledger lifecycle, benchmark, workload clock/timer/
signal/PIDFD/GPU/proof/network or paid work occurs. Previous tests, byte witnesses,
encoding comparison and scientific cohorts are not rerun.

## Preserved evidence and audit

Name lists, stderr, root coverage, both incremental plans/inventories, metadata
provenance,28 unique wheel metadata snapshots,11 cache metadata snapshots and
six library-source snapshots are retained. The independent stdlib saved audit v2
passes on70973 prior hashes,77682 prior fixture entries and55 new entries, plus
source/metadata/corpus/count/decision joins. It does not execute the scanner,
project/test/benchmark/scientific callbacks, or import inspected libraries.
Package catalog JSON is decoded; tensor/scientific-array values are not.
Auditv1 refused a late stdlib gzip import under its no-exec hook; only the
auditor preload was corrected. Both versions and the refusal remain, with no
scanner or package inspection repeated.

The historical FIFO remains LSTAT only, never opened. Device16777229 uses the
frozen previous observation; no mount probe/remap/waiver occurs. Old device
records, target66307 and four directory exceptions remain unchanged. All previous
sources/tests/auditors/failures/results/statuses/consumed claims, including actual
v3 proof/31-of40 partial failure, remain intact.

Raw stage: `runs/pretrained/diagnostic/pgd-target-source-inventory-v1-20261124`.
Seal: `sealed-target-source-inventory-v1.json`. Starting clean user commit
`828e7fa`. Mutable indexes are excluded with exact before/after snapshots. User
alone stages, commits and pushes.

## Evidence needed next

The hypothesis needs a known exact offline source/package artifact path, or a
separately requested task to consult official public source outside this local-
only task. Do not repeat this inventory without a new path or source-state
change, substitute newer documentation, infer ownership from readonly/copy
flags, or implement an adapter from missing evidence.

Even a version-matching artifact would still require source/build provenance,
target-installation joins, ownership/synchronization/lifetime/byte-contract review
and matched complete-unit timing/memory evidence. Source availability alone
cannot admit execution. No new operational route or optimization is selected.

Budget remains `REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE`. The v5 gate refuses before
protocol clocks, ATTEMPT or connection. No scope/package/grant/approval request/
launch command, SSH/query/upload/retry/resume/retrieval/cleanup/target/GPU/proof/
paid work follows. All real grants remain consumed. All original scientific,
ownership/hash/durability/resource/clock guards, training policies, registry/study
YAML and financial200/160/40/single10 caps/holds remain fixed.

Mechanism closed; full units/replay/timing/overhead/memory/recovery/economics/
calibration/data/protocol/training gated. Historical financial snapshots are not
current quotes/balances. No training percentage or time estimate.
