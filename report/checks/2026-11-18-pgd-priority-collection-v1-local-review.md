The local paired priority/deduplication protocol now passes its focused checks
and separate saved-source/wire review. It sends session.json, the combined work
receipt and collection status first, before large raw data. Explicit record
ordering survives canonical JSON key sorting. Duplicate content references the
first earlier verified DATA path with the same SHA-256 and byte count. Every
source, including references, is still read/hashed against its own manifest and
snapshot. The receiver creates independent private single-link files and keeps
each original path/hash/size/mode/nsmtime. No hardlinks implement deduplication.

The receiver saves the validated header and incomplete partial receipt before
writing evidence, then fsyncs a hash-chained verified-file journal. The journal
always remains transport_verified/stream_complete false. A complete transport
receipt is returned only after terminal/prefix hashes, EOF, full copied-tree
and journal hashes, root/parent identities, allocated-disk sample and original
window checks pass. Failed output remains preserved and old paths refuse
exclusive creation. The inherited exact reader may retain a short unverified
chunk only in memory; the journal attests a verified file prefix, not every byte
read from the connection. No on-disk complete marker is minted by the codec.

Forward/chained/wrong-content references, changed or replaced prior files,
symlinks/hardlinks, moved ancestry, corrupt/truncated/noncanonical frames, extra
bytes, journal mutations and exhausted byte/disk/time bounds refuse. The original
legacy validator remains mandatory, with all source/control/asset/inventory and
clock checks unchanged. The raw scanner candidate and consumed results are
unchanged; the two candidates are not yet a complete runtime/source binding.

Limits: original 3 GiB wire/expanded disk envelope, 1 MiB header/chunks,
120-second preparation, 600-second collection and 3000-second connection.
Records cap at 4096 bytes; terminal at 16 KiB. The 8 MiB journal reservation is
inside the existing 64 MiB shared metadata/trace and 3 GiB disk bounds; progress
caps at 4 MiB. No-dedup expansion plus framing must fit too. Cooperative checks
do not interrupt blocking I/O; the future outer must preserve hard alarms and
attempt ownership. Allocated space is a final bounded sample, not a kernel quota.
A smaller payload does not demonstrate a future transfer-time guarantee.

55 distinct new cases pass across 58 executions. Two assertions were corrected:
canonical JSON normalizes tuple/list representation, and the old exact reader
does not write its short buffered tail. Production source stayed unchanged after
its first tested revision. One passing preparation-clock case was unintentionally
repeated by the v2 selector; that extra execution is disclosed in JSON, logs and
JUnit. Subsequent cases used explicit node IDs. No historical/scientific cohort
was repeated. All initial/tested source revisions, failed fixtures and logs stay.
Full-header checking uses the unmodified validator and 172 pinned assets /
177 declared files as synthetic metadata, without materializing payloads.
Round trips use explicit tiny asset substitutions, fake role/clock metadata and
a fake allocation sample in one negative case. Real owned byte copies, hashes,
private modes, mtimes, descriptors and journals run. No arrays are decoded; no
model, proof, process, timer, network, SSH, paid work or new grant is involved.

The separate standard-library saved-file/hash/AST audit passed on its first run.
It verifies all 16,629 previous hashes and 15,626 prior fixture entries, including
lstat-only handling of the historical FIFO. It independently reconstructs the
saved tiny-fixture wire and its terminal/prefix hash, verifies source/copied
files, independent copy inodes, per-file journal hashes/counters and the surviving
status prefix after truncation. It rechecks the original 172 asset metadata pins
and two zero-owned-descriptor records. It executes no project/test callbacks.

Next is LOCAL complete integration: bind both scanner and codec into a fresh
policy/control/bootstrap/cache/coordinator/reader/client/outer package. Resolve
all legacy-helper dependencies, new magic, nested header, journal and receipt
shapes with cold source/cache and inert lifecycle checks before any attended
scope. Do not rely on imports from the local cwd or substitute modules into old
consumed bundles. No university handoff, approval request or command is prepared.
Full eight-gradient PGD, component/replay/performance/native runtime and training
admission stay pending. All original evidence, numerical/resource gates and
financial caps/holds/reconciliation remain unchanged.
