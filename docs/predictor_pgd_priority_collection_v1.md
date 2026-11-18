This local source-only protocol candidate addresses the evidence-collection
failure after the November 17 PGD attempt. It does not change consumed controls,
claims, source bundles or results, and provides no connection/CLI/grant API.
The new module delegates full source/control/inventory/clock validation to the
unchanged reviewed legacy collector. It has a distinct magic/header/frame/terminal
format; neither endpoint guesses or silently accepts the old wire format.

The manifest records explicit deterministic order: session.json, combined work
receipt, collection status, control, authorization and installation first, then
other declared metadata, traces, source/input and raw bytes. Canonical JSON can
sort dictionary keys without changing the ordered record list. Every path keeps
its own original hash/byte count/mode/nsmtime descriptor and full inventory row.

The first file with a given SHA-256 and byte count is DATA. Later duplicates
reference that first earlier DATA path only. Reference chains, forward/stale
references, unknown or repeated records and mismatched counters refuse. The
sender still reads/hashes every exact source, including each reference, before
its frame is sent; deduplication cannot bypass source mutation checks. The
receiver rechecks the completed source's held/named inode, owner, mode, device,
links, size and mtime, copies bounded chunks into a new exclusive private file,
and verifies its own hash/extent/nsmtime. No hardlink, symlink or shared output
inode is used to implement deduplication. Empty duplicates are supported.

Before any evidence file, the receiver saves its validated canonical HEADER.json
and an explicitly incomplete PARTIAL-RECEIPT.json in a fresh private sibling
journal. VERIFIED-FILES.jsonl is appended/fsynced only after each member passes;
rows contain an ordered hash chain and remain transport_verified/stream_complete
false. The journal never claims full transfer completion, even after success.
A full transport receipt is returned only after the exact terminal/prefix hash,
EOF, final evidence-tree hashes, journal hashes, root/parent identities and final
allocated-disk sample all pass inside the caller's continuing window. The future
outer client must persist that return receipt under its original deadline.
All failures preserve existing output; the inherited exact reader may buffer
an unverified final short chunk without writing it. The journal is a verified
file-prefix record, not a forensic capture of every received byte.

Limits remain 3 GiB wire and expanded/disk envelope, 1 MiB header/chunks, original
120-second preparation, 600-second collection and 3000-second connection origins.
The new 4096-byte record cap, 16 KiB terminal, 4 MiB progress and 8 MiB local
journal reserve are fixed bounded subsets. The journal reserve is charged within
the original 64 MiB metadata/trace ceiling and 3 GiB local disk envelope. The
no-dedup expansion plus framing must fit too, preventing a small wire payload
from authorizing larger reconstruction. Logical expansion is bounded throughout;
allocated usage is a final bounded sample, not a kernel quota or continuous
allocation guarantee. File/source buffers are at most 1 MiB; full model data is
not materialized. Cooperative checks do not interrupt blocking I/O: the future
outer driver must retain its independent hard alarms and consumed attempt guard.

55 distinct new checks pass across 58 executions. Two assertions were corrected
without changing production source: JSON canonicalization turns tuples into
lists, and the old exact reader buffers incomplete chunks rather than writing
their unverified tail. A broad v2 selector unintentionally repeated one passing
preparation-clock check; that extra execution is explicitly retained and reported.
Subsequent tests use exact node names. No historical/scientific cohort repeated.
All initial and tested source revisions, failure logs, fixtures and JUnit remain.

A real unmodified validator checks a full 172-asset/177-file synthetic metadata
header. Payload round trips substitute tiny inert byte files for real assets,
with fake role receipts and clocks. Actual owned file/descriptor/hash/fsync/link
refusal and copy paths run, including fragmented I/O, source mutations, truncated
and corrupted frames, journal tampering, metadata/expansion/disk limits and EOF.
No arrays are decoded; no model, proof, process, timer, network, SSH or paid work
runs. The source-only raw scanner candidate remains separate and unchanged.

Next is a fresh LOCAL complete binding: integrate the reviewed scanner and paired
codec into one policy/control/bootstrap/cache/journal/reader/client/outer bundle.
Resolve every import and validate the new nested header/magic/receipt handling
before any attended handoff. Old outer/client prefix parsers cannot accept this
new protocol unchanged. Preserve scientific code, references, ownership and
resource gates. Independently review that complete local binding before a new
exact scoped university approval. This prototype is not native/full-runtime,
transfer-time, full eight-gradient PGD, replay, performance or training admission.

Separate saved-file/source/wire review passed; see
report/checks/2026-11-18-pgd-priority-collection-v1-local-review.md.
