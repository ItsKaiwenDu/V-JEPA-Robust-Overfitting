# Bounded split-work evidence collection

The collector and private local decoder now use explicit file extents, per-file
hashes and a terminal receipt. A source-only package contains all six actual
supervisor files and the 63 unchanged runtime modules. This closes the collection
protocol checkpoint, not the complete target helper. No network, target process,
proof, scientific computation or paid work ran.

## Stream and final receipts

`predictor_pgd_split_collection_v1` has no CLI or connection API. Its future caller
must supply the owned scratch guard and exact current control/authorization bytes.
The collector checks saved control, authorization and derived installation bytes;
it joins the ended parent session and original serial completed-role prefix into a
combined work receipt. A failed prefix stays failed. Work receipts always mark
collection incomplete and independent scientific review pending.

Two exclusive records use existing declared result paths: `final-receipt.json`
and `collection-status.json`. Preparing or streaming consumes that object's one
attempt even on failure. No receipt is overwritten; failed publication names and
old results are preserved. Only a successful publication removes its own temporary
link name, matching the existing store convention. There is no cleanup command.

A typed inventory retains all scratch file and directory metadata and existing
capacity/category accounting. Evidence contents include the declared sources,
input packet, control, session, bounded logs, results and permitted partials.
Cache/tmp and emergency-reserve contents are excluded explicitly; their inventory
metadata is retained. A leftover proof extent or a tree exceeding the original
collection/storage policy causes refusal. This does not promise collection of
arbitrary failed setup trees.

The protocol is a fixed magic, a canonical JSON manifest of at most 1 MiB, file
bytes in sorted manifest order, and a distinct terminal frame of at most 16 KiB.
The entire wire response, including framing, is capped at 3 GiB. There is no target
archive file or general archive extraction. The manifest must match every original
source/control descriptor, the whole-scratch inventory and its category/extent
limits. File reads use held no-follow descriptors and checks of private ancestry,
named/held identities, exact extents, mode, links and nanosecond timestamps. Reads
are chunked at 1 MiB; sources are hashed before and during the stream. A final
unchanged-tree observation precedes the terminal frame. Short writes are completed
within the existing byte/time bound; broken, zero or invalid writes refuse.

The terminal frame binds control, authorization, manifest, combined work receipt,
file count, payload length and the preceding wire hash. It does not include its own
hash or mutate already streamed evidence. A receiver receipt can hash the entire
wire after the terminal frame, avoiding a self-reference. A producer receipt
cannot claim that the receiver actually got the bytes.

## Private reconstruction and continuing clocks

The local decoder accepts only the exact original framing. Empty responses,
noncanonical/duplicate/nonfinite JSON, missing or changed files, truncation,
trailing data and a missing or mismatched terminal receipt refuse. A fresh private
local directory is required. Writes are exclusive 0600 files in 0700 directories;
nsmtime is preserved. Permitted target publication aliases are copied as distinct
private single-link files, with both names retained. A copied partial remains a
partial; transport verification never admits scientific completeness.

Review reproduced a terminal-EOF gap in the first decoder: earlier output files,
the named root or its parent could change before return. The corrected decoder
rechecks the named parent/root and performs a bounded held/named tree, file type,
extent, mode, nsmtime and hash verification after EOF. Extra files/directories also
refuse. Every negative fixture and partial output is preserved. These checks are
consistency observations, not an atomic snapshot or protection against arbitrary
concurrent same-UID code.

The original connection origin fixes its 3,000-second deadline. One collection
origin fixes at most 600 seconds, clipped by that connection deadline; neither
streaming, local final verification nor failure resets it. Manifest preparation
also stays within the original 2,400-second work deadline and a 120-second final
phase. All byte loops check the continuing clock. These are cooperative checks:
the missing outer driver must still enforce alarms for blocking reads/writes,
login-inclusive client timing, and work-to-collection signal/ownership handoff.
Local and target monotonic clocks cannot be interchanged without that driver.

## Concrete source package and validation

`predictor_pgd_split_source_bundle_v1` reads only bounded local source/metadata.
The five previous supervisor sources have fixed digests; the current collector is
bound by the new control. All 63 runtime bytes retain their frozen descriptors.
Compilation is cold only. Output has 69 Python source files and four metadata/
configuration files, all private. `CONTROL-CANDIDATE.json` and
`INSTALLATION-CANDIDATE.json` deliberately remain unarmed; no authorization or
attempt claim is created. The pinned importer can cache all 69 names without
installing the finder or executing a module. No training pixels, wheels or
checkpoint are read by the source assembler.

97 distinct local checks have passing latest results across 114 executions. Eight
failing executions are retained: two initial fixture mistakes, four reproduced
EOF consistency gaps, one assembler nested-directory mode defect and one named
parent replacement gap. The assembler now creates each directory explicitly as
0700, including intermediate parents. Nine passing executions were repeated only
for changed source/hash or affected successful transport paths. Historical tests
and scientific cohorts were not rerun. Tests use tiny inert payloads, explicitly
substituted asset descriptors and synthetic metadata for the complete 172-asset
geometry. They do not establish actual installation, storage allocation, process
or GPU behavior. Raw snapshots, JUnits, damaged streams, source packages and
negative filesystem fixtures remain under the new collection diagnostic directory.

The independent review uses only standard-library saved-file/AST inspection and
hash checks, with project execution and process/network operations blocked. It
also verifies earlier byte/mode/nsmtime records and handles the old FIFO through
lstat metadata only, never opening, hashing or reading it.

## Remaining before a target attempt

The outer local/remote one-attempt mechanism must prevent authorization reuse
across roots. It must bind the packet/source/control setup, install the pinned
parent namespace, preserve the original clocks through login/setup/work/collection,
and persist bounded GPU/storage observations and refusal journals. Those pieces
are not implemented by this collection module or source assembler. The complete
outer package still needs independent review before one exact new university scope
can be requested. No repeated manual GPU check or new approval is needed now.

All original numerical bounds, backend/ownership gates, prior outcomes and partial
rejections, consumed grants, financial limits/holds/reconciliation and training
restrictions remain unchanged. Actual scientific, CUDA/backend, accuracy, full
EOT8/replay, performance, cost and training admissions remain false.
