# Bounded EOT8 ownership records v2: offline correction

The [consumed target attempt](../report/checks/2026-11-06-predictor-owned-target-v1-target-review.md)
saved all ten original-model tensors, then failed when a single result JSON
exceeded its frozen 1 MiB limit. The full call rows and final ownership receipt
were lost. This correction changes how new evidence is stored; it does not
reconstruct that lost receipt or change the old failed result.

## External evidence and final completion

`models/predictor_eot8_record_v2.py` is a new stdlib-only writer/reader. It stores
three bounded snapshots and a final marker:

| File | Purpose | Maximum |
| --- | --- | --- |
| `attention-calls.jsonl` | Every call row in original index order | 4,096 rows × 2,048 bytes = 8 MiB |
| `attention-scopes.json` | Five graph scopes, up to 32 refusals, identity and call-file hash | 64 KiB |
| `worker.json` | State, artifact descriptors, file hashes and byte/count receipts | 256 KiB |
| `worker-complete.json` | Final success binding to the exact worker header | 16 KiB |
| `worker-failure.json` | Failed completion or bounded writer-error marker | 16 KiB |

The existing global operational JSON limit remains 1 MiB. The new worker
header has a lower ceiling. Tensor, trace, aggregate metadata/capture storage,
GPU allocator and numerical limits are unchanged. Call rows remain in their
original order and are never truncated or sampled.

Each snapshot uses an exclusive private `.partial` file, flush, file fsync,
atomic replacement and directory fsync. It preserves the previous final file
until replacement. Earlier durable call rows must remain identical. Reused
writers, mutated snapshots, changed identities, links, foreign files, invalid
JSON, duplicate keys, nonfinite metadata, excessive rows or bytes cause refusal.
An interrupted partial is preserved and cannot be silently reused.

After call and scope files are durable, the writer publishes the small header.
A passing final result additionally requires the separate completion marker,
which binds the exact header hash and worker identity. A success-looking header
with a missing marker, or any failure marker, cannot be accepted as a reference.
The reader verifies all file hashes, extents, row order/counts, status/control
bindings and the unchanged ownership validator before returning a passing view.
It refuses legacy inline records. Loading a failed snapshot for inspection does
not qualify it as a reference.

For normal scientific failures, the writer saves the actual failed graph scope,
refusal metadata and any earlier numerical artifacts. If storing those records
also fails, a separate small failure marker preserves the original error type
and a byte-bounded message. The original scientific exception remains the raised
error; the secondary writer failure cannot replace it. No completion is assumed
when the filesystem cannot save a marker. Operator exit-status and collection
checks remain necessary in a future live handoff.

## Storage within the existing envelope

Four maximum call files plus one active atomic replacement use 40 MiB. Counting
five possible scope/header copies and conservative completion/failure markers
bounds these record files at **43,712,512 bytes**, below the existing 64 MiB
metadata category. This leaves 23,396,352 bytes before other metadata. The
existing aggregate gate must still account for all logs, plans, metadata,
allocated blocks, free space and inodes; this is not a capacity reservation.

Tests write four maximum-size 4,096-row files and a full active snapshot copy,
then apply the frozen storage gate. The complete 8 MiB call file also passes a
round trip through the frozen same-SSH collection encoder/decoder, preserving
hashes, modes and exact nanosecond modification times. No live SSH is involved.

## Local integration and tests

`models/predictor_eot8_record_cpu_v2.py` is a separate local CPU-only integration.
It refuses university hosts and live/CUDA execution before scientific imports.
It retains the frozen five-pass objective, eight draws/two families, graph-owned
route, original numerical bounds, typed before-gate returns, ownership/event
readers, model/target checks and fresh-reference/replay order. The new reference
path loads and validates the split evidence before model setup. It keeps the
original semantic ownership identity and adds a required `record_format` tag
and final marker; old records cannot substitute.

**33 distinct checks pass**, with zero errors, failures or skips. They cover:

- Reproduction of the old 1 MiB overflow using explicitly synthetic full-model
  metadata with 2,736 rows, then lossless external storage of every row.
- The 4,096-row/2,048-byte ceiling, one-row-over and byte/count refusals, aggregate
  storage bounds and complete maximum-volume collection.
- Atomic publication, retained partials, immutable earlier rows, corrupted or
  missing evidence, mismatched controls, unsafe paths/links and legacy records.
- Actual checkpoint backward failures, saved loss/failed scope/hook removal,
  preservation of an original error when the writer also fails, and refusal of
  a final header whose completion marker could not be published.
- Four fresh real tiny CPU workers, all five passes, exact reference and replay
  checks, independent checking of retained event ownership, and before-model rejection
  of an incomplete reference.

All **40 actual FP32 CPU loss/input returns exactly match their corresponding
saved v1 CPU tensors**. Twenty graph scopes complete with hooks removed. Saved
review checks 24 within-process equalities, 20 original-bound comparisons,
20 fresh-process equalities and raw CPU event ownership. Model parameters and
targets remain unchanged, with no parameter-gradient accumulation.

Full-volume metadata fixtures are visibly marked synthetic, contain no computed
CUDA results and cannot pass the reference reader. They are capacity tests,
not reconstructed target evidence. The initial 30-case run caught a local
collection-fixture permission mismatch; the child fixture now uses the existing
operator's private-file umask. Long Unicode errors are bounded by serialized
bytes. The intermediate 31-case run and earlier source snapshots are retained;
only the final 33 distinct checks are counted.

## Remaining work

Independent saved-file review runs with model/backward/SDPA/CUDA and
process/network execution blocked. It rechecks all 5,734 target-review pins and
the consumed claim before reading new files. Scaffold/YAML and diff checks pass.
Local arm64 / Python 3.11.16 / Torch 2.14.1 CPU evidence does not qualify the
university's CUDA runtime, full pretrained record behavior, memory or performance.

Next prepare a separately named live worker, bounded operator/client and exact
unarmed payload using this format. Their reference acceptance, completion-marker
handling, file/category caps, cold imports, continuing deadlines and collection
must be reviewed before any new target-attempt request. No live worker entry,
archive, SSH/upload, storage proof, resource grant, claim, cleanup or paid compute
is created by this offline correction. Both university hosts may idle. All
captured sources, failed claims, numerical gates, historical research gaps and
financial caps/holds remain unchanged.
