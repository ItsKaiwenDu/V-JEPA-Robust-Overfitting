# Raw-correlation component worker v4

This separate component worker integrates the frozen raw Kineto collector/verifier
without changing the original arithmetic, native input reconstruction, analytical
reference, numerical bounds, runtime/precision, flags, fresh replay or startup
owner. The v3 worker, previous collectors, failed evidence, consumed claims and
clocks remain unchanged. No university handoff, control, grant or attempt is
created at this checkpoint.

## Execution and saved trace contract

The v4 plan explicitly requires `fixed-SDPA-raw-Kineto-correlation-v1` with
`single-JSON-envelope` encoding. Each fresh worker still runs one untimed and one
profiled SDPA forward/VJP. The first creates one canonical FP64 reference; the
other three reuse it, in original-0, math-0, original-1, math-1 order. There are no
new warmups or throughput loops. The original component numerical gates remain:
output rtol 1e-4 / atol 2e-5; dQ/dK/dV rtol .002 / atol 2e-7; bookkeeping
rtol 1e-12 / atol 1e-14. Each worker remains bounded to 120 seconds.

The raw envelope is saved as one JSON line in the existing `events.jsonl` slot,
retaining its 8 MiB limit and exclusive/atomic bounded writer. Its descriptor
binds byte hash, format, encoding, record count and PyTorch version. The worker
reads those exact saved bytes back before backend verification. Loading a result
rechecks all these fields against its validated runtime and the new raw schema;
legacy JSONL and a changed runtime/count/format cannot silently substitute.
The original flags/input completion checks still run after backend evidence.

The actual raw collector is unchanged from the separately reviewed checkpoint.
It reads the completed profile and explicit CPU/GPU correlation links, with no
name/duration or nearest-time identity fallback. Captures with missing APIs or
invalid metadata refuse. A failed capture keeps all already-saved numerical
returns and the failure marker; unavailable raw data is not invented. A later
verifier refusal also keeps the saved raw envelope. Neither failure publishes a
completed worker or resource receipt.

## Resource receipt and storage

The separate target wrapper keeps the exact linux10603/Python/device entry guard,
single-owner strict pre-CUDA startup and 7 GiB allocator cap. After its own completed
worker returns, it validates startup, peak and deadline, then publishes a new v4
resource receipt. The new publisher explicitly uses a private creation mask and
restores the caller's mask on success or failure. It no longer relies on a parent
process having already selected that mask.

A bounded readback checks regular/private/single-link file metadata, stable bytes,
exact schema, worker file hash, plan, index, PID, startup, typed peak, elapsed work
and no-admission fields. No tensor or CUDA operation is needed to read a receipt.
The reader does not independently establish scientific correctness: its caller
must already validate the completed worker and its actual arrays/provenance.

The trace slot, raw-array and metadata limits stay unchanged. Actual new CPU
artifacts pass the frozen storage policy and collection codecs under an explicitly
local rehearsal identity. All collected bytes, hashes, modes and nanosecond times
match. This codec exercise is not a new university handoff or reuse of an old grant.
CPU workers never manufacture a CUDA resource receipt.

## Local validation and remaining work

45 focused integration checks pass. Four genuinely fresh small CPU processes
complete numerical, raw profiling, flags/input and exact profile/fresh replay
checks, with one shared FP64 reference. The final traces contain 75, 249, 75 and
249 raw records; retained duplicate CPU nodes explain the difference from older
processed math traces. The tests also cover rehashed false completion/trace claims,
missing raw APIs, source/format/runtime binding, bounded partial-file retention,
local CLI refusal on university hosts/CUDA, actual storage/collection, and typed
resource receipt tampering and private publication.

The real wrapper/executor startup test simulates CUDA state and stops before
operands. Resource fixtures simulate only the CUDA boundary. They do not measure
GPU allocation, qualify actual CUDA numerics/backend behavior, or establish target
PyTorch 2.8 compatibility. Actual scientific work in this checkpoint is local
CPU component work under PyTorch 2.14.1. Original accepted native inputs are reused;
there is no new full-model capture, training or held-out data access.

The first run (27 passed / 16 failed) and sources remain. Its strict reader exposed
that receipt creation used the caller's mask (0644 under the local test runner).
The new publisher makes private creation explicit; the reader stays strict. The
second run (44 passed / one failed) and sources remain: its hostile Infinity test
was rejected by the canonical JSON writer before reaching the reader. That fixture
now deliberately bypasses the canonical writer to exercise the intended readback
refusal. Final 45 passes are not added to earlier counts. No numerical tolerance
or frozen source was changed.

Independent saved-data review verifies the final actual arrays/reference, raw
profiles and proofs, storage/collection copies and resource fixtures while blocking
model/autograd/SDPA/CUDA/process/network/reference execution. Previous evidence,
refusals and consumed claim stay unchanged. No old metadata/scientific suite repeats.

Next prepare a separate university handoff using the unchanged accepted native
capsule and this verified worker. Test its cold bootstrap, exact payload, live
guard wiring, serial deadlines and same-SSH collection before asking for one new
scoped attempt. Full training, the 3,724-entry candidate rejection and financial
caps/holds/reconciliation remain unchanged. Never reuse a consumed helper or claim.
