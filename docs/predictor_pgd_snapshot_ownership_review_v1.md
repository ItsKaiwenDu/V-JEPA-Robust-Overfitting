# Tensor-hash snapshot ownership review

**Defer owned-buffer hashing until versioned ownership and cost proofs exist.**
Keep the current immutable `tobytes` snapshot and every fresh tensor hash. No
hashing adapter, runtime change or new scope is implemented or adopted here.

See the [local review](../report/checks/2026-11-24-pgd-snapshot-ownership-review-v1-local-review.md)
and [proposal requirements](../configs/predictor_pgd_snapshot_ownership_review_v1.json).
The previously rejected phase-row encoder and its local cost results remain
preserved and uncommitted. This review does not revive that candidate.

## Source-level question

The bound `tensor_hash` observes bytes through the original sequence:
`detach → cpu → contiguous → reshape(-1) → view(uint8) → numpy → tobytes → SHA256`.
`Snapshot.check` requires this fresh hash in addition to version, shape, dtype,
device, object/API, parameter-gradient, mode and hook checks. Those observations
must remain; a cached hash or unchanged version cannot replace the byte check.

A possible future proposal would first establish a private, completed CPU
snapshot, then hash a retained byte view of that snapshot without a further
bytes materialization. This is a hypothesis, not a selected constructor/API or
an implementation. It has no demonstrated cost or memory benefit. Creating the
private snapshot or retaining buffers could erase the benefit or increase peak
memory. No phase evidence identifies this as the dominant target cost.

The current `state_digest` has a different contract: it includes state names in
iteration order and its own tensor byte path. This review does not unify or
change that function. The original uint8 **reinterpretation** is also distinct
from numeric dtype conversion; any future proposal must preserve the original
logical-byte order and refusal behavior, not just numerical values.

## Seven new opaque-byte witnesses

One standalone stdlib script executed seven deterministic witnesses. They use
ordinary `bytearray`, `bytes`, `memoryview` and SHA256 only. Inputs are synthetic
opaque bytes, not model/tensor/gradient values. No original project test,
scientific cohort, prior benchmark or worker lifecycle was rerun.

| Witness | Saved observation |
| --- | --- |
| Write through a read-only view | TypeError; bytes unchanged |
| Write through the original mutable alias | Read-only view bytes and digest change |
| Write to the original after constructing a private copy | Private copy's view remains unchanged |
| Write through a retained alias to the private copy | Its read-only view changes |
| Write through the mutable exporter exposed by `view.obj` | Read-only view changes |
| Access a released view | ValueError |
| Write to the source after constructing immutable bytes | Original bytes snapshot remains unchanged |

The result is narrower than a concurrency or tensor proof: a read-only flag
blocks writes through one view, but does not authenticate an immutable snapshot,
exclusive ownership or lifetime. A private copy separates it from the original
source, while any surviving writer to the copy still matters. No threads or
races are needed for these counterexamples, and none were started.

Before/after bytes, hashes and outcomes are saved individually. The script ran
once; an unexecuted draft and a pre-execution keyword-syntax correction remain.
No stopwatch, workload clock, timer, process/PIDFD/signal, GPU/proof or network
port is invoked. Ordinary metadata filesystem/source/AST/hash work and these
stdlib byte operations are real.

## Local source is not target evidence

The local `.venv/.../torch/version.py` source declares2.14.1 and CUDA None.
The installed documentation source was read and archived with hashes/stat
observations, without importing Torch or executing the version file. This does
not verify a loaded local binary, and it does not establish the target's API or
storage/synchronization behavior.

The frozen target provider still requires Torch2.8.0+cu128 and CUDA12.8. Newer
local copy/export documentation cannot qualify that pinned target. No source
version substitution, package installation, import, remote query or download
occurred. No actual CPU/CUDA tensor copy or exporter was tested.

## Proof obligations before any adapter

1. Exact target-version copy/export evidence, including completed host transfer
   and CPU/CUDA aliasing behavior. A copy request or metadata flag is not proof
   that a physical private snapshot exists.
2. No writable alias or exporter escape during trusted hash consumption; keep
   the snapshot/exporter/view alive until hashing ends, including failures.
3. The existing contiguous logical-byte order, flattening, uint8 reinterpretation,
   SHA256 and refusal behavior across the originally accepted device/dtype/layout
   domain. No new numerical conversion, asynchronous observation or fallback.
4. Every fresh hash and all identity/version/owner/RNG/mode/native guards,
   durability, continuing clocks, deadlines and resource caps remain intact.
5. Independent semantic/byte evidence and matched full-unit timing/peak-memory
   evidence. Small local metadata timings and the old31-of40 partial gradients
   cannot predict the result.

The next separate local step can inventory exact target-version source evidence
already available locally. If absent, record the gap. Do not import/install
Torch/NumPy, substitute local2.14 sources, query/fetch the target, adopt an adapter
or create a scope/grant. No next runtime route is selected by this review.

## Preservation and limits

Six existing source pins, the original hash/check AST anchors, two archived local
source observations and seven byte witnesses support the review. The standalone
stdlib saved audit v2 passes on70899 prior hashes,77650 prior fixture entries and32 new
entries, plus source/byte/version/decision joins. It invokes no project/test/
witness/scientific callbacks, tensor/scientific-array decoding or recalculation,
process/clock/timer/GPU/proof/network ports. Auditv1 refused a root-path lookup for diagnostics; that helper is sealed through
the frozen runtime overlay. Its exact bytes match, and only the auditor join
changed for v2. Both versions/refusal remain; no witness reran.
The historical FIFO is LSTAT only,
never opened. Device16777229 uses the frozen prior observation; no mount probe,
remap or waiver, and target66307/four directory exceptions stay unchanged.

Raw: `runs/pretrained/diagnostic/pgd-snapshot-ownership-review-v1-20261124`.
Seal: `sealed-snapshot-ownership-review-v1.json`. HEAD8416b66 and all prior
uncommitted work remain preserved. Mutable indexes are excluded with exact
before/after snapshots. User alone stages, commits and pushes. All prior source/
test/auditor versions, failures, results, consumed claims and actual v3 proof/
31-of40 partial failure stay unchanged.

The active56-source/48-file runtime/cache47/nine-supervisor/270-path graph and
all existing scientific/hashing/ownership/resource/clock guards remain fixed.
Budget is still `REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE`; the v5 client refuses before
protocol clocks, ATTEMPT or connection. No scope/package/grant/approval request/
launch command, SSH/query/upload/retry/resume/retrieval/remote cleanup/target/
GPU/proof/paid work follows. All grants remain consumed.

Mechanism closed; complete units/replay/timing/overhead/memory/recovery/economics/
calibration/data/protocol/training remain gated. All scientific policies,
trainingPGD5/EOT1, routineEOT1, optimizer backward, registry/study YAML and
financial200/160/40/single10 caps/holds stay fixed. Historical financial snapshots
are not current quotes/balances. No training percentage or time estimate.
