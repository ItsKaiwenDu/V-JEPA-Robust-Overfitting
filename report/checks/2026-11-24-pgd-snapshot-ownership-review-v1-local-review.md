# Snapshot ownership: local source review

**Defer the owned-buffer hashing proposal; saved audit v2 PASS.** A read-only view does not establish
an immutable snapshot, and the available local Torch source does not match the
pinned target. Preserve the existing `tobytes` copy and every fresh byte hash.
No new hashing adapter or runtime route is adopted.

See [source/proof review](../../docs/predictor_pgd_snapshot_ownership_review_v1.md),
[requirements](../../configs/predictor_pgd_snapshot_ownership_review_v1.json) and
[machine-readable review](2026-11-24-pgd-snapshot-ownership-review-v1-local-review.json).

| Review item | Finding |
| --- | --- |
| Existing source binding | Six pins; original tensor_hash/Snapshot.check ASTs unchanged |
| New stdlib witnesses | Seven opaque-byte observations, one execution |
| Read-only view | Rejects direct assignment; still changes through writable aliases/exporter |
| Private copy | Survives writes to original, still vulnerable to a retained writer to the copy |
| Lifetime | Released view access refuses |
| Immutable bytes baseline | Retains captured bytes after original source changes |
| Local source observation | version.py declares2.14.1/CUDA None; no Torch import or loaded-binary verification |
| Target | Frozen provider requires2.8.0+cu128/CUDA12.8; not qualified by local source |
| Performance, peak memory, physical tensor-copy proof | Unqualified |
| Current runtime/client/scientific/hash/budget changes | None |
| Preservation basis | 70899 prior hashes/77650 prior fixtures/32 new entries |

The hypothesis is to avoid a further bytes materialization only after a private,
completed CPU snapshot is proven. Constructor choice, ownership, synchronization,
logical-byte order, dtype/layout refusals and retained lifetime all need exact
target-version evidence. Private-copy or exporter retention costs might erase
any benefit. Do not cache hashes, weaken guards or infer cost from partial
gradients. `state_digest` retains its separate ordered-name/byte contract.

Seven stdlib counterexamples save before/after bytes and SHA256. They establish
ordinary byte ownership behavior, not Torch/NumPy or CPU/CUDA tensor behavior.
No original project test, scientific cohort, prior encoding benchmark, FileSink/
ledger/worker lifecycle, workload clock/timer/process/signal/GPU/proof/network
port runs. An unexecuted draft syntax correction is preserved before the sole
witness run. Ordinary source reads, metadata FS/AST/SHA256 and byte work are real.

The standalone saved audit verifies prior preservation, all new fixture stats,
source pins/ASTs, archived version evidence and every witness byte/hash join.
It does not replay the witness or import project/test/scientific callbacks,
decode tensors/scientific arrays, recalculate science or use subprocess/timer/
signal/GPU/proof/network ports. Historical FIFO LSTAT only; device29 via prior
frozen observation, with old/target66307 and four lifecycle exceptions fixed.
No new mount probe, remap or blanket waiver occurs. Auditv1 used a missing
root-path key for diagnostics; v2 joins the byte-exact helper through the pinned
runtime overlay. Only the auditor changed; both histories remain, no witness rerun.

The next separate local step is an inventory of exact pinned-target source
evidence already local. Missing evidence stays missing; no2.14 substitution,
Torch/NumPy import/install, target query/fetch, scope/grant or adapter adoption.
No replacement operational route or performance qualification is selected.

Raw: `runs/pretrained/diagnostic/pgd-snapshot-ownership-review-v1-20261124`.
Seal: `sealed-snapshot-ownership-review-v1.json`; mutable indexes excluded with
exact snapshots. HEAD8416b66 unchanged, prior uncommitted prototype/rejection and
all histories/failures/results/consumed claims preserved. User alone stages/
commits/pushes. Actual v3 proof/31-of40 partial failure remains unchanged.

Budget remains **REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE**, installed v5 gate closed.
No launch/approval/remote/paid work. All scientific policies, clocks/resource caps,
study and financial caps/holds remain fixed. Mechanism closed/research-training
FALSE; actual full units/replay/timing/overhead/memory/recovery/economics/
calibration/data/protocol/training still gated. No percentage or time estimate.
