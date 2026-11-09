# Raw launch-correlation evidence v1: local review

A separate read-only collector records raw Kineto CPU correlation IDs, GPU linked
correlation IDs, event indices, stream/device provenance and integer nanosecond
clocks. It does not choose a kernel by name/duration or nearest time. The original
collectors, workers, scientific gates, consumed claims and refused results remain
unchanged. No new target worker, handoff, authorization or attempt is prepared.

54 focused checks pass, with zero errors, failures or skips. Synthetic fixtures
exercise equal-name/device/duration launches with different recorded CPU owners,
efficient/math/CPU-flash families, cross-thread and parent-contained autograd,
duplicate CPU wrappers, raw enumeration order, foreign/missing/ambiguous IDs,
invalid fields/clocks, legacy/async refusals, bounded capture and cold assessment.
Raw CPU nodes are retained, so processing cannot lift a kernel away from its
original owner. Each CUDA row uses its explicit host correlation and row identity;
individual device correlation IDs are not assumed globally unique per launch.

The only new actual profile is a tiny local CPU allocation/addition probe: nine
records under installed PyTorch 2.14.1, with zero SDPA calls, backwards or model
calls. It validates adapter APIs locally; it does not qualify the university's
PyTorch 2.8/CUDA environment. No old scientific suite or target work reruns.

The first run's 46 passes/two failed assertions and source snapshots remain.
One assertion compared raw floating microseconds instead of the existing rounded
nanosecond representation; the other expected a schema refusal from historical
JSONL, which fails earlier as multiple JSON documents. Only those assertions
changed; six added regressions yield 54 final passes. Collector and CLI bytes
stay identical to their first run. No numerical tolerance or backend rule changes.

Independent saved-only review blocks scientific imports, collection/profiling,
processes and network. It verifies 45,760 prior source/evidence hashes, all 187
latest target-input descriptors, private target metadata, consumed claim and
incomplete worker. The real 106-row trace still triggers the frozen v2 ambiguity
refusal. The new schema refuses absent old IDs; no identifiers are reconstructed
and no target event, result, missing receipt or completion state is relabeled.

The review independently checks four synthetic witnesses (including both equal
1,376-ns fill durations with distinct host IDs), the nine actual CPU records and
the cold CLI assessment's source/input hashes. Non-correlation backend/phase/
required-kernel guard ASTs match the frozen v2 verifier, and the existing
thread/sequence/producer-chain function and family table are reused unchanged.

Versioned PyTorch 2.8 sources support the raw APIs and linked-ID ownership.
[Association implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/profiler.py#L558-L633),
[raw metadata bindings](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/init.cpp#L227-L279).
These internal APIs require future target validation; source inspection and CPU
metadata tests are not CUDA qualification. Structurally valid fabricated metadata
still requires hash-bound provenance. Metadata proofs cannot infer actual CUDA
execution admission, worker completion, resource checks, fresh replay, full-model
adoption, speed/cost or research admission.

Next integrate the new collector/verifier into a separate component worker and
test complete startup, numerical, profiling, storage and receipt paths before
preparing a new university scope. No SSH/upload/proof, target retry/retrieval/
cleanup/fallback or paid work occurred. The rejected candidate, full-training
gates, financial caps/holds and pending reconciliation remain unchanged.

The accompanying JSON contains exact source/artifact bindings and the local input
seal. No files were staged, committed or pushed.
