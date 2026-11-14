# Raw metadata retention v1 / component worker v5: local review

The separate extractor and v5 worker save bounded raw profiler metadata before
semantic validation. The frozen raw-correlation validator, original arithmetic,
reference, numerical bounds, runtime/precision, startup owner, flags, input layout/
alias checks and replay gates remain unchanged. No rejected metadata is accepted
or repaired. The old real university trace remains missing and its exact offending
row/field remains unknown; synthetic negative metadata is not that lost record.

The extractor reads only a completed profiler through the same frozen raw APIs.
It performs no Torch import, profiler start, tensor, derivative, device, process
or network operation. It retains serializable values exactly, bounds complete
count/encoded bytes to the existing limits, and refuses rather than truncating.
Nonfinite, unsupported or excessive scalar values and unavailable APIs report
bounded record/field context; these failures cannot yield a complete envelope.
Retention transport bounds do not replace the stricter frozen semantic bounds.

The v5 plan explicitly requires publication before semantic validation. The worker
writes the raw envelope to the existing private atomic 8 MiB trace slot, checkpoints
its hash/size/count/runtime and publication marker in the worker header, reads the
saved bytes back, then applies the unchanged `project` guard. Later backend,
profile/replay, input/flags and completion checks stay. Failure retains available
arrays, reference and raw bytes with an incomplete marker; no final GPU resource
receipt is published. The target wrapper and local CLI bodies match their frozen
versions after version-identity normalization; private receipt rules stay strict.

79 focused tests pass on the first run with zero failures, errors or skips. Four
fresh actual small CPU workers share one canonical FP64 reference and pass original
numerical/native-value, backend, flags/input and exact profile/fresh replay checks.
Raw trace counts stay 75/249/75/249. Three further actual small CPU workers exercise
injected association, missing-API and invalid-field failures. The invalid-field
worker proves its raw bytes and descriptor are published before the unchanged guard
rejects the injected negative device index. Its later completion flags remain false.
The API failure retains numerical evidence but cannot invent unavailable raw data.

Fourteen handwritten invalid serializable metadata cases retain exact values and
still refuse under the frozen validator. Nonportable/excessive-value diagnostics,
missing APIs, complete byte/event ceilings, cold import, forged completion/trace
claims and resource metadata are checked. Fourteen malformed resource fixtures
still refuse. CUDA startup/resource boundaries use simulations and do not qualify
actual GPU allocations or execution. All original core function ASTs match after
normalizing only the new retention instrumentation/contract; no science changes.

Independent saved-only review verifies 56 unique actual FP32 return arrays, 16
FP64 reference arrays, four separate canonical references, seven fresh CPU workers,
raw proofs, failure artifacts and complete local collection copies. Byte hashes,
private modes and nanosecond timestamps match. The collection exercise is explicitly
local, not a new university grant. The review blocks model, autograd, SDPA, CUDA,
profiling, process, network, reference and extraction execution. No tests or old
suites repeat during review. All 61,172 prior hashes and 189 previous target file
metadata descriptors reverify; old claims, incomplete flags, missing traces and
clocks remain unchanged. New source/test artifacts also match their pre-test snapshot.

All new sources, docs, tests and saved artifacts are frozen by the accompanying
input seal. Actual scientific execution here is local PyTorch 2.14.1 CPU. Target
PyTorch 2.8/CUDA metadata behavior, backend/resource compatibility and speed/capacity
remain pending. There is no new full-model capture, university operator/control/
authorization/attempt, connection, upload, proof, cleanup, retrieval, installation,
held-out data access or paid compute. Candidate rejection, full-training gates,
financial caps/holds and pending reconciliation remain unchanged.

Next prepare a separate university handoff using the unchanged accepted native
capsule and verified v5 worker. Bind its explicit publication contract and v5
resource identity and check the full cold authorization, exact payload/imports,
GPU guard, serial clock, storage and same-SSH collection path before one new exact
scoped approval. Never reuse a consumed helper, claim, grant or clock.

[Retention design and limits](../../docs/predictor_sdpa_retention_v1.md).
