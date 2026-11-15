# Bounded attention component route prototype

The local component route now expresses one original attention forward and one
first-order VJP, with an optional owned inner public-math VJP for the candidate.
It reuses frozen arithmetic and has a new component identity. The old full-model
route, standalone refusals, closed cohorts, thresholds and backend policies are
unchanged. This is an **unbound source prototype**, not a worker or launcher.

## Frozen arithmetic and compilation

`models/predictor_pgd_split_component_v1.py` checks exact SHA-256 values before
copying the frozen CPU public-math `backward`, target `_Attention` class and
device-aware fingerprint observer. It retains the original forward, math backward
and backward-argument bridge bodies. No arithmetic, saved input or arriving
cotangent is replaced. The unused CPU class has no forward entry in the prototype.

The first bytecode comparison failed despite equal arithmetic ASTs. CPython's
module-import symbol context changed `LOAD_ATTR`/`LOAD_METHOD` call instructions
and exception offsets in the extracted source. The correction copies the three
relevant frozen import declarations into a literal `if False` branch. Those
compiler declarations execute no imports, and all generated code objects contain
zero import opcodes. Source AST and compiled function bodies now match the full
frozen sources on the same local Python interpreter, excluding file/line/position
metadata only. No numerical gate was relaxed. The original failure, diagnostic
and both source snapshots remain saved. A future target worker must check its own
exact runtime/source/code binding; this is not a target compilation observation.

The builder itself imports only the standard library. A future reviewed worker
must supply the runtime namespace; current tests supply explicit fake objects.
There is no scientific import, installation, fallback, tensor construction or
execution entry in the builder. The false-branch import declarations are compiler
context only, not authorization to import or run scientific code now.

## One-call lifecycle and ownership

The proposed domain is CUDA device 0, FP32 shape `[1,12,2048,32]`, three distinct
leaf objects with positive strides and storage extent at most 16 MiB each.
Original strides, offsets, alias groups, bytes, versions, input-gradient absence
and flags remain checked. Incoming cotangent bytes/layout/version/hooks and exact
output/root tokens are checked across public boundaries. The route observes
original CPU/CUDA RNG state without changing it.

One namespace claim is consumed before work. A scope admits one creator-thread
forward, then one exact-token `autograd.grad` call with `create_graph=False` and
`retain_graph=False`. Public and private execution paths are checked. Root
prehooks must report one real nonnegative outer GraphTask, and all three input
leaf hooks must belong to that task. The candidate's copied backward additionally
requires one distinct actual inner math GraphTask, one root, all three leaves,
exclusive public math selection and restoration. Callback threads may differ;
public entry and exit belong to the creator process/thread. No extra objective,
warmup, replay or retry is introduced.

All outer handle removals are attempted even if an earlier removal fails. The
inner cleanup loop remains the exact frozen arithmetic source: an injected inner
removal failure can leave later inner handles live. Such a route stays incomplete;
a future isolated worker must fail and exit, rather than publish success or retry.
Success also requires restored flags/modes/hooks/versions/gradient absence/RNG
and one-use closure. Live route input/output references are released on closure.

The bounded snapshot is at most 64 KiB and refuses oversize metadata without
truncation. External design/control/boundary/probe fields are checked for role and
syntax, but their canonical lineage is not authenticated by this route. Even a
complete route snapshot leaves backend, CUDA execution, worker, scientific,
lineage and training admission false. Source-only or simulated receipts cannot
qualify real execution, geometry, numerical accuracy or the interrupted old case.

## Local checks and remaining integration

72 distinct cold/source/fake-engine checks are now covered. The first run was
65 pass / 1 bytecode-context failure. After the implementation correction, all
72 checks passed, including six new import/restoration/metadata-ceiling checks.
65 previously passing checks were repeated because the generated instructions
and boundary guards changed. There are 138 test executions, not 138 distinct
checks. No old test suite, scientific cohort, model, gradient, profiler, GPU,
SSH, upload, storage proof, retrieval, cleanup or paid work was run.

Fake tensors contain tiny opaque byte labels plus shape/layout metadata; they do
not contain mathematical arrays. Fake callbacks use graph IDs 101/202 and saved
receipts explicitly say `FAKE-ENGINE-SIMULATION`. These checks establish source
and control-flow behavior, not any PyTorch/CUDA numerical or scheduling result.

Next integrate this route and the hybrid raw metadata reader into the new bounded
worker, with authenticated canonical reconstruction, original numerical/native
proofs, durable progress, inventory, parent wall/RSS supervision, profiler stop
boundaries, receipt verification and same-session collection. The proposed
120-second / 4 GiB RSS / 7 GiB allocator worker limits are **not implemented by
this source builder**. Cold/AST/simulated work remains the only current scope.
No university helper or authorization proposal is ready, and full training
qualification remains open. All financial caps, holds and pending reconciliation
stay unchanged.
