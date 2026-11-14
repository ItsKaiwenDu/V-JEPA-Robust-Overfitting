# Raw profiler retention before semantic validation

This offline change separates bounded raw extraction from the frozen acceptance
rules. The consumed university run rejected a raw field before returning the
trace, so the offending record was never written. Its exact field remains unknown;
no new trace or correlation IDs are reconstructed from that failed run.

## Separate extraction and publication

`predictor_sdpa_raw_capture_v1` reads only an already completed profiler, using the
same raw API method mapping and CPU/CUDA device names as the frozen collector.
It performs no Torch import, profiling, tensor, model, derivative, device, process
or network operation. It preserves serializable field values without repairing
negative integers, types, clocks, IDs, flags, legacy durations or duplicate rows.
Semantic acceptance remains exclusively in the unchanged raw-correlation verifier.

The extractor bounds the complete event sequence and encoded envelope using the
existing event-count and 8 MiB limits, without truncation. Scalar transport permits
finite JSON values, strings up to 16,384 characters and integers up to 256 bits.
These retention bounds include every semantically valid frozen value, including
the original 4,096-character names and 63-bit clocks/IDs. They do not replace those
strict semantic bounds. Nonfinite values, unsupported object types, excessive
scalars, missing APIs and unknown device types refuse with bounded record/field
context. Those extraction failures cannot produce a complete raw envelope; the
error preserves context rather than substituting a fabricated JSON value.

The separate v5 worker retains the original arithmetic, inputs, layout/alias checks,
reference, numerical bounds, runtime/precision, startup ownership, flags, resource
limits and replay logic. The new plan explicitly requires publication before
semantic validation. After the profiled return arrays are saved, it extracts the
bounded envelope, writes `events.jsonl` through the unchanged private atomic store,
saves its hash/size/count/runtime descriptor and publication marker in the worker
header, then reads the exact stored bytes and invokes the frozen `project` guard.
The original backend and completion checks still follow. A rejected trace remains
saved with a failure marker; it never becomes a completed worker or resource receipt.

The existing 8 MiB trace slot, 256 KiB worker header, array limits and 7 GiB GPU
allocator remain. The v5 target wrapper differs only in versioned source/resource
identities; its validation bodies and private publication behavior are unchanged.
The executable rehearsal remains local CPU only and refuses university hosts/CUDA.
No university operator, target control, authorization, attempt or new remote work
is prepared at this checkpoint.

## Verification

79 focused tests pass on the first run, with zero failures, errors or skips. New
metadata cases retain invalid serializable raw scalars exactly, then confirm the
frozen validator still rejects them. Nonportable or excessive values identify the
record and field. Byte/event ceilings, missing APIs, cold import without science,
completion forgery, resource receipts and private publication are checked.
AST comparisons normalize only version identities, publication-contract fields,
extractor import/call and the saved-before-validation checkpoint; all remaining
core functions and target/CLI validation bodies match the frozen implementations.

Four actual fresh small CPU workers complete original-0/math-0/original-1/math-1
with one canonical FP64 reference. Original numerical gates, native values,
backend proof, flags/input checks and exact profiled/fresh replay pass. Raw counts
remain 75/249/75/249. Three further actual small CPU failure workers retain their
available numerical evidence: an injected association refusal, an injected missing
API, and an injected negative device-index field. That negative field is synthetic;
it is not the unknown offending value from the university run.

The invalid-field worker verifies its saved bytes and worker descriptor before
calling the frozen guard, which then rejects the metadata. Both return sets and
the raw record remain available, while completion/profile/final-check flags and
GPU resource receipt remain absent or false. A valid metadata extraction followed
by a later backend refusal also retains the trace. An injected API failure retains
available numerical arrays and reference but cannot invent missing raw metadata.

Independent saved-only review verifies all 61,172 prior hashes, 56 unique actual
return arrays, 16 FP64 reference arrays, raw profiles, collection copies, private
metadata and failure artifacts. New model/autograd/SDPA/CUDA/profiler/process/network/
reference execution is blocked during review. Four canonical reference builds and
seven fresh CPU workers belong to separate explicitly local test cases. Target
startup/resource fixtures are simulated; actual local science uses PyTorch 2.14.1
CPU. Target PyTorch 2.8/CUDA behavior and speed/capacity remain unqualified.

The real failed university claim, missing trace, incomplete flags, numerical evidence
and clocks remain unchanged. No tolerance, frozen validator or old source is
modified, and no old suite is rerun. Candidate rejection, full-training gates,
financial caps/holds and pending reconciliation stay unchanged. No new paid work.

Next prepare a separate university handoff using the accepted native capsule and
this verified worker. Check exact payload, cold authorization/imports, guards,
serial clocks and same-SSH collection before any new scoped university approval.
Never reuse a consumed helper, grant, claim, retrieval connection or time allowance.
