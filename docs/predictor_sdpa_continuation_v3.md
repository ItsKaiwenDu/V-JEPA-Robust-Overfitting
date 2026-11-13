# Frozen-input component continuation v3

This separate university handoff uses the unchanged accepted native-input capsule
and the locally verified v4 component worker. It is prepared but unarmed: no new
authorization, attempt, connection or university work has occurred. The previous
consumed claim and incomplete component remain unchanged.

## Proposed scope

One attended attempt on `kdu@linux10603.dc.engr.scu.edu`, in a fresh private
`/var/tmp/vjepa-SDPA-components-v3-20261113-*` directory, using the existing
Python 3.11.17 / PyTorch 2.8.0+cu128 / CUDA 12.8 / NumPy 2.4.6 environment.
The expected RTX 3070 UUID and driver 580.178.04 must pass live checks. An unrelated
CUDA compute process, including a remote-desktop process, causes refusal; the
launcher neither stops that process nor transfers work to another host.

The work allowance is 15 minutes, including the full 2.5 GiB allocation/write/
fsync/readback proof, capsule receipt and verification, one canonical FP64
reference, and four fresh serial component workers: original-0, math-0,
original-1, math-1. Each worker has at most 120 seconds. Setup has 180 seconds
including the proof; the static reservation is 780 seconds within the 900-second
work ceiling, with 120 seconds reserved for receipts. No new full-model backward
or native capture is included. Four untimed and four profiled component iterations
are included; no warmup or throughput loop is permitted.

Collection is through the same SSH connection, with at most ten minutes and
3 GiB. The client allows at most 25 minutes including login, work and collection.
A failed worker stops later workers and collects available evidence within the
same limits. No retry, clock reset, separate retrieval connection, cleanup,
installation, private-key upload, held-out pixels or paid compute is included.

Setup/capture ceilings remain 3 GiB / 2.25 GiB. Input and code categories are
128 MiB each, raw captures 512 MiB, metadata 64 MiB, cache 32 MiB and emergency
space 16 MiB. Individual raw traces retain the 8 MiB slot, component metadata
256 KiB, and mask manifests 256 KiB. The GPU allocator ceiling remains 7 GiB.

## Frozen inputs and new evidence format

The capsule is unchanged: 36,042,589 bytes, 118 regular members and 48,627,833
logical bytes, SHA256
`b8ad88e9a34ea55460177d996d40f915efe9d4284e16e9616923a46eeca6391c`.
It contains accepted native observations and frozen source/metadata; it excludes
training pixels and the redundant prior derivative packet. No confirmation or test
pixels are included. The preparation report records the exact overlay count,
bytes, source hashes, bootstrap size and final control hash.

The frozen v4 worker writes explicit raw Kineto correlation metadata in a single
JSON envelope in `events.jsonl`. The plan and saved trace descriptor bind its
format, encoding, count and runtime. Exact recorded CPU/GPU links replace the old
ambiguous name/device/duration witness. Individual device correlation IDs are not
assumed globally unique. Missing raw APIs or invalid links refuse; IDs absent from
historical traces cannot be recovered. The previous real collision still refuses
under its frozen verifier and its worker remains incomplete.

Original arithmetic, operands, analytical reference, numerical bounds, runtime/
precision, flags, input layouts/aliases, strict configure-once startup and fresh
replay checks remain unchanged. Output bounds remain rtol 1e-4 / atol 2e-5;
dQ/dK/dV .002 / 2e-7; bookkeeping 1e-12 / 1e-14. The v4 worker and target wrapper
are byte-identical to their separately verified checkpoint.

The new handoff uses a standard-library metadata adapter when a fresh child reads
its predecessor's resource receipt before scientific imports. Its validator and
reader bodies are exact AST copies of the frozen startup validator and v4 reader;
only the reader's import of the startup module is removed. The same schema
constant and 7 GiB limit are copied explicitly. This avoids the startup module's
transitive PyTorch import while preserving every receipt check. Deep scientific
validation still uses the original v4 worker and resource reader. A synthetic
metadata receipt that passes the shallow check still fails the deep domain check.

## Local validation and limits

78 focused handoff checks pass with zero failures, errors or skips. They cover
cold bootstrap authorization, exact payload/import paths, full capsule identity,
GPU gate wiring, storage/serial deadlines, no old-grant or copied-claim reuse,
failure retention and complete same-SSH collection. GPU cases use mocked status
and synthetic resource metadata; no actual CUDA behavior is qualified locally.

Two actual fresh small CPU cohorts run four workers each, one through the packaged
capsule/overlay path. Each uses one canonical FP64 reference. The saved 64 unique
component returns and eight reference arrays, raw record counts 75/249/75/249 per
cohort, exact profile/fresh replay, private collection bytes/modes/ns timestamps
and frozen source bindings are reviewed independently without new scientific work.
Actual local scientific work uses PyTorch 2.14.1 CPU. Target PyTorch 2.8/CUDA API,
backend, numerical, capacity and speed qualification remain pending.

The first 68-pass/one-failure run exposed a transitive PyTorch import during
pre-import receipt validation. Its sources/logs remain. The second 69-pass/nine-
failure run exposed an omitted schema constant in the new metadata adapter.
Its sources/logs remain too. The exact frozen constant and an equality assertion
were added; no frozen scientific file or tolerance changed. Final 78 passes are
not added to the earlier runs. The old 45/54-test suites were not repeated.

The independent review checks all 47,042 preserved hashes, both final CPU cohorts,
collection and capsule contents, the unchanged consumed claim and old refusal.
Model, autograd, SDPA, CUDA, profiling, process, network and reference execution
are blocked during that review. Current files are sealed after it passes.

## Authorization and next action

Preparation creates an exact `CONTROL.json`, pinned public `known-hosts`, inactive
Mac helper and payload report. It creates no `AUTHORIZATION.json` or `ATTEMPT.json`.
The previous grant is consumed. After concrete preparation and review, one new
exact scoped university approval plus fresh linux10603 status is required. Save
that approval separately and review the armed, unused handoff before giving the
user the exact Mac command. The agent must not connect or execute that helper.

When later authorized, the user keeps the Mac awake, lid and Terminal open and
enters the university password promptly. Any result receives independent review;
never rerun a consumed helper. Candidate rejection, full-training gates, the $200
project / $160 compute / single $10 diagnostic caps, old holds and pending billing
reconciliation remain unchanged. No new paid allowance is reserved here.
