# Exact-path native attention mask-file policy

The [saved metadata review](../report/checks/2026-11-07-sdpa-metadata-result-review.md)
identified the actual interruption: a 146,562-byte mask-index file exceeded the
original 65,536-byte guard. That guard runs during capture and collection.
The old native attempt and separate metadata read are both consumed.

`models/predictor_sdpa_storage_v2.py` is a separate local storage/collection
component. It changes no frozen v1 function or global and grants no target work.

## One typed-file allowance changes

Only these two exact relative paths receive a 262,144-byte (256 KiB) allowance:

- `results/capture/mask-indices.json`
- `results/capture/mask-indices.json.partial`

Aliases, traversal and backslashes refuse. Same-basename files elsewhere keep
their old limits. Attention scopes remain 64 KiB, worker records 256 KiB,
boundary headers 128 KiB, event streams 8 MiB and tensors 16 MiB. Aggregate
metadata remains 64 MiB, raw 512 MiB, allocated capture/setup 2.25/3 GiB, cache
32 MiB and collection 3 GiB. Proof/resource/free-space/inode and filesystem
checks retain their original behavior. This is an individual-file allowance;
it does not validate mask contents or waive the frozen mask-signature checks.

The original eight draws and two mask families contain 32,768 indices. The pinned
CPU sampler reproduces all original context/target signatures and the exact
146,562-byte serialized extent observed in the incident. A worst-case fixture
using those shapes and maximum index 2047 is 164,322 compact JSON bytes, below
256 KiB. It does not require a larger aggregate metadata or tensor allowance.

## Local verification

43 focused checks pass, with zero failures, errors or skips. They exercise the
actual full pretrained mask geometry and frozen compact JSON writer, old refusal
and new acceptance, partial writes and transient atomic hardlink pairs, exact
limit/one-byte-over refusal, unrelated paths, existing category and allocated
ceilings, symlink/hardlink restrictions and collection source-change refusal.
Every entry in the 144-file incident inventory is compared against the frozen
classifier. Only the mask file gets a different limit. For already accepted
inputs, storage accounting and the collection body/header fields match v1;
only the explicit v2 collection kind differs. Collection preserves bytes,
hashes, private modes and nanosecond timestamps in its manifest.

The tests generate CPU masks only. They load no model or weights and start no
CUDA, SSH or university work. Saved review independently reconstructs all 32
int64 signatures with `struct` and `hashlib`, checks every mask partition, and
reverifies 21,507 preserved bindings. Scientific imports and process/network
execution are blocked during that review. Frozen binary evidence is read only
for integrity hashes; tensor arrays are not deserialized.

The first suite had 41 passes. Two added compatibility regressions give 43;
counts are not added together. The first saved-file reviewer stopped because
pytest exposes a real fixture directory and a `current` symlink. Its source and
failure record remain. Only the review path selection changed; the second
saved review passes without repeating tests or scientific work.

## Integration remains separate

This component has no target entry point, authorization, claim or attended
helper. It uses a distinct v2 collection envelope so it cannot masquerade as a
completed v1 handoff. No existing helper is patched to use it.

Next prepare and locally verify a separately bound replacement operator, client
and handoff. They must use this inventory for both live storage checks and
collection, preserve all original numerical/byte-replay/ownership/backend gates,
and bind the consumed incident claims. Cold authorization, real payload import
paths and same-connection collection need integration checks before requesting
one new exact university scope. Old grants authorize no retry or clock reset.
No new approval is requested for an unfinished handoff.

The previous candidate rejection at 3,724 entries and all financial caps,
encumbrances and pending reconciliation remain unchanged. This file makes no
new numerical, CUDA/backend, performance or research claim.
