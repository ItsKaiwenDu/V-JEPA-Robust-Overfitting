# Component-only input capsule

The remaining component checks now have a compact, verified local input capsule.
It contains unchanged saved attention derivatives and provenance metadata from the
accepted eight-objective PGD run. Preparing it performs no tensor deserialization,
array comparison, model work, gradient calculation or university operation.

The archive is 27,279,360 bytes and contains 35 source files plus one generated
manifest. SHA-256:
`82a30efedfb0e85fd39c634c5bd0a8b82bd839b91bc7259841e2d517d495011e`.
It is saved at
`runs/pretrained/diagnostic/pgd-component-capsule-v1-20261119/component-inputs.tar`.
The raw archive and fixture evidence stay local under the ignored diagnostic tree.
The source builder, fixed input contract, tests and review are versioned.

## Input selection and provenance

Eight unchanged serialized files come from the original objective-zero boundary:
Q, K, V, the arriving cotangent, output, and dQ/dK/dV. They belong to the selected
`predictor.0`, draw 0, family 0, forward row 24, `EOT8-before` call. No return from
the failed native component is substituted. The boundary's native reference
returns are included for later review; this step performs no new accuracy check.

The capsule also carries the boundary completion/call/root metadata, original
objective-zero native proof, all eight objective headers and completion records,
numerical run and supervisor metadata, the historical failed session, and the
pinned accepted numerical review. Each member's bytes, length, repository origin
and SHA-256 are fixed in `configs/predictor_pgd_component_capsule_v1.json`.

All eight numerical objectives and four same-input comparison receipts remain
linked to the old control, source process and baseline. Original numerical bounds,
restoration flags, 37-artifact descriptors and non-input hashes remain joined.
The complete numerical arrays and all candidate hook logs are preserved in the
original local evidence tree; they are not copied into this compact capsule. The
capsule relies on their pinned accepted review and cannot replace the complete
numerical audit. Metrics are read from saved receipts, not recomputed.

## Layout and reference contract

The frozen metadata planner retains shape `[1,12,2048,32]`, FP32, logical hashes,
strides, offsets, backing extents, alias groups and input gradient flags. In
particular, V retains stride `[2359296,32,1152,1]`, offset 768 elements and backing
extent 9,437,184 bytes. A future worker must retain the existing reconstruction,
hook/token, RNG, version and restoration gates. The existing plan fills uncaptured
padding with zero and does not claim original physical alignment or addresses.

Serialized `.pt` files remain opaque bytes in this task. Neither Torch nor NumPy
is imported. No full pixel array, checkpoint, private key or old component output
is included. Metadata can still contain historical configuration and budget
information; this local preparation authorizes no upload of the capsule.

## Deterministic packaging and validation

`models/predictor_pgd_component_capsule_v1.py` pins the exact contract and frozen
operand metadata helper. It reads only canonical, bounded, single-link ordinary
files, checking held and named identities and hashes. The closed input inventory
refuses missing, additional, partial, altered or substituted files.

The USTAR archive uses fixed ordinary-file headers, mode 0600, zero UID/GID/time,
no extensions or links and deterministic order. The verifier reads bounded bytes
without extracting files and requires exact member hashes, regenerated manifest,
headers and padding. Duplicate members, traversal, absolute paths, links, changed
modes, oversized headers, extra trailers and truncated archives refuse. Limits
are 32 MiB per archive, 4 MiB per source member and 128 KiB for the manifest.
These are local packaging limits, not a new university resource grant.

33 distinct new local tests pass in 33 executions, with no failures or repeated
passing, historical or scientific cases. The separate standard-library audit v2
verifies 21,150 prior hashes / 21,880 prior fixture entries, lstat-only for the old
FIFO; all source copies; saved metadata joins; layouts; all archive members; and
an independent reconstruction of the full archive. Its first version mistook
arithmetic cap declarations for literal constants; v2 compares their exact ASTs.
Both versions/logs remain. No production change or test rerun followed that audit
correction, and no first-pass audit claim is made.

## Required worker interface

The capsule's historical `source_control_sha256` stays fixed. Its future execution
control and host are explicitly null. The old numerical PID/control must never be
renamed as a new run or treated as fresh component completion.

The next local integration should expose only the following route:

1. Verify this fixed capsule and its accepted numerical provenance as read-only
   source evidence. Preserve the completed eight-gradient cohort.
2. Bind a distinct, later-reviewed execution identity and private result namespace.
   Keep runtime/source, storage, process, ownership and continuing-clock guards.
3. Prepare one native component and one forward-preserving component, each with
   one forward and one outer VJP. The candidate contains its one original inner
   math VJP. Preserve failure-stop ordering and all original numerical gates.
4. Use the reviewed backend v3 / projection v3 metadata reader while retaining
   actual root/leaf ownership, source reconstruction and parent resource evidence.
5. Publish and collect only the new component evidence, linking it to the accepted
   numerical result without calling the old full numerical entry point.

This is an interface contract and byte capsule, not a runnable component worker,
supervisor, handoff or approval request. Source-cache/runtime binding, fresh
component numerical/backend evidence and collection integration remain. There is
no new target scope or authorization. The old native worker stays failed; full
EOT8, replay, performance, study-cost and training remain gated. All financial
caps, holds and pending reconciliation stay unchanged.
