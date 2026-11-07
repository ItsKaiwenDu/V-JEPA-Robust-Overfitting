# Controlled common-cotangent handoff: prepared, unarmed

This separate implementation follows the [reviewed offline plan](predictor_common_cotangent_v1.md).
It preserves every captured source, result and consumed claim. Its final handoff
is `runs/pretrained/diagnostic/common-cotangent-handoff-v1-20261106/linux10603-final`.
Control SHA256:
`d02e3a9ef8fb6e5426b504b5710066c9755960dcdabee839d81eaf3ff28c5821`.
There is no AUTHORIZATION.json or ATTEMPT.json. An unarmed helper refuses before
creating a claim, session, SSH connection, upload, proof or model calculation.
No prior university grant authorizes this new work.

## Four fresh serial cells

| Index / cell | Path | Predictor-output cotangent | Scientific status |
| --- | --- | --- | --- |
| 0 / 00 | Original | Native original c0 | Must reproduce passing native loss/input values exactly |
| 1 / 11 | Predictor-math | Native candidate c1 | Must reproduce the known native rejection exactly and remain failed |
| 2 / 01 | Original | Replayed c1 | Controlled VJP; not a native-loss gradient qualification |
| 3 / 10 | Predictor-math | Replayed c0 | Controlled VJP; not a native-loss gradient qualification |

All four workers retain the original training clip, fixed clean targets,
weights, raw pixels, all eight draws and two mask families, native forward L1
objective, activation checkpointing, shared FP64 context projection/native
FP32 backward, runtime, masks and seeds. Each worker performs one backward on
its fresh graph. There is no training, optimizer, attack/scoring, held-out work,
new oracle or profiling/timing phase. The existing unsupported-candidate study
guard remains active. Backend-selection flags do not establish actual CUDA
kernels or resolve historical failures.

Both native diagonal loss/input returns must reproduce their own saved values
exactly before attribution can complete. Loss rtol 1e-5 / atol 2e-6 and input
rtol 0.002 / atol 2e-7 remain. The original is compared with the historical
reference and its saved anchor. The candidate is compared with the historical
reference and the actual new original files, which must match the passing
anchor. The candidate's expected input rejection is retained with those
unchanged bounds and a canonical failure receipt. No failed worker is loaded
as a passing-native reference.

The new attribution-only protocol explicitly permits the two cross cells only
after both native values have reproduced and their records verify, including
the candidate's known rejection. This permission must be separately approved;
it does not continue or modify the consumed observer run. Any unexpected error,
different diagonal, missing marker, changed previous file, ownership/resource
refusal or deadline failure stops the new plan and collects evidence without
retry. Both supervisor and child entry recheck every earlier cell before the
next worker's scientific imports/model references.

## Controlled boundary and record semantics

For every one of sixteen predictor calls, the module hook returns None and
checks the actual same-path prediction, mask order/operands and typed layout.
The tensor hook records the actual incoming native cotangent. It verifies the
entire saved same-path hash, checks every sparse source bit, patches a separate
copy, and verifies the entire saved destination hash. Exact actual received /
applied uint32 values at all eleven changed coordinates are retained alongside
whole-tensor signatures, original devices/shapes/dtypes/strides and callback
thread identities. All sixteen pairs are verified, including unchanged pairs.
Native incoming tensors are never modified by the intervention.

Diagonals return None from their tensor hooks after checking the native values.
Cross workers explicitly return the copied replacement cotangent on the
original device. They therefore measure controlled VJPs, not derivatives
qualifying the original native L1 objective. Observer and graph hooks must
finish/remove; the attention adapter, model tensor values/versions/modes,
precision settings, targets and absent parameter gradients must be preserved.
The frozen graph-ownership route is unchanged; no extra thread is whitelisted.

The distinct controlled-cell identity binds cell/path/cotangent, PID, host and
control hash. It has no ambiguous generic `passed` flag. Attribution completion
uses `attribution-complete.json`; unexpected failure uses
`attribution-failure.json`. Cell 11 additionally requires a hash-bound
`native-numerical-rejection.json` when rejected. Native numerical status stays
false for the target candidate; it is null for the cross cells. Original native
loaders reject these records. The supervisor's scientific/study/performance
qualification flags stay false even when all four requested captures complete.
An exit of zero means the attribution data completed, not that the candidate
passed a numerical gate or entered the study.

Each cell saves actual FP32 scalar loss, input gradient and clean targets before
numerical gates, plus mask indices and bounded observation metadata. Full
predictions/cotangents are hashed in memory instead of written again. Existing
full saved tensors and the exact sparse metadata are the value anchors.
External call/scope records, the small header and final markers bind every
observation and native GraphTask. Rejection evidence remains separate and
mandatory; missing final publication refuses even a success-looking header.

Native initial conditions and prediction/cotangent hashes are strict replay
controls, not a proof of bitwise internal Q/K/V state across alternative paths.
Once actual target cross cells exist, independently analyze both ordered
decompositions and their interaction. Common external cotangents still leave
the Jacobian dependent on internal forward rounding; this cannot alone prove
a pure attention-backward arithmetic bug. Target causal contributions remain
unknown until those actual saved gradients are reviewed.

## Exact proposed payload

One fresh private `/var/tmp` directory on `kdu@linux10603.dc.engr.scu.edu`:

- Unchanged original packet: 17,192,095 bytes / 99 members.
- Unchanged value-only archive: 20,037,087 bytes / five members,
  25,182,004 uncompressed bytes. Four prior loss/input-gradient returns and
  provenance remain explicitly distinguished as passing-original and failed-
  candidate values, never executable passing-worker references.
- Exact sparse-value metadata: 12,314 bytes, containing sixteen pairs of
  full prediction/cotangent hashes and eleven coordinate/bit-pattern changes.
- Twenty-one verified overlays totaling 298,508 bytes, plus bound control/
  proposal metadata containing the existing budget snapshot.
- Stdin bootstrap: 890,104 bytes, below the 2 MiB limit.

Checkpoint weights and installed dependencies are verified in place. No private
keys, passwords, held-out pixels, weight/dependency uploads or installations are
included. Cold bootstrap verifies scope and a separate authorization before
scientific imports or target writes. The client binds its exact local directory,
public host key, all current sources, old consumed files, both archives, sparse
metadata and 10,681 sealed predecessors. Copying a helper cannot reset a claim.
The unchanged archive is received only after the full storage proof.

## Limits and operation

One attended attempt permits up to 40 minutes of work from before live preflight,
including setup at most three minutes and the full 2.5 GiB allocation/write/
fsync/readback proof at most 60 seconds, then four fresh serial workers each at
most seven minutes. Static allowance 180+4*420+120=1,980 seconds fits inside
2,400, with a continuing receipt reserve. Collection follows on the same SSH
connection even after a refusal, for at most ten additional minutes / 3 GiB.
Transport including login is capped at 50 minutes; enter the password promptly.

Setup/capture allocated ceilings remain 3 GiB / 2.25 GiB. Categories remain raw
512 MiB, metadata 64 MiB, input 128 MiB, code 128 MiB, cache 32 MiB and emergency
16 MiB. Torch allocator stays 7 GiB. Actual typed partial caps are 16 MiB per
tensor, 8 MiB calls, 64 KiB scopes, 256 KiB header and 16 KiB per marker. The
new marker reader/writer preserves regular-file/ownership/device/size checks,
NOFOLLOW opening, exclusive partials, fsync, atomic publication and old-reader
isolation. The global frozen format and limits are unchanged.

Actual logical and allocated totals, directory identity, free-space/inode floors,
continuing clocks, GPU/runtime identity and foreign-compute guards remain.
Expected GPU: RTX 3070, UUID `GPU-e54fae90-feeb-8106-d8c0-1974fe48555f`,
8,192 MiB, driver 580.178.04; Python 3.11.17 / Torch 2.8.0+cu128 /
CUDA 12.8 / NumPy 2.4.6. A fresh status and live checks must pass before work.
Leave other processes alone; sampling and projections reserve no capacity.

The Mac must stay awake, lid open and attended Terminal open through collection.
The helper claims once before SSH. Never rerun it or use standalone retrieval
after a result. Keep partial output and report any error for independent review.
No automatic retry, old claim/clock reset, cleanup, fallback/parallel host or
paid compute is included. Financial caps, holds and pending reconciliation
remain unchanged; known Runpod Pods remain terminated.

## Preparation verification and next action

68 distinct focused local checks pass, zero failures/errors/skips. Across two
cohorts, eight fresh tiny CPU workers run the four cell identities and preserve
native loss/input returns exactly, including all 128 actual observed boundaries,
eight complete graph scopes and restored hooks/model state. The CPU original
and candidate cotangents happen to match; these zero-edit full-worker cells do
not stand in for the target's eleven-entry intervention. A separate actual
nontrivial toy backward check exercises the new observer's two cross directions
against an explicit Jacobian. CPU proofs establish no pretrained CUDA result.

Two deliberate local numerical-error injections retain complete attribution
data and native-rejection markers; they are explicitly simulated refusals, not
newly observed CPU numerical discrepancies. Two real primary-backward-error
paths keep the actual loss, partial metadata and removed hooks. Old-native
loaders, relabelled cross cells, missing/rehashed altered files, typed partials,
marker symlinks/hardlinks/oversize files, stale scopes and expanded/old grants
refuse. Cold bootstrap forbids scientific imports before authorization.

Four same-connection simulations exercise full success, retained canonical
native rejection with the newly scoped cross continuation, unexpected cell
failure and prior-control tampering. The latter stops before any cross work.
Mock GPU/proof values and test authorizations are fixtures only; they are not
resource grants, storage proofs, target qualifications or reservations.

The first run records 28 passes, one stale marker-filename test failure and
24 fixture errors because the new final marker was read through the frozen
reader's older filename allowlist. The separate new bounded marker reader and
test filename were corrected, with fifteen added marker-integrity checks.
The second run passes all 68 checks. Earlier source versions and receipts are
retained; counts are not added. The frozen old reader and limits were not changed.

Next obtain one new exact scoped approval and fresh linux10603 status after
independent handoff review. The former two-worker observer approval is consumed
and covers neither this four-worker attempt nor explicit cotangent replacement.
After approval, reverify bindings, save a separate authorization, review its
armed-but-unused state, then give the attended local Mac helper once. Do not
ask an already approved unchanged scope again. No target command is supplied
while this handoff is unarmed.
