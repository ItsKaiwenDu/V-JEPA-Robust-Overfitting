# Common-cotangent attribution: saved analysis and local CPU prototype

The [reviewed native capture](../report/checks/2026-11-06-predictor-boundary-v1-target-review.md)
contains two complete sets of sixteen output/cotangent pairs. Both observed
loss/input returns exactly reproduce their own prior values. The original
passes; the candidate remains rejected at 3,724 input-gradient entries. Eleven
actual output cotangents differ in nine pairs, despite identical scalar losses.

This checkpoint uses those saved tensors and a **local tiny CPU prototype**.
It creates no university operator, helper, packet, authorization or attempt.
The latest claim remains consumed. Both university hosts may idle. The new
[inert proposal](../configs/predictor_common_cotangent_v1.json) grants no work.

## What can be proved offline

All 10,665 sealed target inputs are reverified. The new 12,314-byte value bundle
binds full shape and FP32 logical hashes for each prediction and native
cotangent, exact masks/targets, and eleven changed coordinates encoded as
two exact uint32 bit patterns each. It contains no Jacobian or executable
model. Every actual saved cotangent is independently checked against the
original native L1 tail. For every one of the sixteen pairs, both diagonal and
both cross patches are compared with the entire saved destination array:
**64 whole-array comparisons are bitwise exact**. Unlisted entries remain
unchanged; changing an incoming entry or omitting a required edit refuses.

The compact representation does not remove provenance checks. Before a future
intervention, the full incoming cotangent must match the saved same-path hash,
every changed source bit must match, and the entire patched tensor must match
the saved destination hash. Every same-path prediction and original context /
target mask identity must also match. All sixteen pairs are checked, including
the seven with no changed cotangent. No native incoming tensor may be mutated.

Existing saved gradients are g00=J0^T c0 and g11=J1^T c1. The actual cross cells
g01=J0^T c1 and g10=J1^T c0 are absent. Sparse cotangent differences do not
provide either Jacobian. Entry counts, equal scalar losses and small cotangent
norms cannot determine the missing VJPs or a numerical causal percentage.

## Tested local intervention

The new `LocalCotangentReplay` refuses university hosts, CUDA, target-sized
data, expanded grants, changed masks/targets/predictions and occupied module
hooks. It supports only the existing 32-token / 128-feature tiny CPU domain.
A module hook returns None and leaves the original forward unchanged. A tensor
hook checks the actual incoming native cotangent, makes an independent copy,
applies the exact sparse value edits, and returns that copy for a cross cell.
For a diagonal it returns None after checking the original values. Returning a
different cotangent is an explicit diagnostic intervention; the resulting
gradient must never be labelled a native L1-loss gradient.

All handles are single use and removed on normal completion, incomplete work
and primary exceptions. The prototype does not change autograd scheduling or
the frozen native GraphTask ownership route. It is not a tested live operator,
resource supervisor or CUDA backend qualification.

39 focused checks pass, zero failures/errors/skips. A deliberately synthetic
two-path CPU model runs four actual fresh graphs. Its four VJP cells match
an explicit Jacobian calculation exactly, preserve both native diagonals and
reconstruct both ordered decompositions with a nonzero interaction term.
Synthetic data are labelled and cannot qualify target behavior.

On the real small upstream CPU model, native before / diagonal / after input
gradients and losses are bitwise equal, including the frozen earlier CPU
baseline. A deliberately hypothetical other cotangent changes one entry and
produces a different input gradient. Its controlled native-hook result is
bitwise identical to an independent direct multi-output VJP in this local
execution. Model values/versions, absent parameter gradients, module hooks
and native GraphTask hooks are preserved. This one-entry hypothetical is not
the actual eleven-entry target intervention. These CPU checks do not establish
pretrained CUDA correctness, performance or memory capacity.

## Minimal future target comparison, not authorized

Prepare a separate source-bound attribution record and bounded operator before
requesting one new exact scoped university approval. Do not mutate or rearm the
consumed boundary helper. Proposed cells run in four fresh serial processes:

| Cell | Forward/propagation path | Incoming predictor cotangent | Meaning |
| --- | --- | --- | --- |
| 00 | Original | Native original c0 | Exact replay of the passing native values |
| 11 | Predictor-math | Native candidate c1 | Exact reproduction of the known rejection |
| 01 | Original | Replayed c1 | Fixed original path, changed external cotangent |
| 10 | Predictor-math | Replayed c0 | Fixed original cotangent, changed propagation path |

The first two cells must reproduce the same-path saved prediction hashes,
all actual native cotangents, clean target/mask/weight/pixel/runtime identities,
and saved loss/input returns exactly. Original numerical gates remain: loss
rtol 1e-5 / atol 2e-6, input rtol 0.002 / atol 2e-7. Cell 11's expected canonical
input-gradient rejection must remain failed and unusable as a passing worker
reference. A new attribution-only protocol may proceed to the cross cells
only after exact value reproduction and failure/restoration verification,
under its own explicit scope. This is not continuation of the old serial plan.
Unexpected failures, changed values, incomplete records or resource refusals
stop the new plan without retry. Fresh diagonal repetitions are necessary to
validate the new code path; historical passing receipts cannot stand in for them.

Cross workers retain the native forward objective, graph and same-path
predictions. Their backward hook replaces only the specified cotangent bytes.
Retain actual received/applied whole-tensor hashes and changed entry values,
complete ownership/scopes, hook removal and all original state checks. Their
input gradients have a distinct controlled-VJP identity. They do not pass a
native-loss gradient test, qualify attacks, or repair the failed candidate.
Do not route them through a passing-native-worker loader or adopt the
candidate. Backend-selection flags still do not prove actual CUDA kernels.

Once actual saved cross cells exist, report both decompositions:

`g11-g00 = (g10-g00) + (g11-g10)`

`g11-g00 = (g01-g00) + (g11-g01)`

Retain the interaction `g11-g10-g01+g00` and reconstruction roundoff, all
elements and original failure locations. Do not choose one order after seeing
the answer or silently assign interaction to a single cause. Fixed external
cotangents still leave the Jacobian dependent on internal forward rounding;
this comparison cannot by itself prove an isolated attention-backward bug.
That would need a later same-Q/K/V and same-cotangent experiment.

## Proposed bounds and next work

One new private `/var/tmp` attempt on linux10603 is proposed, with 40 minutes
of work including three minutes of setup, the full 2.5 GiB write/fsync/readback
proof (at most 60 seconds), four workers at most seven minutes each and two
minutes of receipt reserve. Static allowance is 180+4*420+120=1,980 seconds
inside 2,400. Collection is on the same SSH connection for at most ten additional
minutes / 3 GiB. Setup/capture ceilings remain 3 GiB / 2.25 GiB; allocator
7 GiB, raw 512 MiB, metadata 64 MiB, input/code 128 MiB each. Actual typed-file,
category, allocated/free-space/inode, deadline and foreign-process guards must
refuse excess. Leave other users' processes alone.

The proposed payload uses the unchanged original packet, value-only loss/input
anchors and this compact sparse metadata, not two full cotangent arrays.
The live implementation must recheck archive bytes and source-bound metadata.
Predictions/full cotangents would be hashed in memory instead of all being
written again. Four loss/input/target sets have 83,886,096 logical FP32 bytes;
twelve 64 KiB header allowances plus one 16 MiB atomic copy give a projected
101,449,744 bytes below 512 MiB. This is a storage projection, not a demonstrated
live peak or a reservation. It does not waive any existing bound.

No retry, old-claim/time reset, cleanup, standalone retrieval, parallel target
host, held-out pixels, private-key upload, dependency/weight installation or
paid compute is proposed. Financial caps/holds and pending reconciliation
remain unchanged. No fresh host check is needed for this offline checkpoint.

Next implement/test/review the separate live controlled-VJP record, worker and
attended handoff. Only after that concrete result should one new exact scoped
approval and fresh host check be requested. There is no target command yet.
