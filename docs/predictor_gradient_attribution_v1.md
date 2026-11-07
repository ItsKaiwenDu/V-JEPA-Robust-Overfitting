# Predictor gradient attribution: offline analysis and observer proposal

This checkpoint analyzes preserved values and tests a **local CPU prototype**.
It creates no university operator, packet, helper, authorization or attempt.
The preceding GPU attempt remains consumed. Both university hosts may idle.
The [proposal](../configs/predictor_gradient_attribution_v1.json) is not an
execution grant; a live implementation and independent review must come first.

## What the saved values establish

All 7,505 sealed predecessor inputs are reverified. The original full EOT8
baseline passes; the predictor-math candidate fails the original input rule
`abs(candidate-reference) <= 2e-7 + 0.002*abs(reference)`. All 3,145,728 elements
are included. The analysis partitions them into fixed magnitude strata and all
48 channel/frame groups; no selected subset replaces the full comparison.

The candidate has 3,724 violations across all 16 frames and all three channels
(channel totals 1,360 / 1,253 / 1,111). There are 58 opposite nonzero pixel-gradient
signs, of which three exceed the original bound and 55 lie within its absolute
allowance. These pixel-gradient signs are not predictor-residual signs and do
not demonstrate an L1 sign crossing. They also do not validate an attack.
Most violations are at reference magnitudes between 1e-5 and 1e-3: 2,063 and
1,186 in the two fixed bins. They are not confined to the smallest magnitudes.
The candidate remains rejected; aggregate error or sign counts cannot waive it.

Only scalar losses, input gradients, targets/mask identities and call records
were retained. Actual predictor outputs and their incoming cotangents were not
captured. Equal FP32 loss does not identify the derivative's source.

## Smallest useful new observation

The pinned native loss averages L1 over each prediction's tokens/features,
then over two mask families and eight draws. At batch size one with the outer
scalar sum, the mathematical prediction cotangent is

`sign(prediction - selected_clean_target) / (8 * 2 * target_tokens * features)`.

Exact-zero residuals have zero L1 derivative in this native implementation.
The CPU reconstruction retains the FP32 division order; a future capture must
save the **actual incoming cotangent**, not substitute that reconstruction.
A synthetic counterexample shows equal averaged FP32 L1 loss with opposite
cotangents. A second example shows identical predictions/cotangents with
different Jacobians. Neither synthetic example establishes the target cause.

Propose two fresh serial one-phase arms: original and predictor-math, each
using the original one-clip, all-eight-draw/two-family native objective. A
non-replacing forward hook observes the final predictor return for each of the
16 calls. A tensor hook copies the actual incoming cotangent and returns None.
Hooks neither change the prediction nor the gradient. Save original shapes,
dtypes, device/layout and logical hashes, and exact context/target mask identity.
Remove every observer hook on success and exception; retain failure evidence.
No surrogate loss, new oracle, profiling/timing pass or extra backward is part
of this first observation. Capture all predictions/cotangents and the actual
loss/input gradient before the existing numerical gate can reject them.

The original observed loss/input gradient must replay the passing saved original
exactly. The candidate's observed loss/gradient must reproduce the preserved
failed values exactly before attributing *that* discrepancy. This second check
is descriptive reproduction of a refusal, not acceptance of a failed worker as
a complete passing reference. All original elementwise gates still run. A
candidate rejection remains a rejection with a failure marker; later arms do
not start. Each arm's initial weights/pixels/targets/masks/seeds/runtime and
actual forward/backward call/scope coverage must bind to the frozen diagnostic.
The old five-phase worker identity must not be reused to label this one-phase
capture as a complete full-model qualification.

A future independent reader should compare every output/cotangent pair by draw
and family, reconstruct the native L1 weighting as a check on actual saved
values, count exact residual-sign changes/zeros and locate the first changed
prediction boundary. It must refuse missing pairs, altered mask identity,
changed observed loss/gradient, incomplete hooks or expanded scope. No actual
candidate CUDA kernel proof is inferred without a separate appropriate profile.

## What this observation cannot distinguish

Write `c_o,c_m` for the actual predictor-output cotangents and `J_o,J_m` for the
two complete pixels-to-predictions Jacobian paths (all sixteen outputs grouped).
The end-to-end difference has two valid ordered decompositions:

`g_mm-g_oo = (g_mo-g_oo) + (g_mm-g_mo)`

`g_mm-g_oo = (g_om-g_oo) + (g_mm-g_om)`, where `g_ab = J_a^T c_b`.

A tested CPU helper checks both identities from four *already saved* VJP cells
and retains their interaction term. No such cross-cells exist for the target
run, and none were computed here. If cotangents are identical, the external L1
cotangent-change term is absent, but `J` still includes internal forward rounding
as well as backward arithmetic. If cotangents differ, that alone does not
quantify their contribution to the input gradient. This observer is therefore
not a pure backward-causality experiment. A later same-Q/K/V, same-cotangent
attention comparison or explicitly reviewed cross-cell experiment may be needed.
Those additional calculations are outside this proposal.

## Local proof and operating envelope

29 distinct local checks pass without errors, failures or skips. Three actual
small upstream CPU model passes (before / observed / after) preserve all six
loss/input returns exactly and match the older frozen tiny baseline. All 16
actual prediction cotangents match the original FP32 L1 weighting; target and
mask bytes match the frozen CPU reference. Parameter gradients remain absent,
model tensor versions and values stay unchanged, graph hooks finish/remove,
and observer module hooks restore on success, incomplete work and exceptions.
The prototype refuses university nodes and non-tiny/non-CPU domains.

The initial test run has 23 passes and five fixture setup errors (no model in
those five cases); that receipt is retained. After correcting the test setup,
28 checks pass. Adding saved mask/target provenance yields the final 29 checks;
earlier counts are history, not additional tests. The final saved analysis
binds the final sources, and the exact earlier test source is retained by hash.
These local checks do not qualify the target GPU or pretrained observer.

The proposed target is linux10603 with a fresh private `/var/tmp` directory;
a new live check and exact host-specific approval remain necessary.
The *proposed*, unapproved envelope is 20 minutes including setup/full 2.5 GiB
proof and two fresh serial workers of at most seven minutes each, followed by
at most ten minutes / 3 GiB same-SSH collection. Setup / capture ceilings remain
3 GiB / 2.25 GiB, allocator 7 GiB, metadata 64 MiB. The proposed raw category is
512 MiB. The 22,072 target tokens across all groups require 90,406,912 logical
FP32 prediction bytes per arm. Predictions plus actual cotangents for two arms
require 361,627,648 bytes; two input gradients and two full targets raise that
to 403,570,688 bytes. A conservative allowance for 70 serialized headers and one
16 MiB atomic copy gives 424,935,424 bytes below 512 MiB. These are projections,
not disk reservations or a demonstrated live peak. Actual per-file, category,
allocated-block, free-space/inode, time and foreign-process guards must refuse
excess; no global bound may be raised to make a capture pass.

Any future payload uses the unchanged 17,192,095-byte packet and new reviewed
source overlays; weights/dependencies remain verified in place. No held-out
pixels, private keys, new training/scoring/attack work, cleanup, automatic retry,
old claim/time reset, parallel target host or paid compute is proposed.
Financial caps/holds and the unresolved reconciliation remain unchanged.

Next prepare and test a separate bounded observer implementation and handoff
from this proposal. Only after independent review should one new exact scoped
university approval be requested. There is no runnable target command yet.
