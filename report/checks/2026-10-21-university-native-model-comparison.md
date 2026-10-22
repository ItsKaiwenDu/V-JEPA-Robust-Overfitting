# University full-model native comparison — October 21

**The diagnostic completed and remains rejected.** All four workers finished,
and both backends replay exactly within their own fresh-process pair. The
corrected native backward probe passes on actual CUDA, including a positive
`vol2im_kernel` launch, no cuDNN convolution/kernel execution in its backward
window, finite input/parameter gradients and restored flags. Clean loss agrees;
input gradients fail the original elementwise acceptance gate.

## Verified export

Run `native.5S6yHj` is copied locally and verified against the separate retry
archive. All 17 project source hashes, 32 upstream hashes, four worker JSON
hashes and four tensor hashes match. Saved tensor signatures and both exact
replay comparisons were recomputed. Data, weights, target signatures, masks,
shared runtime, geometry and activation checkpointing agree across backends.
The parent exit status is 1; `complete=true` means the comparison finished,
not that it passed. [Full record](2026-10-21-university-native-model-comparison.json).

Local CPU recomputation reproduces all acceptance flags, elementwise failed
counts, maximum errors and ratios. The double-precision norm reduction differs
only in its last digits across PyTorch/CPU environments (about 1.7e-15 absolute
in relative L2); comparison uses rtol 1e-12 / atol 1e-15 for verifying those
reported reduction numbers. This does not modify any original elementwise
acceptance threshold or within-backend exact replay requirement.

## Numerical result

| Quantity | Result |
| --- | --- |
| Clean loss, cuDNN / native | 0.5805841684 / 0.5805842876 |
| Clean-loss absolute difference | 1.1920929e-7; passes declared bounds |
| Input-gradient relative L2 difference | 0.358258% |
| Maximum derivative absolute difference | 0.0006819982082 |
| Largest error / allowed tolerance | 846.316 |
| Derivatives outside elementwise bounds | 242,613 / 3,145,728 (7.7125%) |
| Nonzero reference derivatives | 1,056,768; 22.9580% exceed bounds |
| Gradient cosine similarity | 0.9999937358 |
| Nonzero derivative sign disagreements | 464 (0.043907% of active elements) |
| Zero-support disagreements | 0 |

The threshold remains `abs(error) <= 2e-7 + 2e-3 * abs(reference)` for every
input derivative. High cosine similarity does not satisfy this pointwise
criterion, and a very similar scalar loss does not establish derivative parity.
The attempt is not relabeled as passed.

## Attack consequences on this one training clip

The local original training cache matches the CUDA worker's pixel signature.
Both saved attacks were independently checked against the clean clip for
finite [0,1] pixels and radius 4/255 (with the existing rounding allowance).
Each saved FGSM result equals a sign step from its own saved clean gradient
exactly. No new pretrained model or video decoding was executed for this
post-processing; the existing verified training tensor was loaded.

| Method | Pixels differing across backends | cuDNN independent loss | Native independent loss |
| --- | --- | --- | --- |
| FGSM | 464 | 0.616876900 | 0.616772771 |
| PGD-5 | 736,998 | 0.633358955 | 0.628537416 |

FGSM differences match the 464 gradient-sign differences. The two five-step
trajectories produce many more differing pixels. Both attacks ascend their
own native attack-bank losses. A single selected training clip and mask bank
cannot establish attack superiority, held-out robustness or robust overfitting.

## Diagnostic timing

Medians of two single attempts on RTX 3070, strict FP32/TF32 off:

| Operation | cuDNN seconds | Native seconds | cuDNN / native |
| --- | ---: | ---: | ---: |
| Clean loss + input backward | 7.964 | 1.202 | 6.62x |
| FGSM + independent score | 8.479 | 1.706 | 4.97x |
| PGD-5 + independent score | 40.338 | 6.086 | 6.63x |

All workers report a whole-worker allocator peak of 3,045,736,448 bytes
(about 2.84 GiB); this excludes driver/other-user memory and is not an isolated
operator peak. Timings include first-attempt effects and unequal prior probe
warm-up. They are a useful performance signal, not steady-state throughput,
training/optimizer memory evidence, RTX 5090 evidence or budget admission.

## Interpretation and next control

The failure is numerical agreement, with a small global gradient discrepancy
and localized elementwise/sign differences. The saved export does not contain
prediction residuals or incoming patch cotangents, so it cannot establish the
cause. One hypothesis is sensitivity to forward rounding through the deep
transformer and the nonsmooth L1 residual signs; a backward discrepancy has
not been ruled out by this full-model result. Earlier random component tests
alone do not resolve the actual trained-weight/cotangent case.

The next controlled investigation is specified in
[the derivative control plan](../../docs/native_gradient_control.md): compare
both patch backward paths at identical incoming cotangents against an
independent CPU FP64 analytical reference, then examine prediction/residual
signs and reconstruct the saved end-to-end gradients. The plan is not a new
implemented or executed CUDA check. No further run or code/tolerance change
was made for this analysis.

The original rejection and raw evidence remain preserved. Native execution
is not adopted for the study. Full-model optimizer/recovery acceptance,
representative RTX 5090 timing/memory, research calibration, novelty refresh,
protocol freeze and measured budget admission remain pending. Confirmation/test
pixels remain closed. Spending stays $1.195; no new compute was launched.
