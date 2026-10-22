# Verified shared-forward CUDA control — October 21

**The complete university input-only precision control passes after independent
export verification.** This qualifies a candidate for the next integration and
training/recovery check; it does not adopt a study backend or pass full-model
optimizer recovery. The two earlier FP32 failures remain rejected.

## Verified evidence and identities

Run `precision.RvqoMB` completes all four fresh workers (native-0/1, cuDNN-0/1),
with worker and launcher exit 0. Session wall time is **267.114s**
(about 4m 27s). Verified the source archive/all 53 manifest entries, 17 historical
project sources plus five precision/dependency sources, all 32 upstream sources,
session/four worker JSON hashes, all twelve tensor-file hashes and signatures,
and original observed-session/worker/tensor pins. The actual new delegator's
native CUDA probe is recomputed from retained device-typed events in all four
workers, including positive-duration native vol2im reconstruction, convolution
backward, finite input/parameter gradients and restored flags.

Weights, checkpoint/data/preprocessing identity, original clean targets,
normalization, both attack/scoring mask banks, actual precision/geometry/
checkpointing identity and strict global runtime match their declared pins.
Each worker reports successful original-module/policy/tensor restoration and
live preservation guards; these process-local mutation checks cannot themselves
be recreated after export. Scientific comparisons and replay below are
recomputed directly from the retained tensors.

The comparison reference is explicitly **shared FP64 context-patch forward,
FP32 tokens and cuDNN FP32 backward**. The alternative keeps that exact forward
and uses native FP32 backward. This is a new declared precision/reference,
not a redefinition of the failed historical all-FP32 reference. Both arms keep
original FP32 learned parameters, normalization, frozen target and tail/L1.

## Numerical and replay checks

Every projection forward/input/weight/bias oracle gate passes in all four
workers, with zero failed values and unchanged pointwise bounds.

| Check | cuDNN backward | Native backward |
| --- | ---: | ---: |
| Maximum patch-forward error vs CPU FP64 | 1.9059e-6 | 1.9059e-6 |
| Maximum raw-pixel patch VJP error vs FP64 | 7.3113e-8 | 3.9396e-8 |
| Maximum patch-weight VJP error vs FP64 | 3.6855e-9 | 5.6348e-9 |
| Maximum patch-bias VJP error vs FP64 | 6.1534e-10 | 5.7257e-10 |

Shared forward tokens are **bitwise identical across both backwards**, and
match the CPU FP64 reference rounded to FP32 at every token value. The saved
FP64 forward reference also equals the prior independently verified trained-
projection reference bit-for-bit; all saved FP64 forward/input/weight/bias
references match across these two arms. This reuses the mathematical reference
without changing either prior failed FP32 outcome.

Both clean losses equal 0.5805841684341431 exactly. Full input-gradient maximum
cross-backward difference is **7.4506e-8**, relative L2 **4.9416e-7**, and maximum
error/allowed-error ratio **0.06826**; zero derivatives fail the original
`atol=2e-7, rtol=0.002` gate. There are 973,283 differing FP32 gradient values,
**zero active sign disagreements**, and zero zero-support disagreements.
Gradients therefore pass numerically but are not bitwise equal across backwards.

Within each backward, all saved tensors and scientific outputs replay exactly
across its two fresh processes. Each worker's full loss/input gradient also
reconstructs bit-for-bit through its own patch/tail chain rule. Both projection
parameter VJPs agree across backwards at the unchanged component bounds; this
is not a full pretrained parameter-update or AdamW recovery test.

## Attack checks on this one training clip

FGSM and PGD-5 tensors replay exactly within each backend and are also bitwise
identical **between** these two backwards here. Independent raw-pixel checks
verify [0,1] bounds and Linf radius 4/255; FGSM exactly matches the saved input-
gradient sign step. Maximum observed delta is 0.01568630337715149, within the
existing FP32 projection allowance. Worker-reported loss ascent and original
clean-candidate checks pass. Scores are copied from verified worker metadata;
without rerunning the model they are not independently reevaluated from pixels.

| Method | Reported attack-bank L1 | Reported independent-mask L1 |
| --- | ---: | ---: |
| Clean | 0.5805841684 | — |
| FGSM | 0.6229815483 | 0.6168547869 |
| PGD-5 | 0.6808600426 | 0.6302738786 |

These are diagnostic one-clip/mask-bank results, not held-out robustness,
attack-convergence evidence, or robust-overfitting findings. Exact cross-backend
attack equality here does not guarantee equality on other inputs or later
checkpoints.

## Timing and memory

Medians of **two single attempts per backend**, on the same university RTX 3070:

| Workload | cuDNN backward | Native backward | cuDNN/native ratio |
| --- | ---: | ---: | ---: |
| Clean loss + input backward | 7.923s | 1.108s | 7.15x |
| FGSM + scoring | 8.674s | 1.807s | 4.80x |
| PGD-5 + scoring | 40.745s | 6.324s | 6.44x |

These are observed input-only timings after this diagnostic's initialization,
not a steady-state benchmark, full training throughput, or an RTX 5090 estimate.
Maximum CUDA allocator peak is **3,047,505,920 bytes (2.838 GiB)**. Driver,
other users and complete trainable-model/optimizer state are outside this
measurement. The original university 8 GB full AdamW OOM is not overturned.

## Next gate

Implement the candidate as an **explicit opt-in** execution policy, preserving
actual forward precision/backward/geometry/checkpointing identities in target
caches, feature/probe work, calibration, anchors, evaluations and checkpoint
recovery. Existing legacy APIs intentionally reject the temporary diagnostic
policy; it is not ready for the training runner as-is.

Then verify CPU clean/adversarial full optimizer recovery with meaningful
parameter/target/RNG/cache checks, and perform a bounded full-model CUDA
update/resume check on sufficient VRAM. Include a few representative already-
open training inputs and strong-attack/convergence/budget measurements before
research admission. See [the next integration plan](../../docs/shared_forward_integration_plan.md).

[Curated source/hash/raw metrics](2026-10-21-university-shared-forward.json).
Raw verification scripts and outputs are ignored under
`runs/pretrained/diagnostic/shared-forward-analysis-20261021/`.
Scientific code/study YAML and pointwise limits did not change; no new GPU job,
paid compute, commit or push. Prior suite remains 373 passed/two CUDA-only
skips and was not rerun for read-only analysis/docs. Reported spend remains
$1.195; pilot reserve $8.805. Confirmation/test pixels remain closed.
