# Verified university derivative attribution — October 21

**The control remains rejected. Its sole failed final check is patch forward
accuracy against the FP64 reference, for both cuDNN and native execution.**
All fixed-incoming-derivative backward checks pass. This distinguishes the
observed forward/tail sensitivity from a large patch-backward error on this
selected training clip, without accepting a backend or changing any gate.

## Evidence verification

Run `derivative.aIsykm` completed in 66.915 seconds, worker/launcher exit 1;
there was no timeout. CUDA allocator peak is 2,961,700,864 bytes (2.758 GiB),
excluding the driver and other processes. It reads the existing first training
clip, mask seed 0 and original pinned checkpoint. No optimizer or held-out data.

Verified the original handoff archive and all 51 manifest entries, 17 historical
project sources, three control sources and 32 upstream source hashes, the
reported control hash `d5c6ad7a25a8ea67afede2daa28efe5abfa748d015ce20f6e6a516f8651e43a4`,
all nine tensor-artifact hashes, prior session/worker/tensor hashes, data,
weights, clean targets, masks and runtime. Recomputed the native CUDA kernel
observer from its retained raw device-typed events. Runtime preservation
assertions passed during the worker; live mutation checks cannot themselves
be replayed retrospectively from an export.

All reported numerical comparisons, residual changes, identical-token replay,
loss reconstruction and gradient reconstruction were recomputed from saved
tensors. The complete original input-gradient failure is reproduced unchanged.
Host norm-reduction reproduction permits 1e-12 relative/1e-15 absolute differences
only in saved report floats; scientific pointwise gates are unchanged.

## Why the forward check rejected

The output gate remains `atol=2e-5, rtol=1e-4`, with 2,097,152 patch-token
values per backend.

| Forward path | Failed values | Maximum absolute error | Maximum error / allowed error |
| --- | ---: | ---: | ---: |
| cuDNN FP32 | 12 | 9.6762e-5 | 2.2489 |
| Native FP32 | 35 | 8.4888e-5 | 2.3750 |

Global relative L2 errors are about 1.19e-6 for both, but that does not pass the
pointwise check. Failed reference magnitudes range from 0.0141 to 0.2135 for
cuDNN and 0.00132 to 0.3497 for native. The largest output magnitude is 54.69.
Thus the synthetic component gate did not transfer to every learned-weight
forward value. This does not by itself establish an implementation bug or
make the error harmless for the L1 tail. Neither forward path passes here.

## Fixed incoming derivatives: all four backwards pass

The raw-pixel VJP gate remains `atol=2e-7, rtol=0.002`, including the original
RGB normalization derivative. Each source supplies the same two ordered
family derivatives to both backwards; no FP32 pre-sum changes branch order.

| Incoming derivative source | Backward path | Maximum absolute error vs FP64 | Failed values |
| --- | --- | ---: | ---: |
| cuDNN forward tokens | cuDNN | 7.7813e-8 | 0 |
| cuDNN forward tokens | Native | 3.4044e-8 | 0 |
| Native forward tokens | cuDNN | 7.1794e-8 | 0 |
| Native forward tokens | Native | 4.4669e-8 | 0 |

At identical incoming derivatives the two backward paths also agree with each
other: maximum difference 8.9407e-8 / 7.4506e-8 for original/native token
sources, with no active sign or zero-support differences. This is specific
to these operands and does not establish every training gradient/update.

## Reconstruction and forward/tail sensitivity

Both reconstructed clean losses and input gradients are **bit-for-bit equal**
to their saved replay-0 results from `native.5S6yHj`. Repeating identical
original token values exactly reproduces predictions, residuals, loss and
incoming derivatives within this worker. The intervention is therefore
faithful for these saved observations.

The two forward-token sources produce four and seven residual sign crossings
in the two L1 mask families, with no exact-zero residuals. Their incoming
derivative relative L2 differences are 0.1972% and 0.4797%. Input-gradient
relative L2 difference remains 0.3583%, including the same 242,613 pointwise
failures and 464 active sign disagreements as the original rejected run.

In the exact FP64 vector decomposition, changing the incoming derivative at
cuDNN backward has L2 norm 0.010637609; changing the backward at the native
incoming derivative has L2 norm 0.0000014683. The former norm is about 7,245
times larger. These are vector effects and their norms do not add. The
observed discrepancy is dominated by the changed incoming derivative on this
example. The eleven L1 sign crossings demonstrate a nonsmooth contribution,
but do not prove that those crossings alone cause the whole discrepancy;
the tail Jacobian also changes with its input.

## Independent FP64 reference reproduction

Rehashed the full original 5,127,726,842-byte checkpoint and checked the first
local training clip's pixel signature. Used only the original projection
weights/bias from the memory-mapped checkpoint. CPU FP64 `conv3d` and
`conv_transpose3d` independently reproduce the saved unfolded-matrix reference:
maximum output difference 1.5632e-13; raw-pixel VJP differences 1.1796e-16 and
1.2490e-16. These checks include the channel standard deviation and both mask
families. They use a separate library-convolution calculation, not the saved
analytical-reference function. Saved GPU normalized inputs were not exported;
recomputed local FP32 normalization agrees through these observed differences.

Reconstructing L1 from saved residuals matches original loss exactly and
native loss within 5.96e-8; CPU/GPU reduction order need not be bitwise equal.
This does not affect the exact same-backend loss/gradient reconstructions.
No scientific acceptance bound was changed for these reproduction checks.

## Next decision

Test a **shared, more accurate patch forward** before comparing backward paths
again. A prospective candidate is FP64 accumulation of the same learned
projection followed by an explicit cast to FP32 tokens. Preserve normalization,
parameters/geometry, the clean original target and the FP32 tail; declare the
new precision identity and reference before executing it. Start with local
CPU mathematical/identity checks, then a separate bounded university control.
This is specified only, not implemented or launched.

Simply retaining cuDNN forward with native backward would be a useful
compatibility control, but it still inherits the twelve forward-oracle
failures; it cannot resolve or bypass the failed gate. A new precision policy
must not be compared with a silently changed historical reference, or used to
retroactively pass either rejected FP32 run. Full-model optimizer recovery,
representative attack checks, RTX 5090 validation and measured budget admission
remain required before research execution.

Curated [hashes and recomputed metrics](2026-10-21-university-gradient-attribution.json).
Raw verification scripts and outputs are ignored under
`runs/pretrained/diagnostic/gradient-attribution-analysis-20261021/`.
No research implementation or study YAML changed; the prior suite remains
345 passed/two CUDA-only skips and was not rerun for read-only analysis/docs.
No new GPU job or paid compute was launched; spending remains $1.195 and the
pilot reserve $8.805. Confirmation/test pixels remain closed.
