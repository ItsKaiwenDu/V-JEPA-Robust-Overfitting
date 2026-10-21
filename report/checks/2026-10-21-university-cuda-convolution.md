# University convolution comparison — October 21, 2026

All 12 workers completed on RTX 3070 in **86.498 seconds**, without worker
errors, timeouts, or out-of-memory failures. The launcher correctly returned
**1**: every worker rejected its numerical validation before timing, including
the original contiguous Conv3d reference. No option qualified; there is no
speedup or exact replay result from this attempt.

The [structured report](2026-10-21-university-cuda-convolution.json) preserves
the fixed tolerances, every worker's validation metrics and hash, provenance,
and interpretation. All 12 result hashes, five v1 source files, six upstream
files, and the recomputed comparisons verified. The sources also match the
original handoff archive (`d1b546513dd759eb9c109587246ad7e5fec1cddaebffff126517825c3a632ffc`).
Raw records remain ignored under
`runs/pretrained/diagnostic/university-convolution-results-20261021/`.
Session SHA-256:
`52a16013dfaebc38fd12de062db6923dfb01b5fe751c6469bde279b73eb77650`.

## Numerical observations

The following values are from parameter-gradient replay 0; the other replay
has the same error metrics. Output, input, weight, and bias comparisons use
the previously declared elementwise relative/absolute tolerances. They have
not been relaxed after this failure.

| Option | FP64 output check | FP64 input gradient | FP64 weight gradient | FP64 bias gradient | Complete gate |
| --- | --- | --- | --- | --- | --- |
| Original contiguous cuDNN | Fail | Pass | Fail | Pass | Reject |
| Channels-last cuDNN | Fail | Pass | Fail | Pass | Reject |
| Native backend | Pass | Pass | Pass | Pass | Reject: differs from cuDNN reference |

The original cuDNN output has maximum absolute error **8.590e-4**, relative L2
error **2.928e-4**, and 1,472,552 out-of-tolerance entries. Its weight gradient
has maximum absolute error **6.795e-2**, relative L2 error **2.935e-4**, and
1,226,398 out-of-tolerance entries. Channels-last matches this contiguous
reference with zero maximum absolute difference for all checked tensors.

Native output has maximum absolute error **2.678e-6** against FP64; native
weight gradient has maximum absolute error **1.354e-4**. Both pass their fixed
gates, as do input and bias gradients. However, output and weight gradients
then fail agreement with the original cuDNN reference. Passing the independent
reference alone cannot qualify the native option as a compatible replacement.

## Precision mismatch in the diagnostic

Every worker records `cudnn.allow_tf32=True`, while matmul TF32 is false.
PyTorch documents that TF32 can round convolution inputs while retaining FP32
accumulation and that convolution and matrix-multiplication controls are
separate. See [PyTorch 2.8 CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

**Inference:** the original TF32-permitted cuDNN reference is incompatible
with these full-precision FP64 tolerances. The error pattern and native
agreement make TF32 the leading explanation, but this run does not establish
causation: no TF32-off control or kernel profile was reached. Deterministic
execution and FP32 tensor storage do not establish full FP32 arithmetic
accuracy. The comparison should have included an explicit precision control
before using the unchanged cuDNN output as an FP32 accuracy gate.

The numerical guard operated as implemented. It rejected every option before
warmup, measured repetitions, or optional profiling. Repeated error metrics
are not exact replay evidence: rejected workers did not save output/gradient
signatures. Peak CUDA allocator usage was 199,239,168 bytes (about 190 MiB),
excluding driver allocations and including reference validation work.

## Next control and scope

Prepare a separately tagged **TF32-off diagnostic** for the same original
Conv3d, retaining strict determinism, existing tolerances, source checks,
worker/overall deadlines, and both gradient modes. Preserve this failed
TF32-on attempt and its archive. Do not widen tolerances or silently adopt a
precision/backend change in the study. The precision-controlled CUDA run,
representative RTX 5090 check, and full-model recovery remain pending.

No weights, videos, optimizer, or held-out pixels were loaded. No paid compute
was used; reported spending remains $1.195 with $8.805 of the pilot reserve
remaining. No research model or study configuration was changed.
