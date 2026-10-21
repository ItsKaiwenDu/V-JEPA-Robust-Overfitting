# TF32-off university convolution outcome — October 21, 2026

The explicit TF32-off control passed all **12 CUDA workers**, all fixed
numerical checks, and observed exact replay for all six option/gradient-mode
pairs on RTX 3070. The native backend is a strong candidate for a full-model
diagnostic: it executes the original `nn.Conv3d` with cuDNN disabled, retaining
its parameter values and state keys. No study model/backend has been changed.

## Isolated timings

The table reports forward plus backward, in milliseconds. Values are medians
of two worker medians, each with two measured repetitions after one warmup.
Validation, CPU references, profiler and progress-file overhead are excluded.
Channels-last input conversion remains included in its forward.

| Original Conv3d execution | Input gradients only | Input + weight + bias gradients |
| --- | ---: | ---: |
| Contiguous cuDNN, strict, TF32 off | 3,409.779 ms | 3,430.724 ms |
| Channels-last cuDNN, strict, TF32 off | 3,434.974 ms | 3,437.048 ms |
| Native backend, strict, TF32 off | **5.897 ms** | **7.043 ms** |
| Native isolated speedup over contiguous | **578.18x** | **487.11x** |

Native forward alone is slower (4.880 / 5.333 ms versus 1.520 / 1.512 ms),
but native backward is much faster (1.018 / 1.710 ms versus 3,408.259 /
3,429.212 ms). Strict cuDNN still uses `dgrad2d_grouped_direct_kernel`;
channels-last does not remove it. Native traces include
`aten::slow_conv_dilated3d`, native convolution backward and copies, with no
comparable multi-second cuDNN kernel. Operator attribution may overlap and
must not be summed into wall time.

These are synthetic **component** speedups on RTX 3070. They cannot predict
whole-model, whole-attack, training, or RTX 5090 speedups. Other attention,
masking, optimizer and I/O costs remain, and the sample count is small.

## Numerical agreement, replay, and precision control

Outputs and the appropriate input/weight/bias gradients pass both the CPU
FP64 analytic oracle and the original contiguous TF32-off convolution at the
original fixed tolerances. Every validation reports zero failed elements;
the largest ratio of error to its allowed elementwise bound is below one.
Exact tensor signatures match the initial validation and measured repetitions,
then match across the two independent workers for each option/mode. This does
not require different algorithms to produce bitwise-identical tensors.

Every current worker matches the corresponding [failed TF32-on attempt](2026-10-21-university-cuda-convolution.md)
in operands, analytic-oracle signatures, and recorded runtime except the
declared cuDNN TF32 flag. The unchanged reference's output maximum absolute
error against FP64 fell from 8.590e-4 to 5.543e-6; weight-gradient maximum
absolute error fell from 6.795e-2 to 4.526e-4. Both now pass their fixed combined
relative/absolute elementwise bounds. Disabling cuDNN TF32 resolves the observed
isolated precision rejection; tolerances were not relaxed. The slow cuDNN
input backward remains, so the precision fix and native-backend performance
improvement are distinct observations.

## Verified evidence and scope

The run completed in **221.718 seconds**, launcher status 0. All 12 result
hashes, five current source hashes, six upstream hashes, handoff provenance,
and all timing medians/comparisons verified. All 12 profiler records include
CUDA events. Environment: Python 3.11.17, PyTorch 2.8.0+cu128, CUDA build 12.8,
cuDNN 9.10.2 (91002), host `linux10609`. The declared runtime retains strict
determinism, warning mode off, benchmarking off, cuBLAS `:4096:8`, matmul
TF32 off, and explicitly cuDNN TF32 off. Upstream remains pinned to
`204698b45b3712590f06245fbfba32d3be539812`.

Raw export (ignored):
`runs/pretrained/diagnostic/university-convolution-precision-results-20261021/`.
Session SHA-256:
`a1ea143c5b945dece3322974375586615cc78451ca20ce15d50e7052574da19a`.
The [structured report](2026-10-21-university-cuda-convolution-precision.json)
retains worker hashes, exact comparisons, validation metrics and operator tables.
The handoff archive remains
`db7fbc8c32872ad560c8e831cecd38f0bc577a80796865a98723fe055e84cf59`.

Maximum worker allocator peak was 165,687,808 bytes (about 158 MiB). This is a
whole-worker diagnostic peak including reference validation/profiling,
excluding driver allocations; it does not establish pure candidate memory
or full-model training fit. No checkpoint, pretrained transformer, optimizer,
video pixels, or study trajectories were loaded.

## Next gate

Prepare an explicit, opt-in native patch-convolution integration. Preserve
default research behavior until validated. Bind backend scope and precision
to target-cache, anchor, and recovery identities; create fresh targets/outputs
for the chosen runtime and reject mixed contracts. Verify full-model native
loss, input/parameter gradients, finite attacks, clean frozen targets and exact
interrupted/uninterrupted recovery, then measure representative RTX 5090
throughput and memory before budget admission. The original 8 GB training
memory limit is not solved by this component check.

No option is automatically adopted and no new cloud rental is started.
Reported spending remains **$1.195**, with **$8.805** of the pilot reserve
remaining, within the unchanged $200 total / $160 compute caps. No new paid
compute, research model change, or confirmation/test access occurred here.
