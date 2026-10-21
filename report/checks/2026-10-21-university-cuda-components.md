# University CUDA component outcome — October 21, 2026

The largest slowdown in this RTX 3070 component check is the original patch
embedding's Conv3d **input-gradient backward**. Its median increased from
1.102 ms to 3,399.544 ms under strict determinism: 3,085x for backward, or
1,471x including forward. Both strict profiler replays identify
`cudnn::cnn::dgrad2d_grouped_direct_kernel` inside `aten::convolution_backward`,
with 3.397 / 3.404 seconds of device time. The forward remains about 1.1 ms.
This establishes the isolated bottleneck on this host; it does not establish
which operation caused the earlier 16x whole-attack slowdown on RTX 5090.

All 48 workers completed in 168.713 seconds of diagnostic wall time. All 48
worker hashes and provenance maps match; all four local source files and six
pinned upstream files match the recorded hashes. Local recomputation reproduces
every comparison and timing median. All workers include CUDA profiler events.
The complete export stays ignored under
`runs/pretrained/diagnostic/university-component-results-20261021/`.
Its session SHA-256 is
`e85c22dbba1f8495439e9d604f1d7db798955fad9197ef5dd5f2c8760ceafcdf`.
The [structured report](2026-10-21-university-cuda-components.json) records worker
hashes, runtime identities, comparisons, and selected operator evidence.

## Measurements and replay

Environment: university `linux10609`, RTX 3070, Python 3.11.17,
PyTorch 2.8.0+cu128, CUDA build 12.8, cuDNN 9.10.2 (91002), host driver
580.178.04. Tensors are FP32; matmul TF32 is off and cuDNN TF32 remains on in
both policies, matching the prior runner. The upstream commit is
`204698b45b3712590f06245fbfba32d3be539812`; source milestone is `d00c7ff`.

Each policy has two independent workers; each worker has one warmup and two
unprofiled timed repetitions. Entries are medians of worker medians, in
milliseconds. The extra operator-profile iteration is excluded. Inputs,
parameters, cotangents, sources, and precision settings match across policies.
Replay includes within-worker and between-worker output/input-gradient hashes.
No parameter gradients were measured in this attempt.

| Case | Default forward ms | Strict forward ms | Default backward ms | Strict backward ms | Total ratio | Default / strict replay |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `patch-conv` | 1.210 | 1.109 | 1.102 | 3399.544 | 1471.11x | pass / pass |
| `encoder-mask-0` | 0.190 | 0.232 | 0.346 | 2.301 | 4.73x | pass / pass |
| `encoder-mask-1` | 0.246 | 0.209 | 0.254 | 1.201 | 2.81x | pass / pass |
| `predictor-mask-0` | 0.194 | 0.229 | 0.243 | 1.162 | 3.18x | pass / pass |
| `predictor-mask-1` | 0.325 | 0.201 | 0.202 | 1.720 | 3.65x | pass / pass |
| `predictor-sort` | 0.194 | 0.232 | 0.493 | 0.540 | 1.12x | pass / pass |
| `sdpa-encoder-0` | 1.037 | 1.083 | 3.207 | 6.664 | 1.83x | fail / pass |
| `sdpa-encoder-1` | 0.362 | 0.407 | 1.039 | 1.721 | 1.52x | fail / pass |
| `sdpa-predictor` | 2.292 | 2.320 | 7.347 | 23.630 | 2.69x | fail / pass |
| `rope-encoder-0` | 2.931 | 3.531 | 4.643 | 8.363 | 1.57x | fail / pass |
| `rope-encoder-1` | 2.395 | 3.231 | 1.897 | 2.887 | 1.43x | fail / pass |
| `rope-predictor` | 4.089 | 4.234 | 8.125 | 24.481 | 2.35x | fail / pass |

All six attention cases fail the combined default replay criterion and pass
under strict policy. Forward signatures agree between fresh workers in all
six. Input-gradient signatures differ between workers in four cases; the other
two still have within-worker gradient replay failures. This identifies a
concrete nondeterminism lead for the original workflow failure, while leaving
the full-model RTX 5090 diagnosis open. Both policies use the memory-efficient
attention operators in these profiles; these records do not show a switch to
a different attention backend. Strict attention costs increase by roughly
1.4–2.7x in total, far below the isolated convolution increase.

Mask backward becomes about 4.7–8.5x slower, but still takes only 1.2–2.3 ms.
Its strict traces include index-put/sorting work. Predictor sorting changes
little. These component timings cannot be added to predict full-model runtime:
call multiplicities, shapes, devices, and overlapping events differ.

## Interpretation and limits

The strict comparison bundles deterministic algorithms, cuDNN determinism, and
cuBLAS workspace configuration; it does not establish which individual flag
selects the slow kernel. Keep the strict policy: default attention gradients
already fail exact replay. Two process replays are observed evidence, not a
universal reproducibility guarantee.

The maximum recorded allocator peak was 162,672,128 bytes (about 155 MiB), below
the explicit 4 GiB allocator limit. This excludes driver allocations and says
nothing about full training fitting in the university GPU's 8 GB memory. No
weights, videos, optimizer, trained features, or full-model recovery were used.
Confirmation/test pixels remain closed. There is no robust-overfitting result
or research budget admission.

## Next targeted diagnostic

Compare execution choices for the **same original Conv3d** with strict settings
retained: the current contiguous reference, an explicit channels-last-3D layout,
and an isolated native-backend option only if deterministic support and bounds
permit. These are untested candidates, not an adopted optimization. Preserve
the existing parameters, state keys, precision flags, and scientific model.

Before a candidate is accepted, check outputs plus input, weight, and bias
gradients against both the pinned Conv3d and an independent patchwise reference,
with declared tolerances. Require fresh-process exact replay and measure layout
conversion, forward/backward, and memory in both input-only and parameter-gradient
modes. Keep worker/deadline/allocator bounds and preserve errors/timeouts.
A layout/backend can change floating-point reductions; numerical agreement
cannot be assumed from mathematical equivalence.

First use the existing university GPU. A representative RTX 5090 check and
full-model recovery still precede any study run. No follow-up candidate was
implemented, executed, or adopted in this analysis. No new paid compute was
used; user-reported spending remains **$1.195**, with **$8.805** of the $10
pilot reserve remaining. The $200 total / $160 compute caps are unchanged.
