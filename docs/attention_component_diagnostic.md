# FP32 attention component diagnostic

**October 30 review:** Both original attended attempts are consumed. The
[predictor saved-input result is promising, but the encoder comparison is incomplete](../report/checks/2026-10-30-attention-component-review.md).
The original cross-platform input-regeneration gate also rejects; the independent
NumPy saved-input audit does not waive it. Preserve both failures and the old
launch claims. Prepare immutable synthetic input fixtures before considering a
new bounded attempt. The design and preparation below are historical.

This source-only diagnostic compares forced memory-efficient and math SDPA on
the shapes identified by the October 30 evaluation profile. It measures synthetic
attention components, with independent output and Q/K/V derivative checks.
It does not qualify a changed full model, attack, optimizer or study budget.

| Workstation | Case | Batch × heads × tokens × head dimension | Fresh workers |
| --- | --- | --- | --- |
| linux10609 | Predictor | 1 × 12 × 2048 × 32 | 4 |
| linux10603 | Encoder family 0 | 1 × 16 × 656 × 64 | 4 |
| linux10603 | Encoder family 1 | 1 × 16 × 400 × 64 | 4 |

The two workstations have different RTX 3070 GPUs and share home storage.
Their remote folders and launch claims are distinct. They can run in parallel;
within each case, backend comparisons stay sequential on the same GPU. There
is no shared result registry or lock and no concurrent workload on one GPU.

Each explicitly invoked attended command allows one attempt of at most 20 minutes,
including remote extraction and verification, plus five seconds of outer kill
grace. Each worker has at most 120 seconds and a 4 GiB allocator limit. Live
host/UUID/driver/foreign-compute checks repeat every five seconds. A refusal,
timeout or interruption stops that host's diagnostic and preserves evidence.
The operator never kills another user's process or retries a model worker.
This allocator bound does not reserve the GPU or qualify full optimizer memory.

## Inputs and numerical checks

The source packet binds the reviewed profile, its first-draw mask signatures,
all 61 original scientific source pins and 33 pinned upstream files. Actual
mask geometry is regenerated. The diagnostic creates synthetic CPU weights
and inputs, then captures Q/K/V at the pinned upstream `RoPEAttention` SDPA
boundary, after its original rotary operations. No pretrained model, weights,
pixels or trained activations are loaded. Q/K/V strides, including the non-dense
V view, are retained during device transfer; unused backing storage is zeroed.

The upstream block enters a default `torch.backends.cuda.sdp_kernel()` context
internally. An outer context alone would therefore not reliably force the
desired backend. The diagnostic temporarily intercepts the CPU operand boundary,
restores the function and flags, then runs standalone SDPA under one exclusive
`torch.nn.attention.sdpa_kernel` context covering both forward and backward.
This is not a backend policy implementation inside the full model.
PyTorch documents the context's selection/restoration behavior and possible
numerical differences between SDPA implementations in its
[2.8 context documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.attention.sdpa_kernel.html)
and [2.8 SDPA documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html).

An independent CPU FP64 softmax and analytical Q/K/V VJP oracle uses query chunks
of at most 128 tokens. Its equations are checked against dense FP64 autograd and
a central directional derivative. The predeclared **component screening bounds**
copy existing output and input-gradient bounds, without modifying those sources:

| Tensor | Relative tolerance | Absolute tolerance |
| --- | --- | --- |
| Attention output | 0.0001 | 0.00002 |
| dQ, dK, dV | 0.002 | 0.0000002 |

All elements must satisfy the finite comparison. Oracle reconstruction uses a
separate FP64 bookkeeping check (rtol 1e-12, atol 1e-14), not a relaxation of
component or historical model bounds. Both backends must pass against the oracle;
math must also pass against its same-case efficient reference before timing.
No bound is widened after observing CUDA results. Exact replay is required
within each backend, including two fresh processes. Numerical agreement across
different algorithms does not establish bitwise full-model equivalence.

Before timing, retained device-typed profiler events must demonstrate the requested
forward and backward backend. CPU operator names alone cannot establish a CUDA
launch, and fallback observations reject. The profile is untimed evidence;
one warmup and three unprofiled synchronized forward/backward passes follow.
The summary reports medians over six observations per backend and shape.
Its speed ratios are efficient-time divided by math-time: above one means the
math arm was faster for that component. Neither these observations nor the
synthetic operands estimate full attack, training or project cost.

Every worker saves operands, actual outputs/derivatives, FP64 oracle tensors,
layout/source/runtime contracts, backend events, timings and peak allocations.
Independent summary reconstructs upstream operands and the FP64 oracle, verifies
all saved hashes and comparisons, and rejects altered domain, bounds, geometry,
runtime, replay, timing or memory claims. Per-host evidence exports are capped
at 1 GiB. Numerical or backend failure is a diagnostic result to review, not
permission to change the study settings.

## Local preparation and attended use

54 focused CPU checks pass with zero skips. Six actual fresh CPU workers test
only the math backend on small upstream operands; they cannot qualify CUDA
memory-efficient attention. Full-size synthetic operand reconstruction also
matches all three original shapes and encoder mask signatures. Source-only
packets restore independently; eight local transport scenarios cover both hosts.
The final production manifest equals the final tested manifest.

The local operators live in the ignored run directory
`runs/pretrained/diagnostic/attention-components-university-20261030/`.
Each host has its own `run-attended.sh` and `download-only.sh`. Run the attended
helper once from a **local Mac Terminal**; it uploads about 291 kB of source and
metadata and uses the existing university environment/upstream checkout. Keep
the Terminal, SSH connection and Mac lid open. The Mac helper requests awake
mode with `caffeinate`. Enter university passwords promptly at each prompt.
10609 and 10603 can be launched from two local Terminal windows at the same time.

Preparation leaves both proposals unarmed. Explicit execution of a helper creates
one local claim and activates only that host's plan. Existing claims and failed
files must never be deleted to repeat an attempt. If download fails, use only
that host's download helper after review; it reads saved files and cannot restart
model work. A final exit zero still requires independent result review.

All known Runpod Pods are terminated. This free university diagnostic changes no
financial hold or cap: $200 project, $160 compute, $40 non-compute reserve and one
$10 diagnostic envelope inside compute. Conservative encumbrance stays $9.8116
with $0.1884 unreserved; the historical $2 hold and $0.0792 reconciliation gap
remain. No optimization, calibration, trajectory, research result or affordable
full study is admitted by this preparation.
