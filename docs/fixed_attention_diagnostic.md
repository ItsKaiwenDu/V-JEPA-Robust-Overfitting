# Immutable-input attention diagnostic

**October 30 outcome:** Both attended attempts are consumed. The
[656-token comparison passes; the 400-token case stops at CPU FP64 bookkeeping](../report/checks/2026-10-30-fixed-attention-review.md).
Input bytes match on both hosts, but the CPU reference discrepancy remains
unexplained. Preserve both claims and all refusal gates. Next prepare a CPU-only
reference probe with derived intermediates saved; the preparation below is
historical and authorizes no repeat launch.

This is a separately identified component diagnostic following the mixed
October 30 attention result. Every backend worker loads checksummed synthetic
inputs and fixed FP64 references from the same files. It never regenerates
FP32 operands at runtime. The earlier generator and cross-platform reconstruction
failures remain recorded and unqualified; their cause is still unknown.

| New attended job | Workstation | Tokens × heads × head dimension | Workers |
| --- | --- | --- | --- |
| Encoder family 0 | linux10609 | 656 × 16 × 64 | 4 |
| Encoder family 1 | linux10603 | 400 × 16 × 64 | 4 |

The hosts can run independently in parallel. Each job compares efficient 0,
math 0, efficient 1, math 1 sequentially on its own RTX 3070. One explicit new
invocation per host permits at most 20 minutes including remote extraction and
verification, 120 seconds per fresh worker, a 4 GiB allocator bound and five
seconds of outer kill grace. Neither a new preparation nor its old attempt
history authorizes an automatic launch or retry.

## Fixture provenance and portable loading

The packet contains three synthetic input fixtures. Predictor and encoder-0
Q/K/V/cotangent bytes come from the previous independently reviewed efficient
workers. Encoder-1 is captured once on the local CPU through the pinned upstream
post-RoPE boundary at the original 400-token geometry. Its provenance identifies
that new capture and does not claim a previous CUDA result. The predictor fixture
supports local verification; the new GPU plans run only the two encoder cases.

Each fixture stores four little-endian FP32 NumPy input arrays and four fixed
FP64 NumPy output/VJP references. The references use the separately tested
NumPy analytical softmax oracle. Descriptors bind the origin, shape, seed,
positions, original non-dense V strides, file bytes and logical tensor hashes.
Loading validates the NumPy header, extent, dtype and finite values before
constructing tensors. Missing or changed inputs reject without generation or
fallback. CPU views and device transfers preserve the recorded strides; unused
backing storage is zeroed. Seeds describe provenance, not a cross-platform
bitwise generator guarantee.

All 61 original scientific source pins and 33 upstream source bindings remain
unchanged. The new packet also binds both old consumed launch claims and plans.
The old diagnostic sources, results, summaries, packets and launch claims are
preserved byte-for-byte. No pretrained weights, pixels, trained activations,
dependency installations, private keys or budget ledger are included.

## Numerical checks and early failure evidence

The component bounds remain output rtol 1e-4 / atol 2e-5 and Q/K/V derivative
rtol 0.002 / atol 2e-7. Each worker independently reconstructs the FP64 oracle
from the loaded bytes and checks the fixed reference at the existing bookkeeping
bounds (rtol 1e-12 / atol 1e-14). Oracle reconstruction differences are kept
outside the exact result replay identity. The fixed reference bytes stay identical
across fresh workers and platforms.

Before numerical acceptance, the worker commits an immutable CPU input snapshot,
then a device-input snapshot with an exact logical-byte check. It saves untimed
outputs and derivatives before numerical or reference-comparison gates can
refuse them. Failed comparisons therefore retain their actual inputs and values.
Atomic snapshots retain an unfinished `.partial` file if writing is interrupted;
the evidence export includes it. A failure before computation still has the
packet's immutable inputs available. Partial work cannot qualify a result.

Forced backend selection still covers standalone SDPA forward and backward,
without changing the upstream model's internal backend policy. Device-typed
CUDA events must show the requested efficient or math forward/backward path;
fallback refuses. One warmup and three synchronized unprofiled passes follow
the untimed numerical/backend checks. Profiling, warmup and timing signatures
must match, and each algorithm's two fresh workers must replay exactly.
The math arm also passes the same-input efficient-reference comparison.

The independent reviewer loads the same fixture files, recomputes the FP64
reference from those bytes, verifies both early snapshots and untimed values,
and rechecks runtime, backend, coverage, numerical bounds, timing, memory and
fresh replay. A verified local upstream path can replace a remote filesystem
path during review; no FP32 operand regeneration is needed.

## Preparation evidence and operation

All 46 focused CPU tests pass with zero skips. Six actual fresh tiny CPU workers
load identical fixtures and replay math SDPA exactly. Two deliberate CPU worker
failures confirm numerical and incomparable-reference refusals keep their CPU
inputs, device inputs and untimed output artifacts. Tamper checks reject altered
arrays, unsafe headers, domains, bounds, snapshots, replay, timing and memory.
CPU fixtures cannot qualify CUDA efficient attention.

The full packet is 85,622,679 bytes and restores 139 regular members. Its nine
new/current source files match the tested CPU packet. Two additional fresh
CPU-only processes load all three full-sized fixtures with regeneration explicitly
forbidden, verify all 24 logical tensor identities, and independently compare
Torch FP64 oracles with the fixed NumPy references. The eight reused input tensors
match their prior reviewed signatures. These loader checks execute no SDPA,
model or CUDA work. Eight local transport scenarios pass across the two helpers.

The new local helpers are under the ignored directory
`runs/pretrained/diagnostic/fixed-attention-university-20261030/linux10609/`
and its `linux10603/` sibling. Each has an unarmed proposal, a single-use
`run-attended.sh` and a download-only helper. The user may explicitly opt into
each new bounded attempt by running its helper once from a local Mac Terminal.
Keep both attended SSH connections, Terminals and the Mac lid open until results
are downloaded. Each helper requests awake mode and checks its verified host
key; live UUID, driver and foreign-compute guards run before work and every
five seconds. Leave other processes alone if a guard refuses.

Old launchers must never be repeated and old claims must never be removed.
The new folders and claims are distinct, and bind the old history. If a copy
fails, preserve its files and use only the new host-specific download helper
after review. Numerical failure or incomplete export stops that host's job;
there is no queue, automatic model retry or detached/overnight guarantee.

Preparation starts no real SSH connection, upload, university GPU or paid job.
Financial state remains $9.8116 conservatively encumbered, $0.1884 unreserved
inside the single $10 diagnostic envelope, with the old $2 hold and $0.0792 gap
retained. Caps stay $200 project / $160 compute / $40 non-compute reserve.
No backend adoption, full-model/attack/optimizer qualification, cost projection
or research admission follows from the new fixture format or CPU checks.
