# Fixed-input attention review — October 30, 2026

**Encoder-0 passes; encoder-1 remains incomplete.** Both exports are fully
downloaded and independently byte-verified. Fixed input bytes match on both
workstations. The encoder-1 math worker stops in CPU FP64 oracle bookkeeping,
before device-input transfer or GPU math-attention execution. Its rejection
remains a failure, with no tolerance change or retry.

The linux10609 archive is 328,707,234 bytes, SHA256
`4d4d577c40957d67eb010e18235e21c03307ea5359e5086867e26515a679ffad`,
with 166 regular members. The linux10603 archive is 128,824,276 bytes, SHA256
`1a71e16529ce258bec54cadccb8e50f2890ed150529f4a651ee9d179e418ffbc`,
with 151 regular members. All 317 archive members match the manifests and
downloaded files, including 278 source/metadata/fixture member visits. All 61
scientific pins, 33 upstream bindings, captured sources, packets and summaries
remain unchanged. Both new claims and the prior claims remain consumed and
unaltered. No remote work occurs during this review.

## Verified 656-token encoder component

Linux10609 completes all four fresh workers with exit zero. The reported
session time before archive export is 43.47s. The unchanged fixed-input reviewer
passes on the Mac with the verified local upstream checkout. Exact fixture
loading, early CPU/device snapshots, fixed-oracle bookkeeping, component bounds,
actual CUDA backend evidence and fresh-process replay all pass.

| Median over six passes per algorithm | Efficient | Math | Efficient time / math time |
| --- | ---: | ---: | ---: |
| Forward | 0.536 ms | 0.967 ms | 0.554 |
| Backward | 3.489 ms | 1.557 ms | 2.242 |
| Total | 4.024 ms | 2.513 ms | 1.601 |
| Peak allocation | 61.8 MiB | 225.4 MiB | — |

This is one synthetic, isolated 656-token SDPA component on one RTX 3070, with
two fresh processes per algorithm. It does not qualify trained-model execution,
full attention blocks, EOT8/PGD, checkpoint recomputation, optimizer memory or
training throughput. No study cost projection or policy adoption changes.

## The 400-token CPU reference refusal

Linux10603 completes only efficient replay 0. That worker passes the fixed FP64
and component checks and retains actual CUDA efficient forward/backward evidence.
Math replay 0 then stops at
`independent_CPU_FP64_forward_and_analytical_QKV_oracle`. The session exits one
after 13.08s reported before archive export; neither fresh replay 1 runs.

The passed and failed workers have identical common contracts and identical
`inputs.pt` files (SHA256
`1c4dc71ab5e5de65cd6a47747dedc24261aa95db801cb1b8c67aaef74e51dc6e`).
All four input tensor signatures match the immutable fixture. The previous
regeneration failure is not repeated here: no regeneration occurs. The new
failure is instead the worker's CPU FP64 result disagreeing with the fixed
FP64 reference at unchanged rtol 1e-12 / atol 1e-14 bounds.

| CPU bookkeeping tensor | Failed worker's reported maximum absolute error |
| --- | ---: |
| Output | 1.2846e-10 |
| dQ | 8.9826e-11 |
| dK | 3.7739e-11 |
| dV | 9.2847e-11 |

The preceding efficient worker reported errors no larger than 2.50e-16 on the
same inputs. Recomputing the failed input snapshot locally with Torch FP64 and
the independent NumPy oracle passes; local maximum errors are at most 1.25e-16.
This does not establish the cause of the failed host's discrepancy. No failed
derived CPU-oracle tensor artifact was saved, only its comparison metrics, so
the affected elements and originating operation cannot be identified from this
export. No device-input snapshot, GPU math-attention output, math backend trace
or math timing exists for the failed worker. It must not be classified as a
failed GPU math-attention comparison or a harmless rounding difference.

There is no complete encoder-1 comparison or accepted speed estimate. The fixed
input loader itself passes its byte checks in both runs; generator replay and
all earlier refusals remain unqualified rather than waived.

## Independent evidence and next step

The audit verifies 104 worker tensor signatures, 44 fixed-input snapshot tensor
comparisons, 20 actual-versus-independent-NumPy output/VJP comparisons, eight
same-input cross-backend checks and 64 fresh/cross-arm byte comparisons. The
unchanged reviewer passes encoder-0 and rejects encoder-1's incomplete session.
Eighteen GPU-monitor samples contain no foreign compute process; sampling does
not prove continuous exclusivity. All reported workers fit their 120-second
bounds and completed GPU workers remain below the 4 GiB allocator cap.

The [machine-readable review](2026-10-30-fixed-attention-review.json) binds the
raw audit source, independent reaggregation, tensor checks, original failure
metrics and all consumed claims. No diagnostic implementation or threshold was
changed during review. The prior 46 preparation checks and 14 NumPy-reader
checks remain earlier evidence, not newly rerun suites.

Next prepare a bounded CPU-only reference-reproducibility probe on linux10603
with a control host. Save computed FP64 intermediates before any acceptance
gate and record the CPU/library/runtime context. Preserve the failed input
fixture and bounds. Do not rerun the GPU launchers or adopt a backend while
the reference discrepancy remains unresolved. Completed SSH sessions may close.

No new SSH, upload, GPU or paid workload starts in review. Caps remain $200
project / $160 compute / one $10 diagnostic allowance inside compute, with
$9.8116 conservatively encumbered and $0.1884 unreserved. The historical $2
hold and $0.0792 gap remain. No financial hold is released, no optimization is
adopted, and full-model correctness, speed, affordability and research admission
remain pending.
