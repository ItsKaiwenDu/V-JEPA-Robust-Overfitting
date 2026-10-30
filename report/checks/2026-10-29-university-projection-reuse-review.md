# University projection reuse — independently reviewed October 29, 2026

**Correctness and fresh replay pass on one RTX 3070 training clip. The measured
speed benefit is small: 1.021×, or 2.06% less elapsed time in this diagnostic.**
This does not establish a useful full-study cost reduction or qualify the
optimization for research training. No new run starts from this review.

[Machine-readable review](2026-10-29-university-projection-reuse-review.json).
[Diagnostic scope and operation](../../docs/projection_reuse_university_diagnostic.md).

## Evidence and independent checks

The foreground university attempt completed with exit zero in 1,981.99 seconds
(33 minutes 2 seconds), inside its original 80-minute limit. All four fresh
workers finished, each within its 18-minute allowance. The two earlier empty
download folders remain preserved; the subsequent download-only command copied
the completed evidence without restarting the model.

The archive is 464,590,468 bytes, SHA256
`c62df1a3f4c7f653a4afac3d3afeeee82cd42ba0af465f4f9fa6b780dd0d9a1a`.
All 102 regular archive members and their extracted copies verify. Seventy
archived Python-source/index members match the original handoff. The handoff
manifest, 61 original scientific pins, 33 upstream source identities, checkpoint
identity and original training pixel checksum match. The remaining handoff
members were verified by the runtime source checks and remain in the original
local packet; the result export does not claim to contain every handoff member.

The original source-bound summarizer recomputes successfully on CPU. A separate
NumPy-based audit verifies 92 tensor signatures and independently recomputes
32 original-tolerance comparisons. Sixteen additional byte comparisons confirm
that full EOT8 input, projection weight/bias and loss values agree exactly
between original and reused paths in every worker. No tolerance was changed.
Both modes reproduce all saved tensors and scientific results exactly in a
fresh process. Selected adversaries and all three independent scoring-bank
results also match exactly between modes.

All four actual new delegator probes retain native CUDA `vol2im` backward
kernel evidence, no cuDNN convolution kernel, finite input/parameter gradients
and unchanged surrounding precision flags. Regenerating the 20 mask draws on
CPU reproduces all 80 context/target tensor signatures. Every attack draw still
contains both mask families. There are 16 projection requests per EOT8 attack
objective and eight per independent scoring objective; reuse computes one
forward for each identical-input bank and retains every first-order backward.

The 399 GPU-process observations contain only the four expected worker PIDs
when compute processes are present. No foreign compute process was observed.
These are sampled checks, not proof of uninterrupted GPU exclusivity.
Host linux10609, its expected GPU UUID and driver 580.178.04 match throughout.

## Measured performance and memory

The timed unit is **PGD20/R2/EOT8 plus main and two audit scoring banks** on
`v_BreastStroke_g15_c01`. Diagnostic source/input/parameter guard overhead is
included. It is broader than the earlier ordinary primary-evaluation cost unit.

| Execution | Fresh timing samples, seconds | Median, seconds | Peak primary allocation |
| --- | --- | --- | --- |
| Original recomputation | 424.534, 425.481 | 425.007 | 4.942 GiB |
| Projection reuse | 416.263, 416.212 | 416.237 | 4.971 GiB |

The observed median ratio is **1.021070×**, with **8.770 seconds saved** and
**2.0635% less elapsed time**. Paired ratios are 1.01987× and 1.02227×.
The isolated 16-request projection component improves by about 8.28×, but that
component benefit does not become a similarly large evaluation speedup.
Reuse increases measured peak primary allocation by about 29.4 MiB; both modes
remain within the 7 GiB allocator ceiling. The maximum sampled device memory
use is 5,541 MiB, which includes more than PyTorch's current allocated tensors.
No optimizer memory or recovery result is qualified by this diagnostic.

These are two observations per mode on one clip/GPU. They provide no population
bound, confidence interval or performance forecast for another GPU. No cost
calculator input is replaced by this differently scoped timing unit.

The identical attack-bank clean and selected losses are 0.561719954 and
0.724106550. Main scoring loss increases from 0.561922610 to 0.697454453;
audit-bank adversarial losses are 0.691400647 and 0.693154097. These are diagnostic
one-clip values, not research metrics. Epsilon remains 4/255 and step 2/255 is
diagnostic-only. Confirmation/test pixels remain unopened.

## Decision and next step

Keep the separate diagnostic implementation and successful correctness evidence.
The observed approximately 2% improvement supplies no evidence of the roughly
46–53% ordinary-primary speed increase sought by the earlier partial SE cost
screen. That comparison is directional: the timing units and hardware scopes
differ. Do not infer an affordable study or pay for integration solely from
this result. The original study policies/runners and full cost screen remain
unchanged, and no optimizer integration is adopted.

Next prepare an offline plan to profile the remaining evaluation work and
execution overhead before selecting another optimization. linux10603 can host
an independently prepared parallel diagnostic after its GPU status is checked;
paired before/after timings should still use the same GPU without competing
jobs. No extra university attempt has been launched or authorized by this review.

This completed attended run no longer requires its SSH connection. All known
Runpod Pods remain terminated. Financial caps and reservations remain unchanged:
$200 project, $160 compute, $40 noncompute reserve, one $10 diagnostic envelope,
$9.8116 encumbered and $0.1884 unreserved. The older $2 hold and $0.0792 gap remain;
there is no new paid allocation or full account reconciliation.
