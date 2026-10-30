# University EOT work profile — October 30, 2026

**The model/observer and fresh-process checks pass. Backward dominates the
measured EOT8 controls; attention backward is the largest named device activity
in the detailed first-draw trace.** No optimization is adopted or launched.

[Machine-readable review](2026-10-30-eot-work-profile-review.json).
[Prepared scope and operation](../../docs/eot_evaluation_work_profile.md).

## Evidence and correctness

The attended run on linux10609 exited zero in **240.49 seconds**. Both fresh
workers finished within their eight-minute limits (101.23 and 95.37 seconds),
and the whole run stayed within its original 20-minute allowance. The download
completed without restarting any model. Its SSH connection may now close.

Archive: **108,589,500 bytes**, SHA256
`dbf458e501d0c1b00250068fda75680f49bee263a905b88dff48e0b6ff81fc44`.
All 100 regular archive members and extracted copies verify. Eighty-six archived
source/reference-metadata members match the prepared packet. All 61 scientific
pins, 33 upstream files, source handoff, runtime, checkpoint, clean targets and
training pixel identities match the verified reference. No held-out pixels open.

An independent audit verifies 24 tensor signatures and 32 byte-exact comparisons
across EOT8 controls, first-draw observer controls, prior reference values and
fresh-worker replay. The four original numerical comparisons to the prior
university loss/input derivative have zero error under unchanged tolerances.
All 32 context/target mask signatures regenerate from the original eight attack
seeds. Both mask families remain in each EOT8 control.

Each detailed trace contains 217,117 retained events. Original draw0 executes
both context/predictor families, 96 encoder block forward/recompute calls and
48 predictor block forward/recompute calls. Native backward activity is present.
The 49 sampled GPU-process observations contain no foreign CUDA compute PID;
sampling is not a proof of uninterrupted exclusivity.

## Unprofiled controls

| Observation | Median of six full EOT8 controls |
| --- | --- |
| Forward | 2.162 s |
| Input backward | 7.914 s |
| Forward plus input backward | 10.079 s |
| Diagnostic guard pair, measured separately | 0.0132 s |

Backward consumes **78.3–78.6%** of the observed EOT8 objective time. These
medians are calculated separately; their displayed rounded values need not sum
to the median total. Guard time is outside the objective timing and is too small
relative to that unit to explain its approximately ten seconds.

Peak measured PyTorch allocation is **4.779 GiB** for EOT8 controls and
**2.756 GiB** for the first-draw controls, within the 7 GiB allocator cap. These
are input-gradient observations, not optimizer memory qualification. The
profiled-draw time is 1.0475× / 1.0363× the corresponding unprofiled-draw controls;
profiler observations are not uninstrumented throughput estimates.

## Corrected device accounting

Independent review found an error in the original accounting reader: it counted
CUDA-side copies of our nested `eot-profile/*` annotation ranges as device
kernels. The captured source, raw events and original summaries remain unchanged.
A separate offline reader, `models/eot_trace_accounting.py`, excludes those known
annotations and preserves the correction as a new artifact. Nine new CPU
regressions cover nested GPU ranges, overlap, CPU-link duplication and malformed
events. They pass with no model execution.

There are **157 excluded GPU annotation ranges** in each trace. Their nested
inclusive durations sum to 654.175 / 651.171 ms and must never be added to actual
device work. Each corrected trace has **32,255 direct device-activity events**
totalling **1,083.473 / 1,083.616 ms**. Direct interval union agrees with those
totals in these captures. The old unfiltered 1,737.648 / 1,734.787 ms totals and
approximately 0.625 linkage ratios are not valid kernel-only quantities.

CPU-linked activity sums are still 2.235 / 2.252 ms above the direct totals
(about 0.21%). That discrepancy is retained explicitly. Its cause is unproved;
CPU-linked module buckets are not used as a conserved GPU partition. CPU event
self times can include thread waits and cannot be treated as CPU utilization or
added to device durations. All device shares below use only direct events after
known GPU annotations are removed.

| Named activity in detailed draw0 | Share of direct device duration |
| --- | --- |
| Attention backward | 53.716% / 53.740% |
| Attention forward, including observed recomputation | 11.021% / 11.019% |
| Other device activity | 35.263% / 35.241% |

The largest individual attention-backward kernel family has 24 calls and
approximately **44.85%** of direct device duration. Its `k32` specialization and
call count are consistent with the predictor's 32-wide heads and twelve blocks
under two mask families. This module attribution is an inference from the pinned
architecture and kernel names, not a conserved module-level profiler attribution.
The observed attention-backward bottleneck itself follows directly from the
retained kernel names and durations.

## Interpretation and next step

The detailed profile covers **one original draw with both mask families on one
clean training clip**. Its device-duration shares are not wall-time fractions for
all eight draws, a full PGD attack, other clips or another GPU. The full EOT8
controls provide a separate before/after correctness and timing observation.
No existing study-cost input is replaced by either diagnostic unit.

Next prepare a separately identified **FP32 attention-backward/backend component
diagnostic**, with explicit numerical, memory and observed-backend gates before
timing comparisons. Existing model policies, precision, masks, attacks and
tolerances stay unchanged. A different backend is not assumed equivalent or
faster. No new GPU attempt or paid session starts in this review. linux10603 can
support an independently prepared parallel component check after fresh status
verification; before/after speed comparisons should stay on one GPU.

Full optimizer/recovery qualification, complete study affordability, research
calibration and trajectories remain pending. All known Runpod Pods remain
terminated. The $200/$160 project/compute caps, single $10 diagnostic envelope,
$9.8116 conservative encumbrance, $0.1884 unreserved allowance and older holds
remain unchanged.
