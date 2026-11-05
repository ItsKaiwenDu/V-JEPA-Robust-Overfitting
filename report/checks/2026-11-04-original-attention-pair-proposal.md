# Original efficient/math pair proposed offline

The [proposal](../../docs/original_attention_pair_proposal.md) advances from the
verified original CPU return to the next original GPU-attention work. It is a
**design only**: no operator, helper, grant, claim or university job is prepared.

The proposed stage is two fresh sequential linux10603 workers using the same
400-token fixture: efficient replay 0, then math replay 0 with the completed
efficient worker as its original reference. Any failure stops the pair. All
original CPU/GPU numerical, actual-backend and within-process replay gates
remain. The original four-worker session and replay-1 coverage are not complete
or qualified by this smaller diagnostic stage.

Static source inspection verifies six iterations per worker: untimed, profiled,
warmup and three timed passes. Capture each actual GPU output/dQ/dK/dV as FP32
before acceptance, plus the actual CPU FP64 oracle return before its gate.
The original first GPU snapshot already precedes its gate; later phases have
only signatures, which motivates the additional saved arrays. The old exporter
omits NPZ files, so a future collection manifest must cover every new capture.
Original sources are not changed and no oracle calculation is executed here.

The proposed work/collection bounds stay 600/600 seconds, with at most two
120-second workers and the full 2.5 GiB proof inside the same work clock.
Setup/capture ceilings remain 3 GiB/2.25 GiB. The raw capture category is
explicitly proposed at **320 MiB**, up from 256 MiB for the preceding single
return: 240 MiB rounded steady evidence + 32 MiB transient save names leaves
48 MiB margin. The individual tensor-file cap stays 32 MiB. This is checked
static budgeting, not a measurement or guarantee that math artifacts will fit.
Refuse oversize artifacts rather than expand a bound or retry.

The accepted prerequisite, its 1,120 sealed inputs, all 166 latest files/exact
modification times and the relevant immutable sources reverify; 1,131 proposal
inputs are sealed. AST checks confirm original reference dependencies, snapshot
ordering, loop coverage and source identity. Resource arithmetic passes. No
tests are added or rerun for this documentation-only step; the unchanged
executable observer retains its 79-test preparation baseline.

Capture copies and I/O can affect the original profiler, cache warmth and
timings. Retain those observations but use no instrumented timing for a speed
claim or study-budget projection. Original failures, bounds, claims, financial
caps/holds and pending reconciliation remain unchanged. This proposal does not
resolve the historical CPU refusal, adopt an optimization or admit research.

Next implement and locally test a distinct unarmed operator/client. Freeze and
independently review its exact source/payload/control before requesting one new
GPU-execution grant. The previous grant stops before GPU transfer and is consumed;
it cannot authorize this pair. Both hosts can stay idle; no action is needed on
university SSH during this design step.

The [machine-readable review](2026-11-04-original-attention-pair-proposal.json)
records source positions, prerequisite hashes, arithmetic and limitations.
