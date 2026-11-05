# Predictor attention in the full clean EOT8 model: offline proposal

Status: **proposal only; no operator, handoff, authorization or attempt**.
The [configuration](../configs/predictor_attention_eot8_proposal_v1.json) records
this design against commit `e0cd8daf78662b2cc310ce77f3d78d9ac91df504`. It does
not change the study configuration or grant university resources.

The next question is whether math attention limited to predictor SDPA preserves
the pretrained model's clean EOT8 loss and input gradient at the original bounds,
with the encoder and clean target paths unchanged. A component result does not
answer this full-model question. Any later success would qualify this one-clip
clean diagnostic only.

## Evidence and coverage decision

The accepted [encoder-1 cohort](../report/checks/2026-11-05-attention-replay1-v1-target-review.md)
covers all four original component workers across two attempts, with exact
artifact replay and the original CPU/GPU/reconstruction gates. The
[fixed-input review](../report/checks/2026-10-30-fixed-attention-review.md)
already accepts four encoder-0 workers. Neither component needs another replay
now. Their respective token lengths, 400 and 656, cover only two encoder shapes.
No original single-parent encoder-1 session is claimed.

The [predictor saved-input review](../report/checks/2026-10-30-attention-component-review.md)
already checks four generated workers' saved operands, numerical results,
observed CUDA backends and exact replay. The portable input-generation reviewer
failed, and no predictor fixed-loader GPU attempt was run. Those gaps and the
historical approximately 1e-10 refusal remain unresolved. Existing saved-input
predictor evidence motivates a separately bounded full-model diagnostic; it
does not waive those refusals or qualify generator replay, a fixed loader or
the pretrained predictor.

The [original full EOT8 review](../report/checks/2026-10-30-eot-work-profile-review.md)
provides two passing fresh original references and one detailed first-draw
trace. The frozen original training clip is `v_BreastStroke_g15_c01`; its pixel,
released checkpoint, model state, clean-target, seed and mask identities are
recorded in the proposal configuration. The existing 17,192,095-byte profile
packet has 99 regular members / 26,446,958 logical bytes. It is authenticated
as old evidence, not rebuilt or presented as a new launcher. No checkpoint,
dependency installation, confirmation pixels or test pixels are proposed.

Reading the saved mask signatures, without regenerating masks, gives these
context/target token pairs:

| Draw | Family 0 | Family 1 |
| --- | --- | --- |
| 0 | 656 / 1,392 | 400 / 1,648 |
| 1 | 816 / 1,232 | 288 / 1,760 |
| 2 | 944 / 1,104 | 288 / 1,760 |
| 3 | 880 / 1,168 | 512 / 1,536 |
| 4 | 1,192 / 856 | 496 / 1,552 |
| 5 | 1,032 / 1,016 | 488 / 1,560 |
| 6 | 856 / 1,192 | 256 / 1,792 |
| 7 | 1,000 / 1,048 | 592 / 1,456 |

All 16 predictor sequences total 2,048 tokens; encoder contexts have 15 distinct
lengths from 256 to 1,192. The proposed change therefore covers the predictor
alone. The encoder's current execution path stays in place throughout all eight
draws and both mask families. The old trace's large k32 attention family was
consistent with predictor work, but its module attribution was an inference;
this proposal does not treat it as a conserved timing partition or speed claim.

## Backend selection and provenance requirements

The pinned upstream `RoPEAttention.forward` opens its own
`torch.backends.cuda.sdp_kernel()` context around the actual functional SDPA
call. A future adapter must select and verify effective backend flags **at that
call, after the upstream context has opened**. An outer model context alone is
insufficient evidence. All original arguments, masks, dropout, causal setting,
scales, shapes, dtypes and strides must remain source-bound.

Implement the intervention separately, with instance/path ownership for
predictor attention. The original reference route must be transparent. Unknown
owners, aliases, arguments, extra calls or source drift must refuse. A global
math switch or unscoped class patch is outside this design. Keep routing active
through the complete input backward, including non-reentrant checkpoint
recomputation, and restore methods, aliases, owner stacks and flags in `finally`,
including early-stop and error paths. Target extraction runs once per fresh
worker, before intervention, through the original frozen teacher path.

The candidate needs a separate diagnostic identity. Existing study policies,
execution guards, captured workers and numerical gates must not be widened to
admit it. Check original execution before installing the adapter and after
restoration; a study guard must refuse an active candidate. Retain the existing
shared-FP64 patch/native-FP32 backward, original normalization, deterministic
flags, FP32/no-autocast, TF32-off settings and activation checkpointing. Do not
combine this with projection reuse or alter `native_objective`'s reduction.

Observed predictor forward **and backward** CUDA execution must prove math,
while reference and encoder execution remain on their original paths. Flags
and kernel counts alone do not establish ownership. The old event export lacks
needed sequence association fields. A separate event writer/reader must retain
and test supported CPU parent/thread, forward/backward association and CUDA link
evidence; missing or ambiguous attribution refuses. API support is pending
implementation tests. Keep raw events, exclude known GPU annotation copies
from device-kernel accounting, and preserve unresolved link discrepancies.
Do not sum nested ranges into wall time or invent a conserved partition.

## Proposed fresh-worker protocol

Use linux10609, the workstation of the original pretrained EOT8 reference,
with its pinned RTX 3070 UUID, driver and Python/Torch/NumPy versions. All four
fresh workers run serially on that GPU. No concurrent second-host job is part of
this proposal. Recheck live identity, storage and foreign CUDA processes only
when a separately tested and approved operator is ready.

| Order | Worker | Required reference |
| --- | --- | --- |
| 1 | Original reference 0 | Accepted prior full EOT8 returns |
| 2 | Predictor math 0 | New complete passing reference 0; prior EOT8 for full controls |
| 3 | Original reference 1 | Prior EOT8; exact complete-tensor replay of reference 0 |
| 4 | Predictor math 1 | New complete passing reference 1; prior EOT8 for full controls; exact replay of candidate 0 |

Every worker performs exactly five loss/input-gradient passes, with no hidden
warm-up: full EOT8 before; first draw before; first draw under CPU/CUDA profiling;
first draw after; full EOT8 after. Each first-draw pass retains both families.
This totals eight full EOT8 objectives plus twelve first-draw objectives across
four workers: 76 draws and 152 mask-family model evaluations. The original
objective and mask bank remain unchanged; this work generates no adversaries.

Save the **actual rank-zero FP32 loss immediately after forward** and the actual
FP32 input gradient immediately after autograd returns, before finite or
numerical gates. Do not cast to satisfy a gate. Retain failed/nonfinite returns,
partial writes, complete state/target/mask signatures and error receipts. Use
input-gradient extraction without accumulating parameter gradients. There are
four original clean-target computations and no separate CPU/NumPy analytical
oracle calls, parameter backward, optimizer steps, training or scoring calls.
The model's existing shared-FP64 patch arithmetic remains unchanged.

Both original full controls must match the accepted prior EOT8 loss/input
return. Each candidate phase must match its corresponding new passing original
reference; its full controls must also match the prior EOT8 directly. Retain the
original bounds: loss `rtol=1e-5, atol=2e-6`; input gradient
`rtol=2e-3, atol=2e-7`. Full before/after controls must be bitwise equal within
each worker, and all three first-draw returns must be bitwise equal. Fresh
replay 1 must exactly reproduce its own mode's complete scientific tensor
signatures. No numerical tolerance substitutes for these replay requirements.
Any failure stops subsequent work; neither a smaller EOT bank nor a dtype
fallback is allowed. The configuration lists all 24 proposed acceptance gates.

## Proposed resource envelope, not an authorization

A future attempt would have one continuing 2,400-second work allowance:
180 seconds for setup including the full 2.5 GiB write/fsync/readback proof,
four workers at most 480 seconds each, and 120 seconds for final receipts.
These reservations total 2,220 seconds, leaving 180 seconds margin. The proof
has its own 60-second maximum and precedes the input archive. Same-SSH collection
would have up to 600 additional seconds / 3 GiB, without restarting calculations.
A single-use grant and claim would be required only after exact-payload review.

Use a fresh private mode-0700 local XFS `/var/tmp` directory separate from the
shared NFS home. Preserve the 3 GiB setup and 2.25 GiB capture aggregate ceilings,
initial 8 GiB free / 2,048 inodes and runtime 1 GiB / 1,024-inode floors.
The proposed Torch allocator cap is **7 GiB**, matching the prior full-model
EOT8 diagnostic and explicitly increasing the last component cap of 4 GiB.
Fit is unmeasured for this candidate; OOM is a refusal, not permission to shrink
the objective. Sampled GPU/process checks do not grant exclusive use, and
filesystem guards do not constitute an aggregate kernel quota or reservation.

Raw capture is explicitly proposed at **1,792 MiB**, up from 400 MiB:

| Raw allowance | MiB |
| --- | ---: |
| Four traces, at most 256 MiB each | 1,024 |
| Twenty input-gradient returns, at most 16 MiB each | 320 |
| One transient atomic trace pair | 256 |
| Targets, masks and scalar returns | 64 |
| Margin | 128 |
| Total | 1,792 |

Additional categories are input 128 MiB, code 128 MiB, metadata 64 MiB,
cache 32 MiB and emergency receipts 16 MiB. Together these total 2,160 MiB,
leaving 144 MiB under 2.25 GiB. A future exact input packet is capped at 64 MiB;
its transient receive pair fits the input category. Exact new packet, overlay
and bootstrap sizes remain pending. Overlay is capped at 2 MiB; stdin bootstrap
is explicitly proposed at 2 MiB rather than the preceding 1 MiB.

Each trace is limited to 800,000 events / 256 MiB, with no truncation accepted.
Typed tensor files are capped at 32 MiB, gradients at 16 MiB, scalar loss at
64 KiB, clean targets at 12 MiB per worker and mask archive at 1 MiB per worker.
The supervisor's file limit must permit the separate 256 MiB trace cap without
weakening typed-writer limits. There are at most 1,024 files, a 16 MiB log and
4,096 SDPA call records per worker. Gradient-pass forwards account for 912
encoder and 456 predictor block calls per worker; one possible checkpoint
recomputation each gives 2,736 calls before any teacher bookkeeping. Actual
recomputation counts must be observed and bounded, never assumed exactly two.
All byte, event, duration and memory values here are proposed ceilings or static
arithmetic, not measured future utilization or guaranteed capacity.

No retries, claim resets, deadline extensions, foreign-process termination,
old-scratch deletion or paid work are included. All earlier grants are consumed.
The $200 project / $160 compute / single $10 diagnostic caps remain unchanged,
with conservative encumbrance $9.8116, unreserved $0.1884, historical $2 hold,
$0.0792 gap, last reported wallet $6.56 and reconciliation incomplete. This
proposal neither releases holds nor creates a paid reservation; known Runpod
Pods remain terminated.

## Implementation and later gates

Next implement the separately named **unarmed** adapter, full-model worker,
typed capture writer, owner-aware event reader, scratch supervisor and collector.
Test real cold tiny CPU checkpoint analogues, transparent reference behavior,
actual-call flags and predictor-only routing, restoration on every exit,
before-gate rank/dtype capture, failed/nonfinite results, exact references and
replay, target/state immutability, no parameter-gradient accumulation, and
missing/ambiguous event association refusals. CPU tests cannot qualify CUDA.
Cold-import the full packet with scientific dispatch blocked and test resource
and transport failures, proof ordering and refusal before SSH/claim when unarmed.
Then independently review the exact tested sources, payload and bounds before
asking for a new scoped university resource exception. No exception is requested
by this proposal, and no launch helper or archive is prepared now.

After any later candidate run, independently review the saved one-clip clean
EOT8 result. Adversarial-input/PGD and independent-scoring behavior, parameter
gradients, optimizer/checkpoint recovery, ordinary uninstrumented matched cost
units, complete study affordability, feature diagnostics, calibration,
evaluation scheduling and research trajectories remain separate gates.
Instrumentation or a passing clean diagnostic cannot qualify speed, budget or
research adoption. Both university sessions can stay idle during preparation.
