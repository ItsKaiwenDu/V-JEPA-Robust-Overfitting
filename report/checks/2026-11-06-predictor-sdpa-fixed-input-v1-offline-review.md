# Fixed-input predictor attention plan — November 6

The [inert proposal](../../configs/predictor_sdpa_fixed_input_v1.json) and
[detailed comparison plan](../../docs/predictor_sdpa_fixed_input_v1.md) are
prepared and locally tested. They create no university operator, helper,
authorization, claim, upload, SSH, storage proof or target computation.
All previous target sources/results/claims remain frozen and consumed.

## Scientific decision

The verified four-cell experiment fixes cotangents at the predictor output;
it does not fix actual attention Q/K/V or incoming SDPA-output cotangent.
Its recorded attention rows contain layouts rather than those arrays.
One native capture is therefore required before a target fixed-input replay.

The predetermined boundary is original cell 00, draw 0 / family 0, predictor
block 0, first forward SDPA row 24. Shape is 1 × 12 × 2048 × 32. Q/K strides
are 786432/65536/32/1, V strides 2359296/32/1152/1. Selection follows source
execution order rather than ranking errors. The native candidate remains
rejected at 3,724 full-input entries; neither cross cell supplies a passing
native-loss reference. Both preceding decomposition orders and interaction stay.

Proposed later work captures actual Q/K/V, SDPA output, incoming grad_output
and native dQ/dK/dV once during an unchanged original full-model EOT-8 pass.
Native loss/input must replay exactly. Four fresh serial standalone workers
then share identical saved Q/K/V and incoming gradient: original 0, math 0,
original 1, math 1. Original standalone output and derivatives must reproduce
the captured native values exactly before math proceeds. Each arm must pass
the original FP64/reference bounds, actual device-typed CUDA evidence and exact
fresh replay. No flags-only kernel claim, generation, fallback or retry.

This would compare public SDPA forward/VJP paths at identical inputs. Each
backend still creates its own forward intermediates; even equal returned
outputs do not establish identical probabilities/log-sum-exp/private state.
A pure backward-arithmetic cause remains unproved. Private backward APIs and
fabricated saved states are excluded. No loss/derivative substitution for
research, tolerance widening, backend adoption or new objective is proposed.

## Actual local checks

25 focused checks pass, zero failures/errors/skips. The first run had 24 passes
and one handwritten expected V-storage-span error; production calculation and
recorded stride were correct. Only the test changed from 9,435,648 to
9,434,112 bytes. Exact first/second source copies and logs remain. Counts are
not summed; the preceding 68-check suite was not rerun.

Two actual small CPU examples exercise direct and nonreentrant-checkpointed
forward/backward. Before/observed/after full toy loss, output and input gradient
are bitwise equal; all hooks are removed. The checkpointed example observes
one selected boundary despite two forward calls. With the same captured Q/K/V
and native incoming gradient, standalone CPU math output/dQ/dK/dV reproduce
exactly and pass the existing independent NumPy FP64 analytical bounds.
Original non-dense V layout is retained. The maximum reported normalized
oracle error is below 0.002346. Negative checks cover changed cotangent,
missing backward, primary exception cleanup, invalid/full-size/nonlocal domain,
changed source selection and promotion to execution or backward-only proof.

These are synthetic 1 × 2 × 6 × 4 local CPU examples on arm64 Python 3.11.16 /
Torch 2.14.1 / NumPy 2.4.6, not pretrained CUDA or the university runtime.
The review checks the test receipts and exact sources; it does not independently
recompute unsaved toy arrays. Local tests themselves execute the actual native
CPU replay and independent oracle. Metadata-only proposal derivation and
independent review forbid scientific imports and process/network creation.

All 11,998 sealed predecessor files reverify, including the consumed four-cell
claim. The new manifest seals 12,027 inputs. Source/plan/doc/structured-receipt
hashes, test histories and exact boundary selection bind the review. Scaffold,
YAML and diff checks pass. No historical source or study configuration changed.

## Prospective resources and next implementation

The proposal is one future attended linux10603 attempt, 25 minutes total work:
180 seconds setup including a full 2.5 GiB proof, one capture worker at most
420 seconds, four component workers at most 120 seconds each, and a continuing
120-second receipt reserve. Static envelope is 1,200 seconds within 1,500;
same-SSH collection at most 600 additional seconds / 3 GiB. These are proposed
limits, not a resource grant. There is no target command now.

Setup/capture ceilings stay 3 GiB / 2.25 GiB, raw 512 MiB, metadata 64 MiB,
input/code 128 MiB each, allocator 7 GiB and tensor file 16 MiB. Logical steady
tensor projection is 171,966,468 bytes plus a 16 MiB atomic allowance. Actual
allocation/temporary/transport peaks remain unmeasured. Layout storage spans,
offsets, alias relationships and materialization must be handled explicitly;
exact native-baseline replay is mandatory. Polling and projections reserve no
capacity. Old claims/time remain consumed, with no retries, cleanup, standalone
retrieval, parallel host, private keys, held-out work, installation or paid compute.

Next implement/test/review a separate native one-boundary observer with exact
ownership, no derivative replacement and original loss/input replay. Then build
the bounded fixed-input operator/transport and its concrete reviewed handoff.
Only afterward request one new scoped university approval and fresh status.
No speculative approval or user SSH action is needed now; both hosts may idle.

Financial caps, holds and pending reconciliation remain unchanged: $200 project,
$160 compute, one $10 diagnostic cap; $9.8116 encumbered / $0.1884 unreserved,
old $2 hold / $0.0792 gap and last reported $6.56 wallet. Known Runpod Pods stay
terminated; no paid work occurred.
