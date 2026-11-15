# Local design: EOT8 numerical capture with separate backend probes

This is a reviewed local design and calculator, not a target worker or grant.
It replaces the proposed whole-EOT8 detailed trace with two explicitly counted
attention-only probes. It keeps all eight matched numerical gradients and the
original scientific/ownership/restoration checks. No model, gradient, profiler,
GPU, SSH, storage proof or paid work runs in this step.

The prior original gradient returned and saved its input derivative before the
worker deadline. Its later stalled interval and SSH failure cause remain unknown.
This design does not assert that the profiler caused that incident or that a
smaller profile will necessarily finish.

## Proposed work and claim

One fresh numerical worker would retain PGD2/two restarts, eight draws/two families,
the original four same-input original/candidate pairs, all 37 return arrays per
objective, fixed clean targets/masks/seeds, consider-before-backward selection,
projection/scoring behavior and no-grad calls. Loss/input bounds remain
rtol 1e-5 / atol 2e-6 and rtol 0.002 / atol 2e-7. Every objective keeps actual
native roots, checkpoint recomputation, inner-task/leaf ownership and restoration
of methods, private aliases, hooks, flags, modes, versions and `.grad` absence.

Detailed full-model profiling would be absent under an explicit **new identity**.
Numerical/native receipts would record that separate scope, and would never be
published as old backend-qualified successes. The new whole-attempt result would
remain incomplete until its numerical, representative-backend, restoration,
resource and collection gates all pass. An individual numerical receipt would
not assert backend coverage. Any failure stops subsequent work and preserves
partial/provisional records; there is no retry, old-clock reset or copied pass.

Inside already-counted original objective zero, observe one actual post-RoPE
call: predictor block 0, draw 0, family 0, original forward row 24, phase
`EOT8-before`. The existing native observer's source supplies this selection.
Preserve its eight real Q/K/V/output/cotangent/VJP arrays, original layout/alias/
offset/alignment provenance, actual native graph hooks and observer restoration.
The observation adds no full-model gradient. It uses a new perturbed input,
not old partial arrays, the completed clean cohort or the fixed-input capsule.

Only after all eight numerical objectives pass would two fresh serial component
workers use that **same new canonical Q/K/V and incoming cotangent**, at
[1, 12, 2048, 32] FP32 CUDA geometry and the original backend flags:

| Probe | Forward phase | Backward phase | Additional work |
| --- | --- | --- | --- |
| Native | One actual efficient forward | One actual efficient backward | One component forward and VJP |
| Forward preserving | One actual efficient forward | One public-math recomputation under the actual custom backward, plus its math VJP | One component forward and VJP, including one nested math derivative |

The total proposed work is eight full-model gradient objectives **plus two
component VJP evaluations**, including 768 full-model initial math VJPs and one
additional component inner math VJP. No FP64 reference, CPU gradients, warmups,
untimed controls, fresh replay or completed-cohort repeats are included. The
component returns would be preserved for finiteness/provenance, without a new
component VJP-accuracy or FP64-equivalence claim.

This supplies representative backend evidence at one actual perturbed geometry.
It does not certify kernel dispatch for every EOT8 call or iterate, numerical
component accuracy, fresh GPU rollout replay, attack strength, optimizer recovery,
performance, study cost, backend adoption or full training. Those gates stay open.

## Exact coverage and reader compatibility

Each full-model objective still requires 36 owners with 16 primary and 16
checkpoint calls each: 1,152 native rows. Each candidate still requires 192
initial public-math VJPs and 384 predictor forward rows including recomputation.
The old full-trace 1,152 efficient forwards / 576 or 384 efficient backwards /
192 math invocations are historical full-EOT8 predicates. A one-call component
trace cannot pass that validator and cannot replace that coverage claim.

Two existing reader assumptions prevent direct reuse. The fixed-input reader
selects a forward backend and requires its matching backward backend. Its region
association also rejects an SDPA forward operation inside the backward phase.
The proposed hybrid intentionally has native forward and math forward/VJP inside
custom backward. A **new metadata-only reader** must explicitly verify this scope,
actual positive linked CUDA kernels and CPU ancestry. Cross-thread custom/inner
GraphTasks need unique recorded forward-thread/sequence/producer association;
timing containment and kernel names alone cannot establish identity. Missing or
ambiguous association must refuse and preserve the complete trace.

Required kernel evidence is efficient forward in both probes, efficient backward
only in the native probe, and GEMM plus forward/backward softmax inside the hybrid
math VJP. The real custom outer/inner root and all three leaf hooks must also
verify. The old all-math component worker is incompatible with this candidate;
a new route must preserve the frozen candidate's exact forward/backward arithmetic.

The earlier progress prototype also ties provisional capture to detailed
profiling. A new lifecycle must separate those requirements, retain explicit
boundary-observer exit/restoration checkpoints and preserve incomplete proof
metadata before cleanup. No old flag or guard is bypassed here.

## Joint budget

`models/predictor_pgd_split_plan_v1.py` constructs the exact 419-result namespace
and refuses unknown paths, broad suffix substitutions and incompatible partials.
The reviewed plan is `configs/predictor_pgd_split_profile_v1.json`. Its actual
runtime inventory, receipts and source package remain to be implemented.

| Budget | Proposed upper bound | Unchanged ceiling |
| --- | ---: | ---: |
| Raw files plus one 16 MiB partial | 1,852,641,328 bytes | 1,879,048,192 bytes (1,792 MiB) |
| Metadata, both traces and mutable partials | 60,702,720 bytes | 67,108,864 bytes (64 MiB) |
| Both raw traces, included in metadata | 8 MiB | 8 MiB |
| Capture category caps plus allocation reserve | 2,256 MiB | 2,304 MiB |
| Proof/setup category caps plus allocation reserve | 2,960 MiB | 3,072 MiB |
| Work including all phases | 40 minutes | 40 minutes |
| Same-SSH collection | 10 minutes | 10 minutes |

Raw accounting includes the old complete numerical/rollout arrays, the eight new
native-boundary arrays, eight component return arrays and a full partial. Each
new attention array has a 4 MiB serialized cap for its 3 MiB logical FP32 payload.
The new formula explicitly counts 48 scalar bytes omitted from the older bound;
the old bound and its report remain unchanged. Metadata includes duplicate native
proofs, provisional snapshots, journals, statuses, logs and both raw traces.
Trace space is **inside** 64 MiB, not an extra 8 MiB category.

Typed native proof byte caps use the required 1,152 rows times the unchanged
2,048-byte row ceiling, below the historical 4,096-row/8 MiB writer maxima. No
original ownership predicate is relaxed. Original-only VJP files remain empty;
candidate VJP files retain 384 rows times 8,192 bytes. One raw partial, at most
two metadata partials and at most 512 KiB of duplicated mutable metadata are
accounted. Static results/traces are exclusive; an existing result cannot be
replaced. Result files plus 256 reserved input/code/control files and three
partials fit the existing 1,024-file ceiling. Complete payload membership still
needs independent review.

The work allocation is setup 180 s, proof 60 s, numerical worker 1,680 s,
two component workers at 120 s each, finalization 120 s and overhead 120 s.
Setup includes login, upload and extraction; the proposed client clock begins
before login, and remaining work must shrink if that interval consumes its budget.
The allocation reduces the numerical worker allowance from 32 to 28 minutes to fit
the additional probes without increasing the outer limit. Actual completion in that
time is unknown; a deadline remains a failure. The full 2.5 GiB storage proof,
7 GiB GPU allocator cap, strict live GPU/storage guards and allocated/free/inode
checks remain. Setup accounting forbids raw scientific capture during the proof.

Components would use the already-reviewed raw Kineto API rather than processed
`profile.events()`, retain all records before semantic validation, and refuse
truncation beyond 20,000 events / 4 MiB per trace. These limits apply after raw
materialization and cannot preempt its allocation or guarantee stop latency.
A new parent wall/RSS supervisor is required, with a proposed 4 GiB component RSS
ceiling. Sampling is not an instantaneous allocation or address-space guarantee.
Live physical-allocation checks remain necessary; calculated logical fit alone
cannot prove target fit or successful completion.

## Validation and next implementation

47 distinct new cold design checks are covered. The first run had 45 passes and
one test-expected-value error (3,028 versus the correct 2,928 MiB proof category
sum). The calculator was unchanged. Only that corrected check and one new check
against predecessor reservations ran next; both passed. Initial/corrected source
snapshots and both logs remain. Earlier passing tests and scientific cohorts were
not repeated. These are policy/arithmetic checks, not actual model/profiler tests.

Next implement and test the hybrid raw-correlation reader on synthetic metadata,
then the bounded component route and numerical/progress/inventory integration.
Only after concrete integration and independent source/resource review could one
new exact university authorization be requested. No new target runner, helper,
authorization, SSH connection, cleanup or paid action is ready or performed.
All prior evidence, consumed grants, rejections, successful frozen cohorts and
financial caps/holds/pending reconciliation remain unchanged.
