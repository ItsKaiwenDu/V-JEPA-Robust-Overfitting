# Fixed-input attention component check completed and independently verified

The single authorized v6 university attempt completes on linux10603, with four
fresh serial CUDA workers: original-0, math-0, original-1 and math-1. Transport
returns zero with no send errors or SSH stderr. The complete response is
228,792,987 bytes; all 239 saved files, 228,553,883 logical bytes, archive members,
private modes and nanosecond timestamps verify. The consumed claim is
`f1193936d4bb415b8ca5c793e34798b8e185b82eabd00c2f7423de26599d1bef`.
No retry or new full-model backward/native capture occurs.

## Scientific result

This is the frozen first original predictor attention call: EOT8-before, family 0,
draw 0, predictor block 0, selected by execution order. Q/K/V and output cotangent
use actual native values and layout, shape `[1, 12, 2048, 32]`. The stored plan
stays unchanged; the independent reader translates local source/capture/reference
paths in memory only. Original runtime, precision, flags, logical values, strides,
storage offsets, alias groups, native anchors and source identities all verify.

All 32 actual FP32 return arrays pass the original elementwise bounds against one
canonical FP64 reference (four saved reference arrays). Output bounds remain
rtol 1e-4 / atol 2e-5; dQ/dK/dV remain rtol 0.002 / atol 2e-7. Zero elements fail.
Math returns also pass the original comparison against the passing original worker.
For the first worker in each mode, maximum tolerance ratios are below one:

| Return | Original vs FP64 | Math vs FP64 | Math vs original |
| --- | ---: | ---: | ---: |
| Output | 0.554180 | 0.431097 | 0.713883 |
| dQ | 0.0000516 | 0.0000165 | 0.0000520 |
| dK | 0.0001626 | 0.0001101 | 0.0001868 |
| dV | 0.0000778 | 0.0000782 | 0.0000881 |

Both original workers reproduce native original output/derivatives exactly.
Untimed and profiled outputs are bitwise equal within every worker. Both attention
arms reproduce all four returns bitwise in their second fresh process. Four
worker PIDs are distinct, serial predecessor receipts bind exactly, and only the
first worker builds the canonical reference. All startup/input/flag/restoration
and numerical-policy receipts pass. Twenty-eight serialized relative-L2 summary
comparisons differ in final rounding while satisfying the existing bookkeeping
rule; exact elementwise acceptance fields and original tensor bounds remain.
No summary field is dropped or rewritten.

## Backend and resource evidence

The original traces retain 106 records and 12 positive CUDA kernel metadata
witnesses each, associated with the efficient family. Math traces retain 159
records and 16 witnesses each, associated with the exclusive math family. All
530 raw records remain intact; all 56 kernel witnesses and their associations
verify. Trace publication precedes semantic validation. This metadata association
does not isolate pure backward arithmetic or qualify study speed/cost.

Recorded runtime is Python 3.11.17, Torch 2.8.0+cu128, NumPy 2.4.6, CUDA 12.8 and
RTX 3070, under the original deterministic and precision configuration. Each
worker owns exactly one CUDA startup and retains the 7 GiB allocator cap. Peak
component allocation is 50,534,400 bytes for each original worker and 916,455,424
bytes (0.853516 GiB) for each math worker. Full-model memory and throughput remain
unqualified. The two math logs retain PyTorch's warning that the autograd thread
sets its primary cuBLAS context; completed startup, numerical, backend and resource
checks pass without suppressing that warning.

The full 2.5 GiB allocation/write/fsync/readback proof passes on local XFS scratch,
separate from shared home, before capsule upload. It is no reservation against
other writers or kernel aggregate quota. All five supervisors finish with exit
zero, no retries and no terminated descendants. Saved GPU samples match the
expected UUID/driver and show only permitted worker processes; sampling does not
prove continuous exclusivity. The session records 106.342 seconds within its
900-second work bound. All four component supervisors remain within 120 seconds,
and storage/file/category caps remain unchanged. These are diagnostic wall times,
not measured study throughput.

## Independent review and preserved state

The saved-only reviewer blocks new model/autograd/SDPA/CUDA/profile/process/
network/reference/extraction execution. It reverifies 9,736 sealed preparation
files, 18 armed files, 77,097 prior hashes, all 40 overlays, the 118-member capsule,
245 current target descriptors, 195 older target descriptors and six earlier
consumed claims. Current bytes, modes and timestamps remain unchanged. No tests
or scientific cohorts rerun; the 127 first-run preparation tests stay covered.
Current evidence and review inputs are sealed separately.

The earlier failed parent, completed producer, unrun workers, claims and clocks
stay unchanged. The full-model candidate remains rejected at 3,724 input-gradient
entries; this one fixed-input component control does not admit that candidate or
full training. No clean/adversarial optimizer, full-model capacity, attack,
performance or budget gate is newly admitted. The $200 project / $160 compute /
single $10 diagnostic caps, existing holds and pending reconciliation stay.
No new paid compute, cleanup or installation occurs.

A first local publication-helper generation hit a Python quoting syntax error
before creating its helper or changing any file. It is retained separately; the
corrected publication helper changes reports/status only. No scientific review,
worker, test or target attempt repeats because of that publication correction.

Next interpret this completed component control and prepare an offline full-model
follow-up plan under the original numerical and research gates. No university
job is active under this consumed grant. Do not repeat its helper or use it for
separate retrieval; any additional target attempt needs its own concrete reviewed
scope. [Original scope](../../docs/predictor_sdpa_continuation_v6.md).
