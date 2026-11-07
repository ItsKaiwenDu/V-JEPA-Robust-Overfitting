# Native attention observer verified locally — November 6

The separate [native one-boundary observer](../../docs/predictor_sdpa_observer_v1.md),
bounded record format, CPU reconstruction helper and CPU-only rehearsal entry
are implemented. **37 focused checks pass**, zero failures/errors/skips.
This is local correctness evidence; no university operator, helper, control,
authorization, claim, upload, SSH, proof or target calculation is created.

## Actual native controls

Four fresh CPU processes, PIDs 87536 / 87541 / 87545 / 87551, use the real small
upstream encoder/predictor with activation checkpointing. Two run backward on
the route's creator thread and two on a different Python runner thread.
Each executes one observed native EOT-8 GraphTask with 128 actual attention
rows (64 forward / 64 backward), all four small-model owners and all original
eight draws / two families. Each selected boundary produces eight actual
FP32 tensors: Q/K/V, output, incoming SDPA grad_output and dQ/dK/dV.

The fresh before, observed and after full native loss/input returns all exactly
reproduce the same frozen CPU reference. All 24 logical byte hashes match that
reference, including signed-zero distinctions. All 32 actual captured boundary
tensors are bitwise equal across the four fresh processes. Model/target values,
versions, modes, precision and absent parameter gradients are preserved.
Tensor hooks, original private alias and native graph-root hooks are restored.

The observer changes only the selected instance-local functional alias around
the original frozen route. It returns the exact original forward tensor and
None from every gradient hook. It patches no global SDPA alias, model operation,
backend setting, autograd scheduling rule or thread whitelist. Callbacks check
the actual running native GraphTask; their threads must appear in actual native
attention recomputation. A real nested foreign GraphTask is rejected.

Native storage offsets, strides, extents, abstract alias groups, versions and
alignment modulo 256 are retained. The files are explicitly contiguous CPU
logical copies, not native physical storage. CPU reconstruction uses zeroed
recorded backings and original offsets/strides, then verifies every logical
operand hash. Unobserved padding and physical alignment are not claimed equal.

Two actual local standalone SDPA examples replay the captured output and
dQ/dK/dV exactly using those rebuilt inputs and actual native incoming gradient.
Independent saved-array NumPy FP64 reconstruction passes unchanged component
bounds for all four captures: output rtol 1e-4 / atol 2e-5, derivatives 0.002 /
2e-7. Maximum normalized analytical error is below 0.000309. The independent
review evaluates no model, autograd, SDPA, CUDA or subprocess/network work;
it checks saved CPU tensors and analytical array arithmetic only.

## Failures remain visible

The first run records ten passes and 26 fixture errors from one actual
checkpoint failure: observed forward saved 40 tensors, recomputation saved 36.
The new finite-value diagnostic had operated on live tensors. Changing only
that check to `tensor.detach()` removes its autograd bookkeeping; the second
run passes all 36 checks. Model/checkpoint configuration, original ownership
and numerical bounds stay unchanged. Actual failed tensors and all source
versions remain preserved.

A subsequent record review identified that a file appearing after writer
construction could be overwritten at publication. The new writer refuses
that collision before writing and again before publication, with one added
preservation regression check. The final run passes 37 checks in 11.41 seconds.
First/second/third sources, tests and artifacts are not overwritten or summed.
Earlier 25- and 68-check suites were not rerun.

Actual primary-backward and capture-write error paths retain their original
error, four native forward tensors or the first Q tensor respectively, partial
coverage and restored aliases/hooks. A separate write failure preserves its
actual partial file and publishes no completion marker. Tampered/rehashed
selection, graph/thread/hook, layout, offsets, aliases, logical values, native
flags, qualification labels, unsafe files and missing markers refuse. Old
passing-native and controlled-cell loaders refuse the new observation format.
Missing final publication is never reconstructed into passing evidence.

## Scope, provenance and next step

Local runtime is arm64 Python 3.11.16 / Torch 2.14.1 / NumPy 2.4.6. Captured
attention arrays are 1 × 2 × 32 × 32 in the small model. The future target
selection remains original draw 0 / family 0, predictor block 0, row 24,
shape 1 × 12 × 2048 × 32; its actual pretrained arrays are still absent.
The local analogue selects row 2. No CUDA kernel or pretrained capture,
five-phase, optimizer, timing/cost, full-model candidate or research gate passes
from these local checks. Source/class domain checks do not grant execution.

The frozen route, adapter, upstream module, all 12,027 predecessor inputs and
consumed four-cell claim reverify. The new manifest seals 12,885 files, with
separate hash-bound bitwise anchor audit. Source/doc/receipt/test histories
are retained. Scaffold/YAML/diff checks pass. No prior source or study YAML changed.

The candidate remains rejected at 3,724 original full-input entries; both
preceding four-cell orders and interaction stay intact. Same Q/K/V and incoming
gradient still allow backend-specific forward intermediates. This observer
cannot isolate backward arithmetic or justify loss/gradient substitution.

Next implement and locally test the fixed-input component worker, one frozen
FP64-reference publication, actual CUDA backend evidence and fresh replay
checks. Then prepare/review the bounded same-SSH operator and concrete handoff
from the inert 25-minute proposal. Only afterward request one new scoped
university authorization and fresh status. No user SSH action is needed now.
Consumed helpers/claims/clocks must never be reused or reset; no cleanup,
standalone retrieval, parallel target, private keys, held-out work or paid compute.

Financial caps/holds/reconciliation remain unchanged: $200 project, $160 compute,
one $10 diagnostic cap, $9.8116 conservative encumbrance / $0.1884 unreserved,
old $2 hold / $0.0792 gap and last reported $6.56 wallet. Known Runpod Pods stay
terminated.
