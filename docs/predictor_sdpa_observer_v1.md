# Native one-boundary attention observer

This implements the observation part of the [fixed-input plan](predictor_sdpa_fixed_input_v1.md).
It does not implement the university operator or a target execution grant.
The only entry point provided here is a small local CPU rehearsal. Earlier
sources, results, numerical bounds, proposals and consumed claims stay frozen.

## Native execution and ownership

`NativeSDPAObserver` attaches only to the first original predictor block's
instance-local functional alias inside the existing graph-owned route. It
checks the frozen route/adapter source hashes, original owned closure,
unmodified upstream forward bytecode, builtin SDPA alias, mode, phase, owner
geometry and local/target domains. The original route and upstream sources
are unchanged. No global functional alias, tensor operation, backend setting,
autograd scheduling rule or allowed-thread set is patched.

The fixed target selection remains original draw 0 / family 0, predictor block
0, forward row 24. The small upstream CPU analogue uses row 2 because its
encoder and predictor each have two blocks. The observer captures only that
first native forward, not a recomputation or later mask draw. It observes
the actual post-RoPE Q/K/V and native SDPA output, then the incoming native
SDPA output gradient and native dQ/dK/dV. Each tensor hook returns None;
the returned forward tensor is the original object. No derivative is replaced.

Callbacks must belong to the running exact native GraphTask recorded by the
existing root hook, with the original PID and phase. Callback thread identities
are retained, then checked against threads observed in actual native attention
recomputation. The observer does not whitelist callback threads itself. A real
nested foreign GraphTask is rejected. Four fresh CPU workers exercise both
the creator thread and a different Python backward-runner thread.

Every handle and the private alias are restored on success and primary errors.
The original route's method, flags and graph-root cleanup still runs. Model,
target, precision, version and absent-parameter-gradient checks belong to the
caller and remain mandatory. The rehearsal saves fresh before, observed and
after full native EOT-8 loss/input returns and compares all six with the same
frozen original CPU values. The target capture still needs its own approved
worker, source/runtime bindings, hardware/resource guard and exact native
value checks before any standalone component worker can run.

## Checkpoint transparency and local history

The first local rehearsal caught a checkpoint mismatch: 40 tensors were saved
in the observed original forward versus 36 during recomputation. The new
observer's finite-value check operated on live tensors and entered autograd
bookkeeping. The selected forward and partial capture are retained as failed
evidence. Only that new check changed to operate on `tensor.detach()`. No
checkpoint configuration, model operation, ownership guard, gradient or numerical
bound changed. The next run passes all 36 focused checks.

Record review then identified a file-preservation gap: a destination appearing
after writer construction could be overwritten on publication. The new writer
now refuses unexpected existing destinations before writing and again before
publication. One preservation regression check was added. The final run passes
37 checks, zero failures/errors/skips. Exact source-first/second/third snapshots,
logs and actual captures remain; counts are not summed. Earlier 25/68-check
suites were not rerun.

The four final positive workers each finish one observed native GraphTask,
128 actual attention rows (64 forward / 64 backward), four owners, all original
eight draws and two families, four actual gradient hooks and eight raw boundary
tensors. Native before/observed/after full loss/input returns exactly reproduce
the frozen CPU reference. All 32 captured boundary tensors replay exactly
across the four fresh processes. These are actual small upstream transformers
with activation checkpointing, not the earlier six-token toy and not released
pretrained CUDA weights.

## Bounded artifacts and honest layout reconstruction

The separate `original-SDPA-one-boundary-observation-v1` record cannot qualify
the native full model, candidate, CUDA kernels, five-phase execution, speed or
research admission. The frozen passing-native and controlled-worker readers
refuse it. A successful capture marker means one observation completed; a
caller must independently establish its original-value controls.

The store writes eight fixed FP32 tensor files, each capped at 16 MiB, a
128 KiB header, at most 8 MiB of ordered call rows, 64 KiB scopes and 16 KiB
completion/failure markers. Regular-file type, owner, private mode-0700 folder,
device, links, prior hashes, exclusive partials, fsync and final marker hashes
are checked. Unexpected destination files and failed partial writes are
preserved. Source/file byte counts, actual logical hashes and finite flags are
rechecked on load; unsafe or relabelled records refuse.

Each source tensor records shape, stride, dtype/device, gradient status,
version, source storage offset/span/extent, abstract alias group and address
alignment modulo 256. Holding storage references prevents recycled storage
addresses from becoming false cross-time aliases. This observation adds memory
and copy/I/O overhead; target allocator and time guards still need measurement.
No captured timing qualifies performance.

Files contain explicit contiguous CPU logical copies; they do not preserve
native physical storage. CPU reconstruction allocates zeroed recorded backings,
rebuilds recorded offsets/strides/alias groups, copies the actual saved values
and rechecks every whole logical hash. Unobserved padding is zero-filled,
not claimed to be captured native padding. Physical address/alignment identity
across fresh allocations is not asserted. Exact native-baseline replay remains
the required later control.

Two final local examples rebuild Q/K/V and the actual SDPA incoming gradient,
then replay standalone CPU SDPA. Their output and dQ/dK/dV match the captured
native values exactly, retaining V's nonzero offset and non-dense strides.
An independent NumPy FP64 analytical softmax/VJP check passes unchanged output
1e-4/2e-5 and derivative 0.002/2e-7 bounds. The saved arrays permit independent
review without another SDPA, oracle-worker or model execution.

## Limits of this milestone and next work

This is local arm64 Python 3.11.16 / Torch 2.14.1 / NumPy 2.4.6 evidence. The
native observer class contains strict future-target domain checks, but this
checkpoint creates no target worker, SSH/upload helper, control, authorization,
attempt, storage proof or university calculation. The CPU entry refuses the
university hosts and CUDA before constructing a model or output directory.

The full-model candidate remains rejected at 3,724 original input-gradient
entries. Both preceding four-cell decomposition orders and their interaction
remain intact. Same Q/K/V and incoming gradient still allow backend-specific
forward intermediates; this observer does not isolate backward arithmetic.

Next implement/test/review the fixed-input component worker, canonical FP64
reference publication, actual CUDA backend evidence and fresh replay checks.
Then build and review the bounded same-SSH operator/transport from the inert
25-minute proposal. A concrete handoff must precede any new scoped university
approval. No GPU status or SSH action is requested now. Neither original nor
new helpers should be run on the university under a consumed grant.

All financial caps, holds and pending reconciliation remain unchanged. No paid
compute occurred and known Runpod Pods remain terminated.
