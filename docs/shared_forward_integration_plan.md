# Shared-forward candidate integration — next milestone

**Status: plan only; not implemented or launched.** The university input-only
control is now [verified](../report/checks/2026-10-21-university-shared-forward.md).
It passes the original numerical limits and observed replay, with a faster
native backward on one training clip. No full optimizer recovery or study
admission follows automatically from that evidence.

## Execution and artifact contracts

Promote the diagnostic implementation to a separately named explicit opt-in
execution policy. Preserve the old default and historical policy/precision
identities. Record the FP64 context-patch accumulation algorithm, FP32 output
and learned tensors, selected backward, original target precision/geometry,
normalization, activation checkpointing, actual modules and strict global
flags in the runtime identity. Continue rejecting unsupported higher-order
operations and autocast/precision drift.

Existing research/cache APIs currently reject the temporary precision-scope
policy. Update them deliberately with the full new identity, rather than
removing their checks. A record lacking the new forward precision must not
match. Bind target caches, features, calibration, evaluations, anchors and
saved training/resume state; reject incompatible artifacts before loading
trainable weights or optimizer state. Preserve clean/frozen original targets,
trained predictor and the native masked L1 objective. No EMA or classifier
training objective is introduced.

## Local full-update and interrupted recovery checks

Use small actual upstream CPU models for clean and adversarial trajectories.
Compare uninterrupted with interrupted/resumed context and predictor updates,
AdamW state, scheduler/step/epoch counters, masks/sampler, Python/NumPy/Torch
RNG and evaluations. Require the existing exact recovery equality; do not widen
it to accept this candidate. Check actual nonzero permitted updates, unchanged
fixed targets and original parameter/state keys. Test cache/calibration/anchor
mismatches, rejection before restoration, and execution drift. Include the
independent numerical derivative evidence and meaningful projection/target
invariants, not tests that merely duplicate control flow.

Do not describe these checks as implemented or passed before they run. Keep
full checkpoint I/O and failure evidence when the pilot advances to pretrained
CUDA execution.

## CUDA pilot and admission

The university input-only workload peaks near 2.84 GiB, but its earlier full
FP32 AdamW run exceeded the 8 GB allocation. Do not infer full training fits
from this successful input-only run. Prepare an offline bounded full-model
update/recovery pilot for a GPU with sufficient VRAM only after local checks.
Provide the concrete artifact, current storage/compute quote and budget before
requesting any new deployment action. No rental is automatic.

Repeat meaningful native-loss/gradient/attack checks on a few already-open
training clips or prescribed monitor inputs. Verify clean and adversarial
updates, interrupted resume, actual new CUDA dispatch and precision, frozen
targets, evaluation barriers and fresh per-checkpoint adversaries. Keep final
and confirmation pixels closed. A one-clip FGSM/PGD-5 match is not attack
convergence or robustness evidence.

Measure full update, evaluation, checkpoint/target-cache/feature I/O and storage,
transfer and idle costs on the intended hardware; refresh the remaining budget
before study admission. Count the reported $1.195 spent and $8.805 pilot reserve
inside the unchanged $200 overall/$160 compute caps. Record any adopted
precision/protocol/configuration change explicitly after the required evidence;
current study/proposal YAML and both historical FP32 failures stay unchanged
at this planning milestone.
