# Whole-PGD preparation RSS failure: local source analysis

The [v2 target review](../report/checks/2026-11-23-pgd-workload-handoff-v2-target-review.md)
establishes an original-worker owned-tree host-RSS breach during checkpoint
copying. The full storage proof and collection succeeded. This analysis changes
no production source, cap, frozen scientific method, authorization or claim.

The saved preparation source orders operations as follows:

1. Require the exact Linux/Python/Torch/CUDA/NumPy runtime; configure the original
   shared-fp64-native preparation and 7 GiB CUDA allocator fraction.
2. Verify the original packet and pinned upstream source.
3. Construct the CPU ViT-L model with activation checkpointing enabled.
4. Verify the checkpoint hash, deserialize with CPU placement/weights-only/mmap,
   audit all three component states, and copy each with strict state loading.
5. Load training input, apply the model policy, move the model and input to CUDA,
   check execution guards, compute byte digests and masks, then enter the PGD unit.

The failure occurred in step 4. The parent watch source refuses an observed tree
RSS strictly above 4 GiB and terminates only held owned tasks. No numeric original
peak or allocation breakdown was published on that failure path. The traceback
contains identical model/checkpoint shapes and the owned interruption message;
it does not establish a shape mismatch, GPU OOM or network disconnection. CUDA
context activity predates full-model transfer, so the GPU sample is not a fit test.

CPU destination storage and touched mapped checkpoint pages may coexist during
the copy. This is an inference from allocation order, not a measured cause.
Memory mapping is already enabled. Source reading alone cannot tell whether a
different order will stay under both host and device limits on the chosen machine.

The next independent local implementation should make preparation allocation
order explicit, with bounded failure/cleanup observations, without widening the
original RSS/allocator/clocks or scientific route. Before a new target proposal:

1. Select and derive a separate loader/preparation variant with exact source
   identity; preserve this failed path and every old frozen entry. Review CPU,
   mapped and device storage lifetimes, checkpoint audit/copy semantics, parameter
   identities/grad flags, modes, policies, RNG, guard coverage and cleanup.
2. Use explicit fake resource/OS ports for cap refusal and cleanup ordering; where
   numerical preparation equivalence needs an actual check, use a fresh small
   independent non-gradient control with distinct fixture/source/seed identities.
   Do not rerun successful closed tiny gradients or native/PGD/clean cohorts.
3. Retain strict all-component name/shape auditing and immutable checkpoint/source
   hashes. No uninitialized parameter, lazy missing state, silent dtype change,
   shortened scientific route, parameter-gradient change, CPU/tiny target fallback
   or weakened guard can qualify as a memory repair.
4. Integrate the chosen preparation path with original owned child monitoring,
   continuing hard timers, storage/GPU gates and same-connection failure collection.
   Consider a bounded saved observation for the actual refusal value; a new
   observation must not alter resource enforcement or claim instantaneous limits.
5. Independently review the complete concrete adapter locally before preparing
   any exact future scope. All current university grants are consumed; this
   analysis creates no scope, handoff, request for approval or live execution.

No design alternative is selected or adopted here. Real current-route primary,
convergence and fresh perturbed replay, matched complete-unit performance and
explicit overhead, chosen-hardware memory/exact recovery, economics, 30 training-
only calibration evaluations and protocol adoption remain outstanding. The
mechanism diagnostic stays closed and research/training stays gated.
