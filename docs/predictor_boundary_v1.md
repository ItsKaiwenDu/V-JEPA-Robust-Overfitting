# Native predictor boundary capture: prepared, unarmed

This separate handoff implements the [reviewed offline proposal](predictor_gradient_attribution_v1.md).
All 48 local checks pass. Preparation creates no new resource authorization,
attempt, SSH connection, upload, storage proof or university calculation.
The active local handoff is
`runs/pretrained/diagnostic/predictor-boundary-v1-20261106/linux10603-final`.
Control SHA256:
`7a586af8080447c0215884432d093db4cb1215c91fef04dfe24feefda72a7668`.
Old helpers and their consumed claims remain frozen and cannot be rerun.

## Observation and scientific scope

Two fresh workers run serially on linux10603: original, then predictor-math
only after the original is accepted. Each performs one native full EOT8
forward/backward on the original training clip, all eight draws and both mask
families. The target encoder, shared FP64 context projection/native FP32
backward, weights, raw pixels, masks/seeds, activation checkpointing, native L1
objective and original numerical bounds remain. No surrogate objective, extra
backward, cross-VJP cell, new oracle, profiling, timing, optimizer or attack is
added. Existing study guards must still reject the unsupported math candidate.

A source-bound observer saves all sixteen actual predictor outputs and their
actual incoming cotangents. Module and tensor hooks return None; they do not
replace a value or gradient. Captures retain original dtype, shape, source
device/layout, logical hash and file hash. Exact context/target indices are
saved separately and verified against the frozen mask identities. Every hook
must be removed on success or failure. Model values, versions, modes, target
identity and absent parameter gradients must remain unchanged.

All predictions/cotangents and the scalar loss/input gradient are saved before
numerical gates. The observed original must exactly reproduce the accepted
original values. The observed candidate must exactly reproduce the earlier
failed values before attributing that specific discrepancy. A mismatch refuses.
The latter is **descriptive reproduction of a refusal**, not acceptance of failed
data as a complete passing reference. The historical-reference checks and fresh
original comparisons use the original loss rtol 1e-5 / atol 2e-6 and input-gradient
rtol 0.002 / atol 2e-7. No tolerance is relaxed or bypassed.

The new one-phase record has its own identity and final success/failure marker.
It never claims the old five-phase completion or research qualification. Calls
and scopes remain external bounded files. One complete native GraphTask, all
owners across sixteen forward groups, actual recomputation ownership, restored
flags/adapters/hooks and all typed captures must verify. A failed or partial
record cannot serve as a passing reference. A numerically rejected candidate
may retain complete data for independent attribution review, but stays rejected.
The parent and child recheck the prior passing capture before candidate work.
Any failure stops the serial plan and the same SSH connection collects evidence.
There is no retry, fallback host, reset or standalone retrieval.

No candidate CUDA-kernel qualification follows from recorded backend-selection
flags. This observation may show whether the external L1 cotangent changed; it
cannot alone separate pure backward arithmetic from internal forward changes
inside the Jacobian. The original hypothesis and broader scientific gates stay
open until actual target data are independently reviewed.

## Exact proposed upload

The destination is one fresh private `/var/tmp` directory on
`kdu@linux10603.dc.engr.scu.edu`. The proposed payload is:

- The unchanged original packet: **17,192,095 bytes / 99 members** (one training
  clip, saved clean derivatives and frozen code).
- A new **20,037,087-byte / five-member** value-only archive. It holds the four
  unchanged saved loss/input-gradient returns and read-only provenance metadata;
  its uncompressed extent is 25,182,004 bytes. Metadata distinguishes the complete
  passing original from the failed candidate. Neither is an executable worker or
  a new resource grant. Local preparation rechecks the complete accepted review
  and all sealed predecessor files before selecting these values.
- **20 overlay files / 252,869 bytes**, including the new worker/observer,
  record/entry/operator and bound plans containing the existing financial snapshot.
- A **735,836-byte** stdin bootstrap, below the existing 2 MiB ceiling.

Checkpoint weights and installed dependencies are verified in place. No private
keys, passwords, held-out confirmation/test pixels, new weight/dependency uploads
or installations are included. The client binds the exact local directory,
public host key, all sources, input archives and predecessor claims. Copies of a
helper cannot reset its attempt. Unarmed execution refuses before a claim,
transport session or SSH connection. Authorization is supplied separately only
after an explicit new scoped approval.

## Continuing bounds

| Limit | Bound |
| --- | --- |
| Whole work, from before live preflight | 20 minutes |
| Setup within that clock | 3 minutes |
| Full allocation/write/fsync/readback proof before either archive | 2.5 GiB, at most 60 seconds |
| Fresh serial workers including their preflight | Two, at most 7 minutes each |
| Same-SSH collection after work | Up to 10 additional minutes / 3 GiB |
| Setup / capture allocated storage | 3 GiB / 2.25 GiB |
| New raw captures / metadata | 512 MiB / 64 MiB |
| Input archives and value anchors / code | 128 MiB / 128 MiB |
| Torch allocator / cache / emergency file | 7 GiB / 32 MiB / 16 MiB |
| Tensor file / calls / scopes / header / marker | 16 MiB / 8 MiB / 64 KiB / 256 KiB / 16 KiB |

Typed partial files have the same caps. Call/scope records and actual mask
indices count as metadata; observation tensors count as raw data. The actual
logical and allocated totals, free-space/inode floors, file counts, per-file
caps and continuing clocks remain enforced. Static work is 180 + 2×420 + 120 =
1,140 seconds inside 1,200; the parent reserves its final receipt allowance.
Storage projections remain projections, not a live peak or reservation.

The expected GPU UUID is `GPU-e54fae90-feeb-8106-d8c0-1974fe48555f`, RTX 3070,
8,192 MiB, driver 580.178.04. Live runtime remains Python 3.11.17, Torch
2.8.0+cu128, CUDA 12.8 and NumPy 2.4.6. A fresh snapshot and live guards must
pass. Foreign CUDA compute causes refusal; leave other processes alone. Sampling
cannot reserve continuous exclusivity. Keep the Mac awake, lid open and attended
Terminal open through work and collection (up to 30 minutes total transport).
A preflight refusal before scratch creation leaves no capture; preserve its
Terminal error. Once scratch exists, collection follows work on the same SSH
connection even after a refusal. An exit of one is possible for a preserved
numerical rejection; report it for review and never rerun the helper.

## Local verification and remaining action

The focused suite has **48 distinct passing checks**, zero errors, failures or
skips. Real fresh small CPU workers reproduce both arms' frozen loss/input
values and save all prediction/cotangent pairs. Independent saved review checks
actual tensors, masks, targets, graph/hook restoration and native L1 weighting.
CPU-only failure injections preserve complete observations after a canonical
numerical refusal and preserve the primary exception after a backward failure.
Failed candidates, missing markers, changed masks/tensors/scopes, false backend
claims and old five-phase relabelling refuse. CPU data cannot qualify live CUDA.

Cold bootstrap checks forbid scientific imports before authorization. Whole
packet/value-anchor verification, copied/unarmed helper refusal, exact control
and source binding, typed/category caps, continuing deadlines, serial stopping
and same-connection collection pass. Transport simulations exercise complete
success, canonical candidate refusal and changed-baseline stopping. Their local
authorization fixtures and mocked proof/GPU checks are test data, not resource
grants or live capacity evidence.

The first invocation missed its local output parent (14 passes, 34 setup errors).
After correcting that invocation, 45 pass and three collection simulations refuse
their default permissions. Correcting only the simulation to use production's
private umask gives 48 passing checks; receiver permissions were not relaxed.
Earlier receipts and the previous test source are retained, not summed.

Next obtain one new exact scoped approval and fresh linux10603 GPU status after
this handoff's independent saved review is complete. The preceding target grant
is consumed; this preparation itself grants no run. Financial caps/holds and
pending reconciliation remain unchanged; known Runpod Pods remain terminated.
