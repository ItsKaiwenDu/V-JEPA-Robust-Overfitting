# Predictor EOT8 on linux10603: reviewed replacement, unarmed

Linux10603's user-pasted status shows the expected RTX 3070 UUID
`GPU-e54fae90-feeb-8106-d8c0-1974fe48555f`, driver 580.178.04, 94 MiB used and
an empty CUDA compute list. This is an availability snapshot, not a reservation.
The live guard must still pass before setup and throughout the run. Linux10609
was blocked by a separate CUDA process; leave that process alone.

The existing approval names linux10609 and its exact control. It remains
unspent and unarmed. The new control is
`29b781e522f16396940cf4e014ff5aa6fe4699bfa2d6f6c49da4385650c8fa56`.
**One explicit replacement approval is required before arming this host.** If
approved, use linux10603 in place of linux10609 for this diagnostic. Never run
both pending versions. No authorization or attempt claim exists in either
handoff, and no university work has started.

## What changes

The new supervisor, client, entry and worker have distinct filenames and bind
linux10603's host, GPU UUID, verified public SSH key, exact source set and the
pending control they would replace. The client refuses if the pending 10609
handoff is armed or claimed. Its authorization requires an explicit replacement
binding; the old host's authorization cannot validate as this new scope.

The worker differs from the frozen worker only in the literal live-host check
and receipt kind/actual-host field. Tensor operations, all eight draws, both mask
families, shared FP64 projection/native FP32 backward, teacher, masks, seeds and
original numerical gates are unchanged. The original adapter, typed capture
writer and owner-aware event reader are reused byte-for-byte. No hostname is
spoofed and no previous scientific/operator source is edited.

Four fresh workers run serially: original 0, predictor-math 0, original 1,
predictor-math 1. The original full controls must pass the original bounds
against the saved prior 10609 EOT8 reference. Candidates require new complete
passing same-host/same-replay references and prior full-control agreement;
fresh replay stays exact within each mode. **Cross-host pretrained agreement is
not yet qualified.** The original loss tolerances remain rtol 1e-5 / atol 2e-6,
and input-gradient tolerances rtol 2e-3 / atol 2e-7. Five passes and before-gate
actual FP32 returns stay as described in [the original plan](predictor_eot8_v1.md).

## Exact upload and bounded work

The proposed destination is a fresh private `/var/tmp` directory on
`kdu@linux10603.dc.engr.scu.edu`. The upload is the unchanged 17,192,095-byte,
99-member profile packet (one original training clip, prior clean derivatives
and frozen source) plus **14 overlay files / 185,456 bytes**. Overlays include
the old bound proposal and new host metadata, including the existing financial
snapshot. The stdin bootstrap is 593,325 bytes, below the 2 MiB cap. Checkpoint
weights and installed dependencies are verified in place; no private SSH keys,
passwords, confirmation/test pixels or new dependency installs are included.

One attended attempt permits **40 minutes total work**, including preflight,
setup (at most 180 seconds), a full 2.5 GiB write/fsync/readback proof (at most
60 seconds, before archive upload), and four serial workers (at most eight
minutes each). The continuing clock starts before preflight. Collection has up
to **10 additional minutes / 3 GiB**, through the same SSH connection after
success or refusal. Keep the local Mac Terminal and Mac awake throughout.

Setup/capture ceilings remain 3 GiB / 2.25 GiB, raw capture 1,792 MiB and Torch
allocator 7 GiB. All lower tensor, trace, inode, free-space and event caps stay.
Only owned descendants may be stopped. Foreign compute causes refusal; there
is no whitelist, retry, old-claim/clock reset, standalone retrieval, cleanup or
paid compute. Sampling does not provide exclusive use of a shared GPU.

## Saved review and limits

All **37 new host-specific checks pass**, zero errors, failures or skips. They
include four fresh tiny upstream CPU workers, unchanged worker tensor-source
comparison, host/replacement authorization refusal, pending-host guard,
unarmed transport, cold 99-member restore/import, continuing deadlines,
serial workers and bounded evidence collection. The earlier 166-check baseline
is unchanged and was not rerun.

A separate saved-file review reverified all 4,441 parent pins, latest 227 and old
199 capture metadata, exact source/payload bindings and 61 new CPU evidence
files. Forty actual FP32 returns pass 24 within-process exact, 20 original-bound
cross-mode and 20 fresh-process exact comparisons. Raw CPU profiler ownership
was reread without SDPA. It sealed 4,526 inputs with scientific dispatch,
process creation and network creation blocked during review.

These are local arm64 / Python 3.11.16 / Torch 2.14.1 / NumPy 2.4.6 analogues;
they do not establish target CUDA behavior, pretrained memory fit or speed.
If later successful, owned CUDA backend proof covers the profiled first draw
and both families only. Historical refusals and generator/fixed-loader gaps
remain unresolved; no adversarial/scoring, parameter/optimizer/recovery,
performance/cost or research admission follows from preparation.

No SSH, target upload, proof, GPU worker, cleanup or paid compute was performed.
All financial caps and holds remain unchanged: $200 project, $160 compute,
one $10 diagnostic allowance, conservative encumbrance $9.8116, unreserved
$0.1884, prior $2 hold and $0.0792 reconciliation gap. Last wallet report $6.56;
account reconciliation remains incomplete and known Runpod Pods terminated.

After one replacement approval, independently verify the exact authorization
and fresh status before giving the single attended Mac helper. Do not launch
either current unarmed helper. A stale or contested GPU check stops the run;
changing hosts again requires review.
