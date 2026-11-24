# Separate early-device whole-PGD preparation

The failed v2 target attempt stopped during CPU checkpoint copying under the
existing observed 4 GiB host-RSS limit. This separate adapter transfers model
destination tensors to CUDA before opening the mapped checkpoint. It preserves
CPU construction and initialization, the original strict loader and the complete
scientific route. Actual target RSS improvement and CUDA fit remain unqualified.

The [local review](../report/checks/2026-11-23-pgd-workload-preparation-v1-local-review.md)
records 17 passing adapter cases and one fresh small CPU non-gradient control.
The [design](../configs/predictor_pgd_workload_preparation_v1.json) and
[57 input pins](../configs/predictor_pgd_workload_preparation_inputs_v1.json)
describe the separate cold operator. No old entry, source, package or grant is
widened; no new attended client, scope, handoff or authorization is created.

## Preparation order and unchanged science

The new provider changes exactly two source spans, with a full-byte inverse:

1. After the original CPU construction, call `model.to('cuda')` and require the
   same module object. Check the continuing unit deadline before opening the
   checkpoint.
2. After the unchanged checkpoint load returns, check that same absolute deadline
   before proceeding to input preparation and the unit.

`canonical_state`, `load_components` and `load_checkpoint` remain byte-exact.
They hash the checkpoint, use CPU/weights-only/mmap deserialization, audit every
component's names and shapes before copying any, and use strict copying into
existing parameters. No state assignment, dtype change, missing state, meta-device
construction, alternate arithmetic or new loader is used. The teacher stays
independent and frozen. Original CPU initialization/RNG consumption remains in
the same position. The existing policy, final CUDA/eval, guard, full-model byte
observations, masks, scoring, native ownership and PGD code remain unchanged.
The final CUDA call is retained after policy application.

The runtime contains 47 files/450,548 bytes: 46 files are byte-exact to v1 and one
preparation file is derived. The executable cache remains 46 modules; upload has
55 sources, including nine supervisor slots. Primary upload is 896,039 bytes;
convergence is 1,005,007. The emitted module cache, bootstrap, source/plan/control
identities, scratch inventory and collection catalogue bind the new provider hash.
Cold primary/convergence controls are distinct from consumed target controls.

Original monitor/proof/clock/OS support and scientific files remain unchanged.
The selected catalogue retains 264/624 result paths, original archive/99 members,
4 GiB observed RSS, 7 GiB allocator, storage and metadata caps, process/thread/sample
limits, 120-second owned objective, 720-second absolute unit, 2280-second soft work,
2400 hard work, 600 same-connection collection and 3000 connection including login.
No later role receives a fresh connection clock. An added preparation check does
not replace or qualify hard OS enforcement. Sampled RSS remains non-instantaneous.

This ordering aims to avoid simultaneous CPU destination weights and resident
mapped checkpoint pages. It does not guarantee allocator release, bound touched
mapped pages, establish the prior failure's allocation breakdown or prove fit.
`mmap=True` was already present. Full-model hashing and all observed overhead remain.

## Local controls and their limits

Seventeen distinct adapter cases pass. The first 17 executions stopped at fixture
setup because the runner had not created its evidence base. Only that runner
setup changed; all 17 affected nodes then passed. Production and test source are
unchanged after the first checks. Every log, source snapshot and failure is kept.

The checks use explicit fake Torch/NumPy/model, CUDA, source-runtime, clock, signal,
process, lease, proof and memory ports. Real generated provider code runs through
those fake ports, then the original metadata worker/completion/readers and priority
collector run in private cache namespaces. Two complete fake primary/convergence
lifecycles collect 432/792 files. Four failure lifecycles (transfer, checkpoint,
continuing-clock exhaustion and original-worker RSS guard) stop later roles and
collect 171 files each. Other cases check full source inverses, pins/cache/caps,
unbound entry, synthetic allocation order and seven preparation refusal paths.
The synthetic memory figures are invented control values, not target estimates.

Proof fixtures have passed/complete FALSE and zero proof bytes; the unchanged real
reader rejects them. No real worker, GPU query, timer, PIDFD, signal, proof or network
work occurs in these checks. Opaque source/archive/pixel-payload copies and ordinary
16 MiB reserve fixtures are real filesystem work. Fake metadata replay equality
is not actual perturbed target replay or original/candidate numerical agreement.

One fresh actual Torch 2.14.1/Python 3.11.16 CPU control runs once, with 60 parameter
elements, seeds 2026112311/12/13, two small forward evaluations and zero gradients.
It compares strict loading before/after a CPU transfer, checks exact state/output
bytes, parameter identities/flags, independent frozen teacher, modes and CPU RNG,
and checks two malformed late-state cases refuse before any component copy.
The checkpoint is new, 4480 bytes, and mapped. This control neither exercises CUDA
copying nor loads the upstream/pretrained model. No closed tiny gradient, native,
PGD, clean cohort, FP64 reference or full-model trace is repeated. The target entry
continues to require the original full-size CUDA0/runtime; there is no CPU fallback.

The standalone stdlib saved audit v2 passes: 46,720 prior hashes, 51,412 prior
fixture entries, 4326 new entries, all pins/inverses and six independent full wire
parses including references, modes, mtimes, hash chains, terminal/prefix and trailing
bytes. Audit v1 rejected uppercase fake coordinator authorization fixtures; v2
requires exact explicit-FAKE basis, matching controls and identical upper/lowercase
authorization bytes inside those fixtures. Both auditors/logs remain. No production,
test or numerical control reran for this audit-only correction. The audit itself
does not import project/test code, decode tensors or recalculate scientific values.
Old FIFO LSTAT only. Device 16777233 is the prior conditional local observation;
old device records and target mappings remain unchanged.

Raw stage: `runs/pretrained/diagnostic/pgd-workload-preparation-v1-20261123`.
Seal: `sealed-workload-preparation-v1.json`. Mutable index before/after snapshots
are retained, with live indexes excluded. User alone stages, commits and pushes.

## Next local binding

Derive and test a separate client/bootstrap binding around these concrete bytes,
retaining the corrected workload claim prefix, supervised authorization kind and
claim-before-refusal recording. The existing v2 handoff is bound to the consumed
old source/control and must not be reused or rewritten. This cold helper emits
controls/catalogues/payloads only. Legacy generated outer/client intermediates
are unbound source-generation data, not a reviewed attended client for these bytes;
the outer setup/timer code was used only through explicit fake ports.

No new scope/handoff/approval request precedes that concrete local binding review.
All university grants remain consumed. Real current-route primary/convergence,
fresh perturbed replay, complete-unit timing/overhead, chosen-hardware memory and
exact recovery, economics, training-only calibration and data/protocol adoption
remain outstanding. Mechanism diagnostic stays closed. Training PGD5/EOT1,
routine EOT1, optimizer backward, policy registry, study YAML and financial caps/
holds remain unchanged. Research/training admission stays FALSE; no measured
percentage or time-to-training is established.
