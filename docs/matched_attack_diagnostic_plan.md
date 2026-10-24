# Matched attack diagnostic plan

**October 24 outcome:** the one-clip university run and fresh A-only replay now
[verify independently](../report/checks/2026-10-24-university-matched.md).
The sections below retain the October 23 preparation/design context; they do
not indicate an active job or admit research execution. No method is changed.


October23. Status: implemented and tested on small upstream CPU models; the
university GPU diagnostic has not run. See the [implementation and handoff guide](matched_attack_diagnostic.md).
This is a bounded training-only diagnostic, not an amendment to the study protocol.
No current GPU availability, full-model memory fit, runtime or recovery result is claimed.

## Question and controlled factors

The saved A40 pair changes step count, attack-bank size, mask draws and random
starts together. Its higher loss under the combined stronger setting cannot
be attributed solely to taking more steps. Separate those factors before
using the result to change a research evaluation or its budget.

Use the same released pretrained checkpoint and the already declared first
training-monitor clip, within the existing shared-fp64-native diagnostic policy.
Keep the fixed clean target, normalization, native masked L1, two restarts,
radius 4/255 and diagnostic step 2/255. Research step calibration remains separate.

| Arm | PGD steps | Attack mask draws |
| --- | ---: | ---: |
| A | 20 | 4 |
| B | 50 | 4 |
| C | 20 | 8 |
| D | 50 | 8 |

Generate one declared eight-draw attack bank; A/B use its first four draws.
Every arm uses the same pre-generated starting perturbation for each restart.
All arms score with the same independent four-draw main bank and two fixed
four-draw audit banks. Check actual upstream seed-space collisions. Sample
banks before comparing arms; do not choose masks from scoring outcomes.

Use the unchanged PGD core and attack-bank candidate selection over clean,
starts and iterates. Retain the best perturbation only by attack loss.
Independent score/audit losses must never select an iterate, restart or bank.
For each mask count, verify identical start/prefix inputs and that extending a
matched run to 50 steps cannot reduce the best *attack-bank* objective. An
independent scoring loss is not required to be monotone.

The20/EOT8 and50/EOT4 arms are diagnostic-only. Do not weaken or extend canonical
EvaluationSettings to serialize them as an existing primary/convergence method.
Use a separate named contract recording sources, actual runtime/backend,
checkpoint/target identities, exact pixels, bank seeds, starts and settings.
Do not promote these results to training or research calibration artifacts.

## Evidence and execution requirements

Before a future GPU handoff is executed, verify this separate entry point on
actual small upstream CPU models. Verify exact seed/start/prefix
pairing, bounds, clean targets and model/RNG/mode/gradient preservation;
reject incomplete, tampered or mixed contracts. Prove that altering scoring
banks cannot alter the selected adversary. Keep candidate perturbations and
sufficient hashes/trace evidence for independent recomputation.

Then prepare an explicit, checksummed university handoff using only training
pixels and the existing released checkpoint. Verify the current host and GPU
before launch. The prior A40 memory peak is not proof that it fits an 8 GB RTX 3070.
Set a reviewed allocator bound, finite per-worker/outer deadline and detached
launcher; export failures and incomplete attempts for inspection. A timeout,
OOM or missing replay is not a pass. Do not silently switch backend or loosen
numerical gates. No paid Pod or resumed terminated endpoint is authorized.

Keep this diagnostic's timings separate from A40 cost observations if executed
on another GPU. The first clip checks mechanism and plumbing; it cannot establish
study-wide convergence or choose the final primary attack. Additional declared
training-only checks and research calibration may still be needed.

## Decision after evidence

Assess step and EOT contrasts with the paired main/audit losses. Any change to
research evaluation requires protocol review, fresh calibration as needed and
an updated full budget before trajectories begin. Preserve the 20-class design,
held-out closure and all original failed evidence. A hypothetical cheap quote
or successful one-clip diagnostic is not study admission.

## Bounded implementation detail

The implementation uses one four-arm worker and one fresh A-only replay
(20 steps/EOT4) to bound repeated work. Exact first-20-step pairing is still
required within each EOT count. This does not claim fresh-process replay of
all four arms. Timings include trace instrumentation and are excluded from
economic pooling. The user deferred host inspection until tomorrow; no
university job or paid compute has started.
