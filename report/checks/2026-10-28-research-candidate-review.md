# Research attack candidate and feasibility decision packet

**Recommendation for further qualification:** propose PGD-20/R2 with **eight**
fixed attack-mask draws and four independent scoring draws as primary evaluation.
Retain PGD-50/R2/EOT8 with the same calibrated step, starts and attack bank as the
stronger control, plus two independent scoring audit banks. The configuration
is explicitly **proposed, not adopted and not runnable**. Existing study/proposal
YAML, evaluator, calibration code, data, training settings and budget remain unchanged.

## Scientific decision and its limits

The complete training-only panel shows4→8draw gains of3.54–3.95%at20steps and
3.84–4.15%at50steps, positive on all10clips in all three independent scoring
banks. This supports qualifying the eight-draw candidate. Extra20→50steps
also improve mean scores; twenty steps is not declared converged. Keeping50
as a stronger control retains this sensitivity check while avoiding an automatic
choice of the largest-loss arm as primary. The strongest tested arm itself has
not been compared with a still-stronger setting.

Every result remains diagnostic at the released checkpoint. No numeric
convergence threshold, statistical significance claim or robust-overfitting
conclusion follows. The panel's percentage differences and the proposed study's
trajectory gates use different references; do not equate those percentages.
Checkpoint selection remains the complete primary-validation anchors, including
initialization with earliest ties. Routine PGD-5 is monitoring only. Changes in
interpretation under stronger attacks or audit banks must remain visible and
can make a trajectory inconclusive; no post hoc retuning to recover a signal.

The staged20-class/100-training-clip design, clean/adversarial conditions,
seed0 exploration followed conditionally by seeds11/22/33, eight runs,
30epochs,390updates/run, fixed clean target, native L1 and original training
PGD-5/step1/255 are retained. Confirmation/test pixels remain unopened.

## Exact proposed calibration

- Use the original first ten training-monitor rows at released initialization.
  Compare0.5/1/2per255 under PGD-20/R2/EOT8/score4 at epsilon4/255: **30 complete
  candidate-by-clip evaluations**. The selected step remains null until these run.
- Share weights, clean targets, runtime, sources, rows, masks and both restart
  starts across candidates. The proposal records a new seed namespace and200
  distinct actual upstream mask seeds, with no overlap with the completed panel.
- Select the highest mean independent main-bank loss, then the smaller step
  only on an exact tie. Audit banks do not select the step or adversary. Adversary
  selection remains attack-bank-only. The full research-panel seed manifest
  still needs qualification before the eventual protocol freeze.
- Keep a matched eight-draw bank and starts for primary/stronger step-only
  comparisons, and recompute fresh perturbations at every checkpoint. Reuse no
  four-draw calibration or saved step2/255 panel result as research calibration.
- Calibrate on the **chosen research runtime** after hardware/economic
  qualification. University calibration cannot be transferred automatically to
  an A40/another GPU: the existing loader checks exact runtime identity.

The current evaluator intentionally fixes primary attacks at20/R2/EOT4/score4.
It rejects the proposed eight-draw settings; that guard still passes its test.
A new opt-in, versioned evaluation/calibration/artifact path must be implemented
and tested. Old cache, evaluation, calibration and checkpoint identities cannot
be relabeled. No running command or GPU handoff is created by this proposal.

## Hardware evidence reconciled

| Hardware | Accepted evidence | Remaining qualification |
| --- | --- | --- |
| RTX5090 | Exact fresh-process clean/adversarial two-update recovery,5351logical entries per comparison, under the original bounded shared-fp64-native workflow | New candidate evaluation/source contract and chosen research-runtime qualification; no full research trajectory was established |
| A40 | Bounded clean/adversarial updates, representative numerical checks, saved attacks/features and timing units | Exact optimizer resume was not tested; candidate EOT8 primary and calibration timings are missing |
| University RTX3070,8GB | Complete20-case input-gradient/attack panel and exact fresh A replay | Earlier full AdamW OOM remains unresolved; no full training/recovery qualification or A40 cost credit |

The [RTX5090 recovery](2026-10-22-shared-forward-cuda-workflow.md) remains a
valid bounded result, not a blanket pending failure. It is not transplanted to
the A40,8GB university machine or new candidate contracts. Historical FP32
numerical rejections and all failed attempts remain intact.

## Conditional affordability requirements

The historical A40 four-draw screen remains **$221.56–$224.01** at its then
$0.61/hour quote. It already exceeds the$200 cap and is not a full-study bound.
No EOT8 price is fabricated by doubling the old attack timing, using the
university instrumentation, or treating PGD50/8 as a measured PGD20/8 unit.
Both new units, the current quote and the total candidate cost remain **null**.

Let `p` be a measured full EOT8 primary clip in seconds and `c` its measured
complete three-step calibration bundle per clip. Preserving all2580primary
clip-evaluations and ten calibration bundles, the conditional screen is:

```text
total = 50 + 1.25 * rate * (8 + F + (2580*p + 10*c)/3600)
F = 79.48707928668328 hours (historical median other-component units)
F = 81.59770684375269 hours (historical maximum-observed other-component units)
```

The$50 reserves contain the$10 pilot envelope once plus$40 noncompute. Study
compute must also fit$160 including that pilot reserve, leaving$150 for new
study compute. The eight-hour allowance is historical and **not a measured bound**.

| Hypothetical all-in rate | Optimistic EOT8 primary seconds/clip ceiling, maximum-observed-unit scenario | Median-unit scenario |
| ---: | ---: | ---: |
| $0.29/hour | 452.37 | 455.31 |
| $0.40/hour | 293.58 | 296.53 |
| $0.50/hour | 209.86 | 212.81 |
| $0.61/hour | 149.47 | 152.42 |

These ceilings **temporarily set calibration cost to zero only to derive an
optimistic necessary condition**. Positive calibration cost and additional
missing work tighten them. All other units are held at historical A40 values
for this algebra only. These are hypothetical rates, not current offers or a
prediction for another GPU. Both caps,25% headroom, all eight runs, original
evaluation counts and reserves remain intact. Missing data/feature/I/O/setup/
idle/failure costs remain unknown.

At the historical$0.61rate, even this optimistic requirement is about149–152
seconds per new primary clip, below the old four-draw189.55–191.10seconds.
That requires a demonstrated cost/performance improvement; the new profile is
not admitted at the old configuration. A lower-rate machine must be measured
for **all affected units**; passing only the primary ceiling cannot admit it.

## Next concrete implementation and measurement

1. Implement/test the opt-in EOT8 settings, matched primary/control seed contract,
   full calibration identity and rejection of incompatible old artifacts. Keep
   the current evaluator unchanged until the new path is explicitly qualified.
2. Prepare a tested bounded hardware benchmark before requesting quotes or
   deployment. Include original numerical gates, both clean/adversarial exact
   recovery, representative pretrained/trained EOT8 primary units, one three-step
   training-clip bundle, matching control timing evidence and state/save/load/
   memory/disk boundaries. A one-clip bundle is diagnostic, not research calibration.
3. The proposed two-billed-hour ceiling is not authorization or a new time
   grant. Fix the actual inclusive dollar allowance and deployment-based deadline
   only after the package, current quote and remaining diagnostic budget are
   reviewed. Previous80-minute cases/recovery grants provide no new allowance.
4. Rebuild the complete study costs using matching chosen-hardware units and
   measured bounds. If a defensible path fits the unchanged caps, then run the
   full30-evaluation calibration on that runtime, freeze the research protocol,
   and review admission before any exploratory trajectory. A cheap first pair
   alone does not justify bypassing the conditional confirmation/test budget.

## Verification

All **23 focused tests pass**, zero failures/errors/skips, and the scaffold
check passes. Tests cover scope and no-launch flags, exact seed serialization,
mask collision rejection,30-evaluation requirements, calibration/audit selection,
runtime portability refusal, unchanged evaluator guards, retained workload,
null unknown costs and budget ceilings checked against exact fraction arithmetic.
A separate verifier, importing no production budget function, reproduces all
eight conditional budget rows and confirms the retained hardware scopes and
13 source pins. It verifies that any positive calibration cost at the optimistic
ceiling exceeds the partial cap; zero calibration is not a measurement.

The [candidate configuration](../../configs/research_attack_candidate_v1.json),
[structured review](2026-10-28-research-candidate-review.json) and
[validation](2026-10-28-research-candidate-validation.json) retain the exact
proposal, ten rows/seeds, unchanged study settings, null measurements and hashes.
Raw source/test/scaffold/independent arithmetic proof are saved under
`runs/pretrained/diagnostic/research-candidate-review-20261028/`.
No model, checkpoint tensor, new pixels, GPU or remote connection is opened;
no paid compute, new hold, calibration result or research adoption is produced.
User commits separately.
