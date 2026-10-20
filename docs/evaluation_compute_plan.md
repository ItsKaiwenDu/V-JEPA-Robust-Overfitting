# Evaluation schedule and compute proposal

**Historical October 19 proposal:** the [October 20 staged study](budget_study_plan.md)
is the latest budget candidate. This earlier schedule and its cost evidence
are retained unchanged below.

October 19, 2026. **Draft for review; no experiment or rental is admitted.**
The [machine-readable proposal](../configs/compute_plan.yaml) supplements
protocol 0.3 without changing `study.yaml`. The
[reproducible calculation](../report/checks/2026-10-19-compute-projection.json)
keeps measured unit times, rough proxies, and missing timings separate.

The Mac measurements show that evaluation determines much of the cost.
The proposed schedule below preserves three seeds, both fine-tuning
conditions, all 101 classes, and the 505-video final test set. It reduces
how often the expensive primary attack is run. That changes the temporal
resolution and the meaning of the selected checkpoint, so it must be
reviewed and frozen before training, not chosen after seeing curves.

## Proposed measurements

| Stage | Clips | Frequency/checkpoints | Attack and purpose |
| --- | --- | --- | --- |
| Routine monitoring | Fixed 101 train + 101 validation; one per class in each split | Initialization once, then every epoch of six runs | Clean loss and PGD-5, one restart, one attack-mask draw, one independent scoring draw; inexpensive trajectory diagnostics |
| Primary trajectory | The same 101 train + 101 validation clips | Initialization once; epochs **1, 5, 10, 15, 16, 20, 23, 24, 30** of every run | Clean loss and PGD-20, two restarts, four attack-mask draws, four independent scoring draws |
| Checkpoint selection | Primary 101-video validation subset | Only the declared primary checkpoints, including initialization | Lowest primary adversarial native validation loss; earliest epoch breaks ties |
| Attack checks and baselines | Fixed 101 validation clips, one per class | Pretrained plus selected-best/final for each run; at most 13 distinct checkpoints | PGD-50/EOT-8, epsilon sweep, noise, FGSM, embedding attacks, and probe-attack checks |
| Final native/probe evaluation | All 505 test clips | The same at-most-13 checkpoints, after all decisions are locked | Primary native attack, clean and native-transfer probe accuracy, direct classifier attacks |

An upstream mask draw contains two mask families. Counts above refer to
draws, not individual masked regions. Epsilon is 4/255 in raw pixel space
for routine/primary attacks; proposed step size is 1/255. The primary mask
and restart settings stay stronger than the routine diagnostic. Cheap
PGD-5 loss never selects the reported best checkpoint.

Epochs 15/16 and 23/24 bracket the draft learning-rate changes. The other
anchors give early, intermediate, and final observations. This is a planning
choice motivated by the schedule, not evidence that a JEPA effect must occur
there. The robust-overfitting literature motivates examining training
dynamics and early selection, but does not validate these JEPA settings.
[Rice, Wong, and Kolter (2020)](https://proceedings.mlr.press/v119/rice20a.html).

Report the selected state as **best among the scheduled primary checkpoints**.
Do not claim it is the best of all 30 epochs, or that the sparse strong curve
rules out deterioration between anchors. Retain every epoch's weights.
A later expansion to more epochs is a separately labeled analysis with its
own cost; it cannot retroactively become a preregistered choice.

The denser alternative evaluates the primary attack on the same 101
validation clips every epoch. It permits primary-metric selection over all
epochs, with stronger temporal evidence, but has a higher projected cost.
The calculator includes that alternative so the tradeoff can be reviewed.

## Sampling, randomness, and interpretation

Before training, choose one clip per class from each existing train/validation
manifest by the smallest SHA-256 of `20261019|split|video_id`, using video ID
as a tie breaker. Save the selected rows, original manifest hashes, and
selection rule. No new manifest has been created yet. Source-group separation
comes from the existing manifests; don't move clips between splits.

If the declared 10-training-videos-per-class fallback is used, take the
first ten rows per class under that same training hash order. The monitored
training clip is then guaranteed to remain in the actual training set.
Create a new nested training manifest and hash it; validation/test remain
unchanged. Do this before confirmatory training.

Fix attack/scoring mask banks and random-start seeds per clip and evaluation
mode before training, with disjoint namespaces for attack versus scoring.
Use the same banks across conditions, seeds, and checkpoints. Recompute
adversaries against each checkpoint; never reuse an earlier checkpoint's
perturbation. Record bank definitions and sampled masks in provenance.

One clip per class gives class coverage, not strong within-class coverage or
guaranteed statistical power. Show per-seed curves and group-level uncertainty
on paired gaps. Strong train and validation measurements must support the
interpretation, including continued training-loss improvement and held-out
deterioration. Cheap-monitor behavior alone is insufficient. Failure to
observe an effect in this regime is not evidence of immunity.

Check zero-epsilon identity, finite gradients, perturbation bounds, restarts,
step sizes, and independent-mask behavior. On 101 training clips at the
pretrained checkpoint, compare primary attack step sizes 0.5/255, 1/255,
and 2/255 before freezing the choice. Recheck attack strength on selected
validation checkpoints before final testing. Different EOT banks and
independent scoring can prevent monotonic score changes; inspect paired
results instead of treating one attack failure as robustness. Gradient-based
evaluation needs checks against misleading gradients.
[Athalye, Carlini, and Wagner (2018)](https://arxiv.org/abs/1802.00420).

If stronger attacks materially change the conclusion, pause and budget the
needed evaluation. Don't weaken evaluation to fit a favorable result or
open the test set to resolve tuning choices.

## Baselines, probes, and reuse

The native epsilon sweep is 0, 1, 2, 4, and 8/255 on the fixed validation
subset at the selected checkpoints. Reuse the matching clean and epsilon-4
primary scores; the calculator budgets three additional nonzero attacks.
PGD-50 uses two restarts and eight attack-mask draws, with the same independent
four-draw scoring bank used for primary comparison.

Random noise and EOT FGSM use the same 4/255 bound. The proposed embedding
baseline is random-start PGD-20/two restarts against squared distance between
normalized pooled encoder embeddings of clean and perturbed full clips;
it is scored with the independent native-loss bank. This baseline is not
implemented or timed yet. Final objective/step details need a recorded
implementation check before the protocol is frozen.

For every selected representation, fit the same linear-probe procedure using
training features only. Reserve four hyperparameter candidates per checkpoint,
chosen using validation data; freeze the fitting procedure before testing.
This is **52 candidate fits** for 13 distinct checkpoints, not one probe
reused after the encoder changes. Optimizer, normalization, candidate grid,
and fitting runtime remain implementation decisions.

Fit the matched probe before generating each checkpoint's final native
adversaries. Score each adversary with both native loss and that probe in the
same evaluation pass, so native-attack transfer does not require generating
the attack twice. Direct classifier attacks remain separate: PGD-20/two
restarts at 4/255, with PGD-50/step-size checks on validation before testing.
Native attack transfer alone does not establish classifier robustness.

## What the measured timings imply

All rows below are projections from the short FP32 Mac pilot, not completed
research work. They assume eight-clip updates and cached clean targets.

| Work with an existing timing anchor | Count | Projected Mac hours |
| --- | ---: | ---: |
| Clean training | 22,770 updates | 57.51 |
| Adversarial training | 22,770 updates | 312.02 |
| Routine PGD-5 monitoring | 36,562 clips | 49.20 |
| Primary train/validation evaluations | 11,110 clips | 497.16 |
| Primary final test evaluations | Up to 6,565 clips | 293.77 |
| **Subtotal from timed units** | | **1,209.66 hours / 50.40 days** |

The estimate includes one shared pretrained reference and assumes all
best/final checkpoints are distinct for budgeting. Deduplicate by exact
checkpoint identity if some coincide; don't assume that saving in advance.

An additional **483.60 hours** comes from rough proxies for step calibration,
PGD-50/EOT-8, and the extra epsilon attacks. In particular, scaling gradient
work by five is not a measured PGD-50 runtime. The resulting **70.55-day
partial projection is still not an end-to-end estimate**. Fifteen components
have no timing yet, including clean scoring, baselines, preprocessing/caching,
probe extraction/fitting/attacks, checkpoint I/O, and billed setup/transfer.
Their entries remain `null`, never zero.

The predeclared fallback to 10 training videos per class saves about 184
training hours, but leaves validation/test work and the three seeds intact:
its partial projection is still 62.88 days. Dense primary validation every
epoch increases the partial projection to 94.39 days. Neither is a measured
cloud runtime or an admitted Mac schedule.

Reproduce the calculation without launching a model or contacting a provider:

```bash
python scripts/estimate_compute.py \
  --output runs/pretrained/diagnostic/compute-plan-01/estimate.json
```

## Cloud prices and a bounded pilot

Public prices checked October 19, 2026:

| Candidate | Listed GPU price/hour | Base workload hours allowed by the draft gate |
| --- | ---: | ---: |
| Lambda A6000, 48 GB | $1.09 | 110.09 |
| Lambda A100 PCIe, 40 GB | $1.99 | 60.30 |
| Lambda H100 PCIe, 80 GB | $3.29 | 36.47 |
| Runpod A6000, 48 GB, advertised Pod rate | $0.59 | 203.39 |

Sources: [Lambda's single-GPU catalog](https://lambda.ai/instances) and
[Runpod's pricing page](https://www.runpod.io/pricing). These are public
listings, not reserved capacity or account quotes. The Runpod page includes
cloud-tier selectors; the tier and deployable price have not been verified.
Confirm the actual tier, storage charges, taxes, region, and availability
before using any row to choose an instance. An hourly rate does not predict
the cost of this workload.

The calculation reserves **$10 within the $160 compute envelope** for pilots,
leaving $150, and applies 25% runtime headroom to the eventual measured full
workload. The **$40 noncompute reserve** remains for storage, taxes, and
contingency. For example, Lambda A6000 permits `150 / (1.09 × 1.25)` base
hours. Spending and export/idle time must count; the $10 is not extra money.

For the already-timed subtotal alone, the Lambda A6000 would need an average
11.0-times speedup over this Mac sample, or 15.4 times after including the
rough proxies. At Runpod's advertised rate, the corresponding requirements
are 5.9 and 8.3 times. **These are algebraic requirements, not predicted or
observed GPU speedups**, and missing components make the requirement stricter.
Do not assume the $200 cap is feasible from a GPU name or FLOPS specification.

Kaiwen confirmed an existing Lambda account on October 19. Use a single
Lambda A6000 as the first candidate, subject to its live price and
availability. The Runpod row is a comparison, not the selected provider.
Prepare the complete pilot bundle locally. Proposed limit: **one GPU, at
most two billed hours**, within the shared $10 pilot allowance. At Lambda's
listed price that is $2.18 compute before other charges. The initial pilot should
answer correctness, native update throughput, primary attack time, and memory.
Only then consider a second candidate or additional timing modes within the
remaining pilot allowance. A reminder or stopped Python process is not a
provider-enforced spend limit.

Lambda bills running instances even when idle, in one-minute increments,
until termination. Persistent filesystem billing continues separately, and
its actual storage rate is shown during creation; the documentation's
example storage price is not a current quote.
[Lambda billing documentation](https://docs.lambda.ai/public-cloud/billing/).
Use the chosen provider's actual export/termination procedure and verify its
billing state; do not copy provider-specific commands to a different service.

## Timing gaps and the admission decision

The existing CUDA pilot implements clean/adversarial updates and primary
PGD-20 timing. It does not yet cover the entire ledger. Before a full run:

1. Use the [initial pilot handoff](cloud_pilot_handoff.md) and prepare the exact
   code, environment recipe, checkpoint, and training-only clip bundle before
   billable setup. The first two-hour pilot uses the existing native checks;
   it does not need the future full training runner. If its already-timed
   components fail the cost gate, stop and revise before more spending.
2. On the candidate GPU, pass pretrained loss/gradient/attack checks, then
   measure updates and all required evaluation modes with synchronized timing.
   Start with FP32. Mixed precision or another projection needs its own
   correctness checks and timing; no speedup from either is assumed here.
3. Finish remaining local baseline/probe/timing interfaces and resume tests
   before a subsequent full-admission timing pass. Replace proxy and missing
   rows with that device's measurements. Include
   setup, preprocessing, feature fitting, checkpoint I/O, exports, and idle
   time. Measure real storage sizes and choose a verified backup destination.
4. Require `spent compute + remaining pilot allowance + 1.25 × measured
   remaining GPU hours × actual hourly rate <= $160`, with storage/tax/etc.
   fitting the separate $40 and total <= $200. Also fit the calendar plan,
   leaving time for implementation, analysis, and human review.
5. If the 20-video training subset fails, price the declared 10-video fallback.
   If it still fails, revise scope or budget explicitly before confirmatory
   work. Preserve the test set and seeds while comparing those two options.
6. Review and freeze the evaluation schedule, subset hashes, seeds, attack
   parameters, precision, optimizer, and backend. Recheck novelty before the
   paid experiment as required by the existing research plan.

No instance, filesystem, cloud account, or purchase was created here.
`budget_approved` remains false in every estimate. Account access will be
needed when the local pilot bundle is ready; no credentials belong in Git.

## Storage planning

Saving FP32 context/predictor weights for all 180 epochs is about **218.58 GiB**
of tensor data. Save the frozen target once via its pinned source checkpoint,
not again inside every epoch file. Two rolling full resume slots add about
**7.29 GiB**, assuming one sequential run and AdamW's two FP32 moments.
Processed FP32 clips are about **35.51 GiB** and fixed targets **23.67 GiB**
for all 3,030 videos. Test caches are produced only after choices are locked.

These are tensor-size calculations, not observed serialized file sizes.
Budget additional space for raw videos, the 6.46 GiB archive, the 4.78 GiB
official checkpoint, software, metadata, atomic temporary writes, and exports.
A 512 GiB instance disk is a candidate, not a verified free-space guarantee.
The Mac had about 591 GiB free at this check, so a local backup appears
possible; verify the destination and transferred hashes before termination.
Keep every-epoch weights, the current/best resumable state, metrics, and
provenance; a sparse evaluation schedule must not silently discard weights.
