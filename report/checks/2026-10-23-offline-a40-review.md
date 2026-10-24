# Offline A40 price/performance and protocol review

October 23. This review uses the completed, copied A40 evidence. It verifies all
95 export members against the pinned archive and rechecks
198 existing non-Markdown source pins. It opens no pixels, launches no
cloud job and changes no study configuration or scientific threshold.
The 49 selected tests pass with zero skips/errors/failures, including 26 new
analysis checks. The expanded full project suite was not rerun.

## Cost is dominated by evaluation

The original 20-class, 100-training-clip, eight-run workload, all attacks and
schedules are retained. The review independently reproduces the combined screen:
**$221.56 median units / $224.01 maximum observed units**
at the historical $0.61/hour quote. Neither is a full-study bound or admission.

| Component, median units | Timed hours | Share of timed work |
| --- | ---: | ---: |
| Primary evaluation, including final test | 135.847 | 62.61% |
| Convergence checks with both audits | 30.693 | 14.14% |
| Routine evaluation | 18.537 | 8.54% |
| Adversarial updates | 23.950 | 11.04% |
| Clean updates | 4.161 | 1.92% |
| Checkpoint state and save | 1.692 | 0.78% |
| Three-candidate research calibration bundles | 1.656 | 0.76% |

Native routine/primary/convergence/FGSM/random evaluation is
85.49% of timed work; calibration is separate.
All native times already include their declared restarts, EOT, clean scoring,
and applicable audit banks. No factor is applied to them a second time.
Checkpoint state and save are disjoint spans; outer workflow spans are excluded.

At $0.61/hour, the screen requires **13.03–
14.30% less native runtime**, equivalent to about
14.98–
16.69% higher throughput under a uniform-scaling assumption.
A 15% runtime reduction would place the maximum-unit screen below $200;
a 10% reduction would not. These are affordability targets, not validated speedups.

## Price sensitivity is conditional on performance

No new price or availability was checked. Every row below is hypothetical.
The third column is the largest uniform native-runtime multiplier relative to
the measured A40 that would just meet the partial $200 screen at that rate.
A different GPU must be measured; A40 throughput is not assigned to it.

| Hypothetical all-in rate | Maximum-unit screen at unchanged A40 speed | Maximum uniform native-time multiplier |
| --- | ---: | ---: |
| $0.29 | $132.73 | 1.843× |
| $0.40 | $164.11 | 1.326× |
| $0.50 | $192.63 | 1.054× |
| $0.52 | $198.34 | 1.012× |
| $0.53 | $201.19 | 0.992× |
| $0.61 | $224.01 | 0.857× |

The binding same-performance rate is about $0.526/hour.
At $0.52/hour the partial screen is about $198.34, leaving little room beyond the
retained allowances. A low price alone cannot establish feasibility.

The eight-hour allowance, 25% runtime headroom, $10 diagnostic envelope and $40
noncompute reserve are unchanged. The existing diagnostic charges belong inside
the $10 envelope; adding them again to the $50 reserves would double count them.
The hypothetical prices and time scalings do not change the study or admit it.
Representative throughput, complete feature/probe work, I/O/storage/repair costs
and exact recovery still need validation.

## The attack gap does not isolate iteration count

On the same training-monitor clip and main independent scoring bank, the saved
primary loss is 0.660437 and the stronger setting gives
0.705661: a difference of
0.045224 (6.85%).
However, the comparison changes 20→50 steps, 4→8 attack-mask draws, the actual mask
bank and the random-start seed together. It therefore measures the combined
configuration contrast. It cannot identify how much of the gap comes from steps,
mask sampling or random starts, or establish study-wide attack convergence.

The additional scoring banks also have different clean baselines. Their paired
adversarial-minus-clean differences are
0.136675,
0.246014, and
0.154347. These describe this
clip; their spread is not a diagnosis of mask overfitting or gradient masking.
No bank or perturbation is selected using these scoring losses.

## Next concrete control

Prepare the [matched training-only diagnostic](../../docs/matched_attack_diagnostic_plan.md):
20/50 steps crossed with 4/8 nested attack-mask draws, identical starts per
restart, and the same independent main/audit scoring banks for every arm.
Use a separate diagnostic artifact contract; keep the canonical study evaluation
methods guarded and unchanged. Prefer a bounded free university GPU check if
available before another paid hardware benchmark. This diagnostic is specified,
**not implemented or executed** by this review.

After that evidence, review the primary attack and mask-bank choices, seek a
validated cost path, check exact optimizer recovery on the chosen GPU, finish
research calibration/data/novelty/protocol review, and only then consider the
exploratory pair. No attack, seed, class count, epoch schedule or study budget
was weakened. Confirmation/test pixels remain closed. No new rental is authorized.

The latest billing snapshot remains $6.10 provisional wallet-implied use, with an
unassigned $0.0792 difference from the Details rows. Keep the current inclusive
$2 hold / $2.53 conservatively unreserved allowance pending settlement. This small
accounting difference does not prevent the offline work described here.
