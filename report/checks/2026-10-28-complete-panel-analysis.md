# Complete diagnostic panel: attack sensitivity remains

All20 originally reviewed and registered cases have been reverified from their
saved evidence. The frozen descriptive analysis is complete, using **ten clip
units and two technical attack-bank replicates per clip**. The main and both
audit scoring banks remain separate. The independent exact-rational arithmetic
audit agrees with every case contrast, clip mean and summary row.

## Findings

Increasing attack-mask draws4→8 at20steps raises independent-scoring native L1
loss by **3.54–3.95%** across the three separate scoring banks. Each bank has
a positive paired change on **all10 clips**. At50steps, the corresponding
increase is **3.84–4.15%**, again positive on all10 clips in each bank.

Increasing steps20→50 at4draws raises loss by **1.53–2.03%**, with positive
paired changes on all10 clips in each scoring bank. At8draws, its mean increase
is **1.94–2.49%**; main has10positive clips and each audit has9. Thus both
factors matter within this controlled diagnostic, with larger mean relative
changes from added mask draws in these tested contrasts.

| Controlled contrast | main | audit_0 | audit_1 | Positive clips (main / audit_0 / audit_1) |
| --- | ---: | ---: | ---: | --- |
| B-A: 20→50 steps, four draws | 1.9907% | 1.5289% | 2.0321% | 10 / 10 / 10 |
| D-C: 20→50 steps, eight draws | 2.4871% | 2.0176% | 1.9432% | 10 / 9 / 9 |
| C-A: 4→8 draws,20 steps | 3.6658% | 3.5414% | 3.9474% | 10 / 10 / 10 |
| D-B: 4→8 draws,50 steps | 4.1495% | 4.0436% | 3.8391% | 10 / 10 / 10 |

Percentages are the mean of case-level relative gaps to the lower-setting
adversarial loss, first averaged over two technical banks within each clip,
then averaged equally across ten clips. They are not relative changes in
classification accuracy or ratios of pooled loss means. The range across banks
above is descriptive; the banks are not pooled into a single estimate.

The one cross-bank sign exception is **TennisSwing** for D-C. Its mean absolute
change is+0.00009251on main, -0.01326537on audit_0 and-0.00596023on audit_1.
Retain these negative outcomes. Independent-scoring loss is not required to
increase monotonically with attack steps; adversary selection uses only the
attack bank. This exception alone does not establish gradient masking or mask
overfitting. The other nine clips are positive in all three banks for D-C.

## Interpretation and admission review

The tested20-step settings have additional independent-scoring loss under the
matched50-step controls. Added mask draws also improve the mean scores across
both step counts. These observations do **not establish attack convergence**.
No numeric convergence margin was declared or introduced, and the strongest
tested configuration has no stronger-setting comparison in this panel.

Do not automatically select an arm, weaken evaluation or adopt a research
configuration from this table. The canonical study settings remain unchanged.
The next work is an explicit attack/mask-policy and training-only step-size
calibration decision, followed by numerical-policy/selected-GPU optimizer
qualification and measured budget review. The old October23 A40 screen at its
then$0.61/hour quote was$221.56–$224.01with further qualifications still needed;
this panel adds no new price, A40 timing bound or affordability admission.
The$200 total/$160 planned-compute caps remain unchanged.

These observations concern attacks on the **released pretrained checkpoint**
and the first ten fixed training-monitor clips. The first clip was previously
examined; current technical seeds are new and the earlier one-clip result is
not pooled into this panel. The analysis provides no training trajectory,
held-out result, statistical significance claim, or conclusion that robust
overfitting occurs during V-JEPA fine-tuning. Research calibration, protocol
freeze and the main clean/adversarial experiments remain separate gates.

## Evidence and reproducibility

The new read-only loader uses the actual20-case registration receipt as its
authority, preserves every original execution/attempt origin, and verifies
each registered receipt, review marker, execution record, original archive,
source hashes, saved native CUDA evidence, finite gradients, pixel bounds and
exact fresh A replay through the frozen verifier. Shared targets, starts,
runtime, weights, settings, scoring mask hashes and clean scoring baselines
match within all ten technical pairs. All739 input fingerprints match again
after analysis and during the independent arithmetic audit. The seven original
frozen-v1 cases keep their original seven-case bootstrap review identity.

All **41 focused tests pass**, zero failures/errors/skips, including independent
fraction arithmetic, separate scoring banks, zero-denominator handling,
incomplete collection refusal, paired identity drift, original mixed execution
IDs, wrong registration authority, path redirection, and the declared split
source layout. The initial40-test result preceded an integration refusal in
the new loader: it assumed every scientific dependency was repeated inside
each case export. The corrected loader explicitly verifies the seven runner
files in each case and the other dependencies in the shared frozen snapshot;
it has no existence-based fallback. No evidence or scientific tolerance changed,
and the refused integration pass produced no aggregate.

The independent arithmetic audit imports no production aggregation code. It
uses exact fractions of original saved arm-record floats to check480case-level
contrast values,240clip-level values and12separate summary rows. Its1e-12
comparison allowance covers arithmetic rounding only and creates no scientific
strength, replay or convergence tolerance. No model forward, GPU job, new
decode, held-out access, remote connection or paid compute runs in this task.
Original journals, reviews, failed evidence, clocks and study/configuration
files remain unchanged. User commits separately.

All clean/adversarial arm losses, case contrasts, clip means, medians, ranges,
sign counts and relative gaps are retained in the
[structured analysis](2026-10-28-complete-panel-analysis.json).
The [explicit input manifest](2026-10-28-complete-panel-inputs.json) records all
twenty original paths and registered identities. The raw analysis,739-file
fingerprint inventory, independent audit source/output and test proof are in
`runs/pretrained/diagnostic/complete-panel-analysis-20261028/`.

## Full predeclared descriptive tables

Positive contrasts mean the higher-setting attack produced greater independent-scoring native L1 loss.
Relative values average case-level percentages within clip, then average ten clips; they are not ratios of pooled means.

| Contrast | Scoring bank | Mean L1 difference | Mean relative % | Median | Min | Max | Positive / zero / negative clips |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| B-A | main | 0.01326280 | 1.9907 | 0.01053372 | 0.00040409 | 0.03152972 | 10 / 0 / 0 |
| B-A | audit_0 | 0.01027997 | 1.5289 | 0.00527586 | 0.00006074 | 0.03100333 | 10 / 0 / 0 |
| B-A | audit_1 | 0.01361707 | 2.0321 | 0.00911105 | 0.00157437 | 0.04058149 | 10 / 0 / 0 |
| D-C | main | 0.01683936 | 2.4871 | 0.01709466 | 0.00009251 | 0.03126168 | 10 / 0 / 0 |
| D-C | audit_0 | 0.01325450 | 2.0176 | 0.01442873 | -0.01326537 | 0.02703059 | 9 / 0 / 1 |
| D-C | audit_1 | 0.01308549 | 1.9432 | 0.01201493 | -0.00596023 | 0.02630007 | 9 / 0 / 1 |
| C-A | main | 0.02406168 | 3.6658 | 0.01790285 | 0.01102394 | 0.06093457 | 10 / 0 / 0 |
| C-A | audit_0 | 0.02312513 | 3.5414 | 0.01792315 | 0.01329345 | 0.04568720 | 10 / 0 / 0 |
| C-A | audit_1 | 0.02591547 | 3.9474 | 0.01996173 | 0.00987142 | 0.07705095 | 10 / 0 / 0 |
| D-B | main | 0.02763824 | 4.1495 | 0.02525154 | 0.01865819 | 0.04818687 | 10 / 0 / 0 |
| D-B | audit_0 | 0.02609966 | 4.0436 | 0.02599101 | 0.00141850 | 0.06397885 | 10 / 0 / 0 |
| D-B | audit_1 | 0.02538390 | 3.8391 | 0.02481252 | 0.00581264 | 0.04987183 | 10 / 0 / 0 |

## Per-clip absolute differences: main

| Clip | B-A | D-C | C-A | D-B |
| --- | ---: | ---: | ---: | ---: |
| v_BreastStroke_g10_c02 | 0.02842271 | 0.02935910 | 0.02341247 | 0.02434886 |
| v_JumpRope_g16_c02 | 0.01073560 | 0.01646420 | 0.01292959 | 0.01865819 |
| v_Swing_g20_c03 | 0.01856950 | 0.02356330 | 0.04319307 | 0.04818687 |
| v_TennisSwing_g14_c04 | 0.03152972 | 0.00009251 | 0.06093457 | 0.02949736 |
| v_FloorGymnastics_g17_c01 | 0.00411397 | 0.00735509 | 0.02801132 | 0.03125244 |
| v_Biking_g16_c04 | 0.01033184 | 0.01772511 | 0.01631412 | 0.02370739 |
| v_Archery_g19_c04 | 0.00438237 | 0.01924446 | 0.01102394 | 0.02588603 |
| v_YoYo_g14_c03 | 0.00040409 | 0.01335123 | 0.01387987 | 0.02682701 |
| v_Bowling_g19_c04 | 0.00485146 | 0.00997692 | 0.01949158 | 0.02461705 |
| v_HandStandPushups_g21_c01 | 0.01928669 | 0.03126168 | 0.01142624 | 0.02340123 |

## Per-clip absolute differences: audit_0

| Clip | B-A | D-C | C-A | D-B |
| --- | ---: | ---: | ---: | ---: |
| v_BreastStroke_g10_c02 | 0.02403554 | 0.01556349 | 0.02776697 | 0.01929492 |
| v_JumpRope_g16_c02 | 0.00598541 | 0.01352012 | 0.01706681 | 0.02460152 |
| v_Swing_g20_c03 | 0.00457814 | 0.02703059 | 0.04152641 | 0.06397885 |
| v_TennisSwing_g14_c04 | 0.03100333 | -0.01326537 | 0.04568720 | 0.00141850 |
| v_FloorGymnastics_g17_c01 | 0.00077885 | 0.01237017 | 0.01464525 | 0.02623656 |
| v_Biking_g16_c04 | 0.00589025 | 0.01533735 | 0.02427873 | 0.03372583 |
| v_Archery_g19_c04 | 0.00187713 | 0.01691687 | 0.01329345 | 0.02833319 |
| v_YoYo_g14_c03 | 0.00006074 | 0.01246169 | 0.01362041 | 0.02602136 |
| v_Bowling_g19_c04 | 0.00466147 | 0.01184264 | 0.01877949 | 0.02596065 |
| v_HandStandPushups_g21_c01 | 0.02392879 | 0.02076745 | 0.01458657 | 0.01142523 |

## Per-clip absolute differences: audit_1

| Clip | B-A | D-C | C-A | D-B |
| --- | ---: | ---: | ---: | ---: |
| v_BreastStroke_g10_c02 | 0.03209710 | 0.01670158 | 0.02120817 | 0.00581264 |
| v_JumpRope_g16_c02 | 0.01036122 | 0.01850033 | 0.01411933 | 0.02225843 |
| v_Swing_g20_c03 | 0.01220378 | 0.02614048 | 0.03593513 | 0.04987183 |
| v_TennisSwing_g14_c04 | 0.04058149 | -0.00596023 | 0.07705095 | 0.03050923 |
| v_FloorGymnastics_g17_c01 | 0.00401399 | 0.00743780 | 0.02788064 | 0.03130445 |
| v_Biking_g16_c04 | 0.00515622 | 0.00798082 | 0.02084675 | 0.02367136 |
| v_Archery_g19_c04 | 0.00299022 | 0.01146835 | 0.01747555 | 0.02595368 |
| v_YoYo_g14_c03 | 0.00157437 | 0.01256150 | 0.01569009 | 0.02667722 |
| v_Bowling_g19_c04 | 0.00786087 | 0.00972423 | 0.01907670 | 0.02094007 |
| v_HandStandPushups_g21_c01 | 0.01933140 | 0.02630007 | 0.00987142 | 0.01684010 |

The structured JSON retains all case-level clean/adversarial losses, paired contrasts, relative gaps, exact original identities and evidence hashes.
This pretrained training-only diagnostic is separate from research step-size calibration, numerical-policy adoption, optimizer validation and budget admission.
It contains no training trajectory, held-out evaluation, statistical significance claim or robust-overfitting conclusion.
