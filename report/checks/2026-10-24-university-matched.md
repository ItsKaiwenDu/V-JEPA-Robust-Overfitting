# Verified university matched attack diagnostic

October 24, 2026. The one-clip, released-checkpoint four-arm diagnostic and its
fresh A-only replay **pass independent local verification**. It ran on
`linux10609`, RTX 3070, within the existing shared-fp64-native diagnostic policy.
This is an input-gradient/attack diagnostic, not fine-tuning or a research result.

The Python session took **40 minutes 57 seconds**, including worker setup.
The launcher log spans **11:53:04 AM–12:34:10 PM Pacific** (41m06s), including
export/final checks, within the 80-minute outer limit. Matrix peak CUDA allocator
use was **4.89 GiB**; fresh A-only replay peaked at **3.69 GiB**. The recorded
allocator bound was 6.88 GiB, below the configured 7 GiB ceiling. These are
allocator observations, not total driver/device peaks. This demonstrates fit for
this pretrained attack diagnostic only. The earlier 8 GB AdamW OOM is preserved;
full optimizer/update/recovery fit is not established.

## Evidence verification

The archive is **110,384,818 bytes**, SHA-256
`3e9d17e10c9cdd1fa814ad2ccdbceeef71921158c3b34634023a79e145352f2b`.
All **29** regular members, archived hashes and extracted bytes match the
externally recorded checksum. The handoff manifest matches the accepted local
snapshot. All 198 original non-Markdown pins, five retained new source files,
49 bound project files and 34 upstream files match.

All **173 logical tensors** verify. The exact first declared training-monitor
pixels match the original prepared file; all **80 mask tensors** reproduce from
the 20 declared upstream draws. Shared random starts, independent scoring-bank
seeds, both first-20-step prefixes and endpoints, best attack-bank selection,
finite nonzero input gradients, bounds and model/target preservation checks pass.
All **320 input-gradient observations** are finite/nonzero; prefixes and the
fresh replay match exactly. All three component logical weight digests reproduce
independently from the rehashed original released checkpoint, without another
model forward. Saved target identities match across arms and fresh replay.

Both retained native CUDA backward probes revalidate from their profiler events.
Runtime, strict precision/backend, geometry, normalization and activation
checkpointing match the declared contract. The fresh process replays **A only**,
including its trace, gradients, selected pixels and scores; no all-arm fresh
replay is claimed. Both workers exit zero and the parent reports complete/pass.

## One-clip results

All arms use radius 4/255, step 2/255, two identical starts and fixed clean
targets. Four-draw attack banks are the first half of the same eight-draw bank.
Each independent scoring bank has four draws. Higher independent native loss
means a stronger perturbation on that particular bank; no scoring result selects
a candidate, restart or arm. The clip is `v_BreastStroke_g10_c02`, actual training.

| Arm | Steps | Attack draws | Main loss | Audit 0 | Audit 1 | Instrumented attack seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 20 | 4 | 0.699150 | 0.712586 | 0.704787 | 200.428 |
| B | 50 | 4 | 0.699278 | 0.710885 | 0.706450 | 497.217 |
| C | 20 | 8 | 0.680543 | 0.697317 | 0.695989 | 405.124 |
| D | 50 | 8 | 0.692156 | 0.709003 | 0.709850 | 1003.526 |

The same clean baselines for all arms are **0.562045** (main), **0.553706**
(audit 0) and **0.554316** (audit 1). Attack-bank best losses rise from 0.749822
to 0.764134 at four draws and 0.709975 to 0.727069 at eight draws. Those
four-/eight-draw objective values average different masks and are not a direct
cross-bank strength ranking.

| Paired contrast | Main difference | Audit 0 difference | Audit 1 difference |
| --- | ---: | ---: | ---: |
| 50 minus 20 steps, four draws | +0.000128 | -0.001701 | +0.001664 |
| 50 minus 20 steps, eight draws | +0.011614 | +0.011686 | +0.013861 |
| Eight minus four draws, 20 steps | -0.018607 | -0.015269 | -0.008798 |
| Eight minus four draws, 50 steps | -0.007121 | -0.001882 | +0.003400 |

At four draws, extra steps change the main loss by only **+0.0183%**, with
opposite signs on the two audits. This is descriptive, not a convergence pass.
At eight draws, extra steps increase loss on all three banks by approximately
**1.71%, 1.68% and 1.99%**. More iterations matter for that eight-draw attack on
this clip. Eight draws do not uniformly outperform four: at 20 steps all three
scores are lower, and at 50 steps the direction differs by scoring bank.

A larger mean-objective mask bank changes the optimization problem; it need
not dominate a smaller bank after a finite PGD run. The result does not diagnose
mask overfitting or gradient masking, establish study-wide convergence, or prove
that 50 steps are sufficient. The earlier A40 comparison used different starts
and bank draws; these matched contrasts do not retroactively isolate its gap.
Do not choose the final primary method or reduce evaluation strength from one
training clip and three scoring banks.

## Preserved limits and next step

Canonical EvaluationSettings, PGD, study/proposal YAML, 20 classes, all study
seeds and held-out closure remain unchanged. The one-clip diagnostic is not a
research calibration artifact or a robust-overfitting finding. Original FP32
rejections remain rejected. The earlier 91 focused/26 new and 26 extracted CPU
checks are separate evidence; no pytest suite was rerun or relabeled as CUDA
passes during this audit.

Times include tracing overhead and belong to this RTX 3070 session. They are
excluded from A40 economic pooling and imply no cross-GPU speedup. No paid Pod
was launched, hold released or bill reconciled here. The prior 20-class
$221.56–$224.01 screen still exceeds the $200 cap; study admission is pending.
The $2 hold, conservative $2.53 unreserved diagnostic allowance and unassigned
$0.0792 billing gap remain unchanged.

Next is **offline predeclaration of a broader training-only multi-clip and
mask-bank check**, followed by research calibration, evaluation/protocol and
budget review. Use the already declared training-monitor panel; freeze its
ordered clips, bank seeds, decision criteria and execution bounds before any
new run. No further university command or cloud deployment is required now.

Raw archive, manifest, logs, tensors, audit script and verification receipt are
retained under `runs/pretrained/diagnostic/university-matched-session-20261024`.
The companion JSON binds the archive, original sources, contract, weights,
verification script and receipt. Reproduction requires those ignored local
artifacts and the existing released checkpoint; the audit performs no model
forward or additional GPU run.
