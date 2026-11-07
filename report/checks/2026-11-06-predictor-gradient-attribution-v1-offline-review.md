# Predictor gradient attribution prepared offline — November 6

**29 distinct local checks pass**, with zero errors, failures or skips. Whole-
tensor saved analysis and a small native CPU observer proof are independently
reviewed. The cause of the rejected pretrained input gradient remains unknown;
the candidate remains rejected under the original bounds. No new university
operator, handoff, authorization, attempt, SSH connection, upload or paid work
was created. Both university hosts may remain idle.

## Complete saved-gradient analysis

The analysis revalidates all 7,505 sealed predecessor inputs and all four saved
loss/input-gradient artifacts. Every one of 3,145,728 gradient elements enters
the original elementwise comparison, all 48 channel/frame groups and fixed
reference-magnitude strata. NumPy independently reproduces those groups and
counts, with model/autograd/SDPA/CUDA, process and network creation blocked.

The two scalar losses remain exactly equal. There are 3,724 violations of the
original input bound and 58 opposite nonzero pixel-gradient signs. Three sign
changes are beyond the bound; the other 55 are inside its absolute allowance.
These are **pixel-gradient** signs, not the missing predictor-residual signs.
They identify neither an L1 sign-crossing cause nor attack validity. Magnitude
error remains the basis of rejection; sign or aggregate statistics waive no gate.

| Reference-gradient magnitude | Elements | Bound violations | Sign changes |
| --- | ---: | ---: | ---: |
| [0, 1e-7) | 501,929 | 3 | 54 |
| [1e-7, 1e-6) | 92,303 | 33 | 4 |
| [1e-6, 1e-5) | 548,939 | 436 | 0 |
| [1e-5, 1e-4) | 1,047,229 | 2,063 | 0 |
| [1e-4, 1e-3) | 635,854 | 1,186 | 0 |
| [1e-3, 1e-2) | 307,390 | 3 | 0 |
| [1e-2, 1e-1) | 11,658 | 0 | 0 |
| [1e-1, infinity) | 426 | 0 | 0 |

Violations occur across all 16 frames and all three channels (1,360 / 1,253 /
1,111). The full channel/frame table is preserved in the JSON report. This is
one clip and one refused candidate: spatial/magnitude patterns do not identify
a backend defect, accumulation mechanism or harmless rounding explanation.

## Missing observation and local prototype

The original loss averages L1 over prediction tokens/features, then two mask
families and eight draws. Equal reduced FP32 loss can hide different residual
signs and output cotangents. Conversely, identical output cotangents do not
imply identical pixels-to-predictions Jacobians. Both possibilities have tested
synthetic counterexamples; they are not assertions about this target run.

A separately named observer prototype uses non-replacing module/tensor hooks
on the existing small upstream CPU model. It copies sixteen actual predictions
and their actual incoming cotangents, returns None from hooks and removes all
observer handles on success, incomplete work and primary exceptions. No surrogate
objective or backend rewrite is introduced. It refuses university nodes and
non-tiny/non-CPU domains.

Three real native CPU passes before/during/after observation preserve all six
loss/input-gradient returns exactly, also matching the frozen older CPU baseline.
The saved masks and clean target are byte-identical to that reference. Independent
NumPy checking reconstructs each selected target and all sixteen actual FP32
cotangents at the original eight-draw/two-family L1 weighting. Model values and
tensor versions remain unchanged, parameter gradients stay absent, and native
graph scopes complete with their hooks removed. This is arm64 / Python 3.11.16 /
Torch 2.14.1 evidence; it does not qualify the pretrained CUDA observer.

The initial local suite recorded 23 passes and five setup errors from calling a
pytest fixture directly. After correcting that test setup, 28 checks pass. An
additional mask/target provenance check and saved proof produce the final 29.
Earlier receipts are retained, not added to the final count. The exact earlier
test source referenced by the initial analysis is preserved separately by hash;
the final analysis binds the final sources. No old target or historical source
was edited. No unrelated test suites were repeated.

## Concrete, inactive next proposal

The [proposal and interpretation limits](../../docs/predictor_gradient_attribution_v1.md)
call for two fresh serial one-phase arms on linux10603: original and predictor-
math, observing every native prediction and cotangent under the original full
EOT8 objective. The original observer must reproduce the passing loss/gradient
exactly. The candidate observer must reproduce the preserved failed values
exactly before attributing that discrepancy; a known refusal is not promoted
to a complete passing reference. Original numerical checks still run, and any
candidate rejection remains a rejection. Missing/mismatched pairs refuse.

A proposed twenty-minute work limit covers full 2.5 GiB proof/setup and two
workers of at most seven minutes, followed by at most ten minutes / 3 GiB of
same-SSH collection. Setup/capture remain 3 GiB / 2.25 GiB, allocator 7 GiB,
metadata 64 MiB and proposed raw capture 512 MiB. Static logical data plus
serialization/atomic-copy allowances are 424,935,424 bytes below that raw cap.
This is arithmetic, not a reserved volume or demonstrated live peak. All live
storage, free-space/inode, process, typed-file and continuing-time guards must
remain. No retries, old-claim/time reset, cleanup, private keys, held-out pixels,
parallel target host or paid compute are proposed.

The observation could establish whether the external L1 cotangent changed.
It cannot alone isolate pure backward arithmetic: Jacobian-path changes also
include internal forward rounding. A tested saved-cell helper retains both
orders and the interaction of the four-cell VJP decomposition; the target has
no cross-cells, and this proposal does not calculate them. A later same-input,
same-cotangent attention experiment may be needed after reviewing these data.
No candidate CUDA backend proof, performance/cost result or study adoption is
inferred from an observer or selected flags.

Independent saved review rechecks 7,505 prior pins and seals 7,526 inputs with
scientific dispatch/process/network work blocked. All financial caps and holds
remain unchanged; reconciliation stays incomplete. Known Runpod Pods remain
terminated. Next prepare and test a separately bounded live observer and handoff,
review it independently, then request one new exact scoped target approval.
There is no new runnable target helper or resource authorization at this point.
