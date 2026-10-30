# V-JEPA Robust Overfitting

**Fixed-input encoder review (October 30):**
[The 656-token encoder comparison passes; the 400-token comparison stops](report/checks/2026-10-30-fixed-attention-review.md).
Both runs load identical fixture bytes. Math attention takes 2.51 versus 4.02 ms
for the isolated 656-token component (1.60×), with higher allocation. The second
host rejects CPU FP64 reference bookkeeping before GPU math attention runs.
That discrepancy does not reproduce on the Mac; its cause remains unresolved.
Next prepare a CPU-only reference probe with saved intermediates. No backend,
full-model speedup, study cost, retry or research result is admitted.

**Immutable attention inputs prepared (October 30):**
[The new fixed-input diagnostic passes 46 focused CPU checks](report/checks/2026-10-30-fixed-attention-preparation.md).
Workers load verified NumPy arrays and fixed FP64 references, and retain input/
device-input/untimed-value evidence before acceptance checks. Two fresh full-size
loader processes and eight simulated transport scenarios pass. New unarmed
20-minute plans assign encoder-0 to linux10609 and encoder-1 to linux10603.
Prior attempts and generation failures remain preserved; no new GPU/SSH/paid
work, backend adoption or research admission has started. Earlier entries below
record historical preparation and outcomes.

**Attention component review: mixed result (October 30):**
[Both exports are verified; predictor saved-input checks pass](report/checks/2026-10-30-attention-component-review.md).
Math attention takes 12.12 ms versus 25.54 ms for the isolated predictor component,
with higher allocation (899 versus 75 MiB). The encoder stops on a changed Q
input signature before its final replay; its second shape never starts.
The original Mac input-regeneration gate also rejects, so input-generation
replay remains unqualified. A separate NumPy audit verifies saved-input values
without waiving that refusal. Next prepare immutable synthetic input fixtures;
no optimization, retry, paid work or research admission starts. Earlier entries
below record historical preparation and outcomes.

**FP32 attention component diagnostic prepared (October 30):**
[54 CPU checks and eight simulated transport scenarios pass](report/checks/2026-10-30-attention-component-preparation.md).
A source-only packet prepares predictor checks on linux10609 and encoder checks
on linux10603, independently in parallel. Synthetic post-RoPE operands, FP64
Q/K/V derivative oracles and observed-backend gates precede timing. Both attended
20-minute proposals remain unarmed; no real SSH/GPU or paid work has started.
Full-model equivalence, optimization adoption and study affordability remain
pending. Earlier entries below record historical preparation and outcomes.

**Evaluation work profile verified (October 30):**
[Exact observer/replay checks pass](report/checks/2026-10-30-eot-work-profile-review.md).
Full EOT8 controls take about 10.08 seconds, with 79% in backward. Attention
backward is about 54% of direct device duration in the traced first draw. A
separate offline reader corrects GPU-annotation double-counting; original
evidence remains frozen. Next prepare a strict FP32 attention component
diagnostic. No optimization, paid work or research admission follows.

**Evaluation work profiler prepared (October 29):**
[The bounded attribution diagnostic](docs/eot_evaluation_work_profile.md)
passes 52 focused CPU checks and fresh-worker replay. Full EOT8 controls retain
all original draws; detailed tracing covers one original draw with both mask
families. The 17.2 MB handoff and attended 20-minute Mac helper are verified
and unarmed. Real CUDA attribution remains pending; no further GPU work or
financial change has started. Earlier outcome entries below are historical.

**University projection-reuse result verified (October 29):**
[Four fresh CUDA workers pass correctness and exact replay](report/checks/2026-10-29-university-projection-reuse-review.md).
The one-clip median falls from 425.0 to 416.2 seconds, only a 1.021× speedup;
peak measured allocation stays below 5 GiB. This does not establish an affordable
study or qualify optimizer integration. The result archive is copied and
independently verified. Next profile the remaining evaluation work offline.
No further GPU work starts; earlier preparation/retrieval entries are historical.

**University diagnostic invoked; download pending (October 29):**
[Both download attempts timed out at login](report/checks/2026-10-29-projection-reuse-retrieval-pending.md).
The local launch claim exists; use only the download helper and enter the
university password promptly. No GPU result has been copied or reviewed yet.
Never repeat the model launch to recover files. Earlier preparation entries
below describe the state before this user invocation.

**University projection-reuse diagnostic prepared (October 29):**
[The separately identified one-clip comparison](docs/projection_reuse_university_diagnostic.md)
passes 75 focused CPU checks, four fresh tiny-worker replays and simulated
transfer/recovery checks. The 7.1 MB packet is verified; its attended 80-minute
university run has not started. Full pretrained correctness, memory and GPU
speed remain unknown. Existing scientific sources/policies and all financial
caps remain unchanged. Earlier entries below are historical.

**CPU-only cost-reduction proof verified (October 29):**
[Projection reuse passes 32 new CPU checks](report/checks/2026-10-29-projection-reuse-CPU-proof.md)
and all 140 focused checks in their documented process groups. It retains every
EOT draw/backward while reusing identical FP64 forward values within one bank.
Existing scientific sources and runners remain unchanged; CUDA speed and full
pretrained equivalence are unmeasured. Next prepare a separate university-GPU
correctness/profile diagnostic before any further paid work. RTX 3090 billing
settles to $1.1328, wallet $6.56; $0.1884 remains unreserved in the diagnostic
envelope, with the historical hold retained. Earlier entries below are historical.

**RTX 3090 partial benchmark reviewed; Pod terminated (October 29):**
[Ten of thirteen phases completed before the original cutoff](report/checks/2026-10-29-rtx3090-partial-benchmark-review.md).
Saved numerical checks and clean/adversarial recovery signatures pass; controls
and both feature phases remain incomplete. Evidence is copied and verified; the
Pod was terminated at **7:06 PM Pacific** after CPU-only retrieval. The observed
primary-only cost screen exceeds the project cap. Final billing remains pending;
no full hardware/research qualification or new paid work is authorized. Earlier
live and stopped-retrieval entries below are historical.

**RTX 3090 stopped; final evidence pending (October 29):**
The user reports stopping the Pod at **4:00 PM Pacific**, before its original cutoff.
[The retrieval checkpoint](report/checks/2026-10-29-rtx3090-stopped-retrieval.md)
preserves the 77 CPU/preflight checks and completed calibration bundle observed at
2:56 PM. Final benchmark completion is unknown. Keep the volume; review billing
and a separate download-only restart before retrieving evidence. Earlier live-run
entries below are historical; no model rerun or research launch is authorized.

**RTX 3090 diagnostic benchmark started (October 29):**
[The verified bounded setup is running](report/checks/2026-10-29-rtx3090-benchmark-launch.md)
on the user-deployed $0.52/hour RTX 3090. Its original deadline is **4:01 PM Pacific**
with a $1.30 session limit. CPU checks and numerical preflight precede benchmark
qualification; research calibration/trajectories remain pending. Earlier entries
below describe historical preparation stages.

**Lower-cost benchmark packet verified (October 28):**
[The corrected v4 handoff](report/checks/2026-10-28-lower-cost-candidate-packet.md)
passes 10 packaging and 77 restored CPU checks with unchanged scientific sources.
Billing and fresh sufficient-memory A5000/3090 quotes are deferred until tomorrow.
The packet is unarmed; resource allowance and an actual bounded operator remain
pending. Earlier benchmark and preparation entries below are historical.

**Measured EOT8 cost screen reviewed (October 28):**
[The retained full design screens at $243.71–249.62](report/checks/2026-10-28-candidate-measured-budget-review.md)
at the recorded SE rate, above the $200/$160 project/compute caps. All 32 focused
accounting/provenance tests and independent arithmetic pass. Missing costs remain
unknown; no research calibration or trajectory is admitted. Next reconcile billing
and prepare lower-cost hardware qualification while keeping the 20-class design.
Earlier benchmark and preparation entries below describe their historical stage.

**EOT8 hardware benchmark verified (October 28):**
[All 13 diagnostic phases completed](report/checks/2026-10-28-candidate-benchmark-verification.md)
on RTX PRO 4500 SE, including exact clean/adversarial two-update recovery.
Evidence is copied and independently checked; the Pod was terminated at 10:05 PM.
Next rebuild the full-study cost projection and reconcile billing before full
research calibration. The research step and research admission remain pending.
Earlier preparation and launch statements below describe historical stages.

**Opt-in EOT8 diagnostic path and offline benchmark prepared (October 28):**
[The separate candidate implementation](docs/candidate_eot8_implementation.md)
passes 77 focused checks from a fresh handoff extraction. Legacy four-draw code
and study settings remain unchanged. The packet includes strict artifact checks,
fresh-process recovery tests and deployment-inclusive worker deadlines. GPU
qualification, the research step size and complete cost remain unknown. Next
check lower-cost hardware quotes and reconcile the launch allowance; no Pod is
authorized or launched by this work. Earlier preparation entries below are historical.

**Research attack candidate prepared (October28):**
[The offline proposal](docs/research_attack_candidate.md) recommends qualifying
PGD20/R2/EOT8 primary evaluation with a matched PGD50/R2/EOT8 stronger control.
All23 focused checks, scaffold and independent budget arithmetic pass. The30-
evaluation training-only calibration is specified but unrun. New timing units
and total cost remain unknown; existing study settings and caps are unchanged.
Next implement/test the opt-in path and prepare a bounded hardware benchmark.

**Complete diagnostic panel analyzed (October28):**
[The frozen comparisons and independent arithmetic audit pass](report/checks/2026-10-28-complete-panel-analysis.md).
All20 registered cases and ten technical pairs reverify, with739 unchanged input
fingerprints. Both more steps and more attack-mask draws increase mean independent
loss; convergence and research admission remain unestablished. All41 focused
checks pass. Next review the research attack/calibration policy, numerical
qualification and measured budget. No new GPU work ran. Earlier entries are historical.

**All20 diagnostic reviews registered (October28):**
[The actual university receipt passes independent review](report/checks/2026-10-28-final-review-sync-verification.md).
University registration and local review now agree at **20/20 cases and ten
complete clip pairs**. All1126 protected hashes, original review identities and
time allowances match. The command is finished; SSH may close. Next analyze
the complete diagnostic panel and assess research admission. Earlier launch
and registration instructions below are historical.

**Final review registration prepared (October28):**
[Run the new metadata-only command](docs/final_panel_review_sync.md) once, then
verify the actual18-to20 university receipt. All72 local and72 extracted checks
pass, and the full saved-evidence rehearsal preserves1126 files and both exact
markers. Local progress stays20/20 diagnostic cases and ten pairs; university
registration remains18 until verified. This command starts no GPU work.

**Diagnostic panel complete (October28):**
[Both final bank1 cases pass independent review](report/checks/2026-10-28-last-two-verification.md),
including checks against their original bank0 partners. Progress is **20/20
diagnostic cases and ten complete clip pairs**. Both jobs and copies have ended;
SSH sessions may close. Next register the final two exact review markers
(university last verified at18), then analyze the complete panel and assess
research admission. Earlier launch instructions below are historical.

**Final-two activation verified (October28):**
[Both attended launchers are ready](docs/last_two_panel_cases.md).
The actual receipt verifies six exact records,928 unchanged hashes and all18
original case identities. Bowling has76m34s remaining; HandStandPushups bank1
has80minutes. Run each helper once from separate Mac Terminal tabs and keep
both connections/Mac awake through copies. No final numerical case has started.
[Actual audit](report/checks/2026-10-28-last-two-activation-verification.md).

**Final two cases prepared (October28):**
[Activate the new final-two handoff](docs/last_two_panel_cases.md), then verify
its actual receipt before either attended launch. All119 focused and195
combined extracted checks pass; full CPU path rehearsal preserves928 files.
Bowling carries206seconds and retains76m34s; HandStandPushups bank1 has80minutes.
Registered and independently reviewed progress is18/20 with eight clip pairs.
No final GPU case has started. Earlier instructions are historical.

**Preparing the last two cases (October28):**
[The HandStandPushups review-registration handoff is ready](docs/handstand_review_sync.md).
Its86 local and86 extracted checks pass; the complete CPU rehearsal preserves
918 files and the original marker. Run this metadata step once and verify its
actual17-to18 receipt before the final GPU handoff. Bowling retains76m34s;
HandStandPushups bank1 has its unused80-minute slot. No new GPU job is launched.

**HandStandPushups bank0 verified (October 28):**
[The completed export passes independent review](report/checks/2026-10-28-handstand-bank0-verification.md).
Progress is **18/20 diagnostic cases and eight complete clip pairs**; university
registration remains17. Both jobs have ended, so SSH sessions may close.
Next register this original review, then prepare Bowling bank1 with76m34s
remaining and the final HandStandPushups bank1 case. Earlier entries are historical.

**Bowling startup evidence verified (October 28):**
[The actual capture](report/checks/2026-10-28-bowling-startup-verification.md)
confirms 177 CPU passes, unchanged 774-file history and no Bowling case claim
or model attempt. Charge 206 seconds to its original allowance; 76m34s remain.
HandStandPushups on 10609 has reached matrix arm C and still has no completion
status. Keep its existing Terminal and the Mac awake. Progress stays **17/20
diagnostic cases and eight complete clip pairs**. Earlier entries are historical.

**Bowling startup guard refusal (October 28):**
+[linux10603 stopped before model dispatch](report/checks/2026-10-28-bowling-guard-refusal.md)
after 177 transcript CPU passes. A later snapshot shows an NX CUDA process.
Leave 10609's running HandStandPushups case open. A tested read-only startup
capture is ready; do not repeat either original model launcher. Progress stays
17/20 diagnostic cases and eight complete clip pairs. Earlier snapshots follow.

**Bowling / HandStandPushups activation verified (October 28):**
+[five exact records and 774 preserved hashes match](report/checks/2026-10-28-bowling-handstand-activation-verification.md).
Both attended helpers are ready for Bowling bank1 on 10603 and HandStandPushups
bank0 on 10609. Actual 177-check campus results and numerical exports remain
pending; progress stays 17/20 diagnostic cases and eight complete clip pairs.
Earlier snapshots follow.

**Bowling / HandStandPushups pair prepared (October 28):**
+[the new handoff](docs/reviewed_final_pair.md) preserves17 original reviews,
774 file hashes and consumed attempt history. All101 focused and177 combined
extracted checks pass; the full local activation/concurrent execution/copy
rehearsal rejects generated CPU outputs as CUDA evidence. First activate and
review the metadata before either attended GPU launch. Current progress stays
17/20 diagnostic cases and eight complete clip pairs. Earlier snapshots follow.

**YoYo/Bowling review registration verified (October 28):**
+[the actual university receipt](report/checks/2026-10-28-yoyo-bowling-review-sync-verification.md)
confirms both original reviews appended and763 unchanged hashes. Registered
and independent progress now agree at **17/20 diagnostic cases, eight complete
clip pairs and three remaining**. Next prepare the final cases with fresh GPU
checks. Earlier snapshots follow.

**YoYo/Bowling review registration prepared (October 28):**
+[the metadata-only handoff](docs/yoyo_bowling_review_sync.md) rechecks both saved
cases and appends their exact original reviews while preserving763 file hashes
and charged-time history. All64 local and64 extracted checks pass; the full
local rehearsal verifies15→17 and safe repetition. Actual university registration
is pending:17 independently verified cases,15 last registered and eight complete
clip pairs. Earlier snapshots follow.

**YoYo and Bowling verified (October 28):**
[both actual exports pass independent review](report/checks/2026-10-28-yoyo-bowling-verification.md).
The diagnostic panel now has **17/20 verified cases, eight complete clip pairs
and three cases remaining**. Bowling's copy retry succeeded without restarting
the model. Next register the two original reviews; university registration
remains15. No further GPU work is launched at this checkpoint. Earlier snapshots follow.

**YoYo bank1 verified; Bowling copy pending (October 27):**
[YoYo's actual export passes independent review](report/checks/2026-10-27-yoyo-bank1-verification.md),
bringing the diagnostic panel to **16/20 verified cases and eight complete
clip pairs**. Bowling bank0 also finished with exit0; both copies received
zero bytes. [Retry its existing download helper only](report/checks/2026-10-27-bowling-bank0-copy-failure.md).
The university coordinator last registered15 cases. Earlier snapshots follow.

**YoYo/Bowling activation verified (October 27):** [five records and588 preserved hashes match](report/checks/2026-10-27-yoyo-bowling-activation-verification.md).
The two attended helpers are ready for YoYo bank1 on10609 and Bowling bank0
on10603. Actual173-check campus results and numerical exports remain pending;
progress stays15/20 diagnostic cases and seven complete clip pairs. Earlier snapshots follow.

**Next pair prepared (October 27):** [YoYo bank1 and Bowling bank0](docs/reviewed_next_pair.md)
have a separate source-only activation and two bounded attended helpers.
All97 focused and173 combined extracted checks pass; the full local rehearsal
preserves588 hashes and rejects generated CPU cases as CUDA evidence. First
activate and independently review the fifteen-case boundary. No new university
job has started; progress stays15/20 cases and seven complete clip pairs.
Earlier snapshots follow.

**Archery review registration verified (October 27):** [the actual university receipt](report/checks/2026-10-27-archery-review-sync-verification.md)
confirms one original review appended and578 unchanged file hashes. University
and independent progress now agree at **15/20 diagnostic cases, seven complete
clip pairs and five remaining**. Next prepare YoYo bank1 and Bowling bank0;
no new GPU job is started. Earlier snapshots follow.

**Archery review registration prepared (October 27):** [the metadata-only handoff](docs/archery_review_sync.md)
rechecks saved evidence on CPU and appends its original review while preserving
578 file hashes and consumed attempt history. All76 focused tests pass locally
and after extraction; the full local rehearsal passes. Actual university
registration is pending:15 independently verified cases,14 last registered,
and seven complete clip pairs. Earlier snapshots follow.

**Archery recovery verified (October 27):** [the actual completed export passes independent review](report/checks/2026-10-27-archery-recovery-verification.md).
All172 campus checks, four attack arms, exact fresh replay, six GPU guards and
470 preserved hashes pass. The panel now has **15/20 verified diagnostic cases
and seven complete clip pairs**; five cases remain. The one-use grant is
consumed. Next register the original review through a separate metadata-only
step; the university coordinator last registered14 cases. Earlier snapshots follow.

**Archery recovery activation verified (October27):** [all five records and470 protected hashes match](report/checks/2026-10-27-archery-recovery-activation-verification.md).
The approved single80-minute attended helper is ready, and the new GPU check
is clear. The failed claim/STOP and39m08s charged are preserved. Actual GPU
recovery is pending; progress stays14/20 cases and six pairs. Earlier snapshots follow.

**Approved Archery recovery prepared (October27):** [one extra80-minute attempt has a verified handoff](docs/archery_recovery.md).
All96 focused/172 extracted checks and the full local rehearsal pass. The
failed claim/STOP and39m08s charged are preserved, with470 protected hashes.
The fresh GPU check is idle; first activate and independently review the
metadata. Actual recovery is unrun; progress stays14/20 cases and six pairs.
Earlier snapshots follow.

**Archery partial evidence verified (October27):** [the saved incident passes integrity review](report/checks/2026-10-27-archery-timecarry-partial-verification.md),
including161 JUnit checks and392 unchanged historical hashes. The incomplete
case remains rejected; progress stays14/20 accepted cases and six pairs.
A separate80-minute recovery exception is proposed for review. Earlier snapshots follow.

**Archery guard rejection (October27):** [the matrix-after check found another CUDA process](report/checks/2026-10-27-archery-timecarry-guard-failure.md).
The case remains incomplete after161 transcript CPU passes; its numerical claim
is consumed and the fresh replay did not start. A tested read-only evidence
copy is ready. Progress stays14/20 cases and six clip pairs. Earlier snapshots follow.

**Archery activation verified (October27):** [all five records and392 protected hashes match](report/checks/2026-10-27-archery-timecarry-activation-verification.md).
The single attended linux10603 helper is ready, with43 seconds carried and
79m17s remaining including setup. Its actual GPU run is pending; progress stays
14/20 accepted cases and six complete clip pairs. Earlier snapshots follow.

**Archery continuation prepared (October27):** [the verified source-only handoff](docs/archery_continuation.md)
carries43 seconds already used, leaving79m17s within the original allowance.
All85 focused/161 extracted CPU checks and the local full-path rehearsal pass.
First activate and independently review five metadata records; the GPU helper
stays gated. No new university case has run. Progress remains14/20 cases and
six complete clip pairs. Earlier snapshots follow.

**YoYo registration verified (October27):** [the actual university receipt](report/checks/2026-10-27-yoyo-review-sync-verification.md)
confirms14/20 accepted diagnostic cases, six complete clip pairs and382 unchanged
file hashes. Archery remains unclaimed with79m17s left under its original
allowance. Next prepare its separate guarded continuation; no new GPU helper
or research admission is added. Earlier snapshots follow.

**Next metadata step prepared (October27):** [register YoYo's original review](docs/yoyo_review_sync.md)
with one CPU-only append,382 preservation hashes and unchanged Archery startup
accounting. All71 local/extracted checks and the real-evidence/lock rehearsal
pass. The university sync has not run; independent progress stays14/20 cases.
No new GPU execution is authorized. Earlier snapshots follow.

**YoYo bank0 verified (October27):** [the completed export passes independent review](report/checks/2026-10-27-yoyo-bank0-verification.md),
bringing the diagnostic panel to14/20 cases and six complete clip pairs.
Archery bank1 remains unclaimed after its guard refusal;43seconds of setup
are preserved. Next register the one original YoYo review through a separate
metadata-only operation. Both current launchers have ended. Earlier snapshots
follow; full-panel analysis and research admission remain pending.

**Current university status (October27):** [Archery's stopped startup is verified](report/checks/2026-10-27-archery-startup-verification.md).
Its155 CPU tests passed, but GPU exclusivity blocked numerical dispatch;
no case was claimed. The corrected evidence copy verifies all286 historical
hashes, and43seconds of startup are preserved. Archery stays paused while
YoYo runs on10609. Acceptance remains13/20 cases and six complete clip pairs.
Earlier snapshots follow.

**Archery/YoYo activation verified (October27):** [all five records and286 preserved hashes match](report/checks/2026-10-27-archery-yoyo-activation-verification.md).
The two attended helpers are ready for Archery bank1 on10603 and YoYo bank0
on10609. Actual campus tests and numerical exports remain pending; progress
stays13/20 accepted cases and six complete clip pairs. Earlier snapshots follow.

**Next pair prepared (October27):** [Archery bank1 and YoYo bank0](docs/reviewed_panel_batch.md)
have a separate source-only activation and two bounded attended helpers.
All79 new and155 combined extracted CPU checks pass. First activate and
independently verify the thirteen-case metadata boundary; GPU helpers stay
blocked until that review. No new university case has started. Current
progress remains13/20 accepted cases and six complete clip pairs.

**Latest October27 update:** the [university review sync independently verifies](report/checks/2026-10-27-reviewed-pair-sync-verification.md).
Exactly two original markers were added and275 preserved file hashes match.
The coordinator now agrees with independent progress: **13/20 accepted cases,
six complete clip pairs and seven remaining**. All original failures, grants,
sources and time history remain intact. Next prepare the separate Archery/
YoYo continuation; no GPU job or scientific aggregate starts here. Older
snapshots follow.

**Current panel status (October 26):** ten of twenty diagnostic cases independently
verify, including Biking bank0. Floor bank1 stopped at the strict GPU-process
guard; its partial evidence is audited and remains incomplete. One explicitly
authorized additional 80-minute [attended recovery is prepared](docs/contended_panel_recovery.md),
with 63 passing CPU guards and the original failure/STOP retained. No recovery
GPU work has started. The remote ledger still records nine pending review sync;
full-panel analysis and research admission remain pending. Older snapshots follow.

The [October 26 two-host batch independently verifies](report/checks/2026-10-26-two-host-first-pair-verification.md):
nine of twenty diagnostic cases and four complete clip pairs. The university
bookkeeping is [synchronized](report/checks/2026-10-26-two-host-bookkeeping-sync-verification.md);
the stop flag remains active before the next bounded pair.

**Latest panel status (October 25):** [seven of twenty diagnostic cases are independently verified](report/checks/2026-10-25-attended-panel-seventh-verification.md), including the first bank for TennisSwing. Three training clips have both attack banks complete. The latest case passes native CUDA, attack-bound/prefix and exact fresh A-only replay checks in 41 minutes 33 seconds; peak allocation is 4.815 GiB. All prior cases and partial evidence are preserved. Thirteen cases remain; next is clip-03-bank-1. A separate second-host reference replay matches exactly on one training clip/arm; multi-host panel execution, full-panel analysis and research admission remain pending.

**Latest implementation (October 24):** the [matched-panel runner and handoff](docs/matched_panel_handoff.md) implement the frozen ten-clip/two-bank plan with first-case review, atomic whole-case recovery, bounded serial execution and complete-panel analysis. All 110 focused local checks pass, including 40 new; all 40 new checks also pass after extraction. [Preparation report](report/checks/2026-10-24-matched-panel-runner.md). These implementation checks preceded the independently verified university cases reported above.

**Frozen plan (October 24):** the [broader training-only plan](docs/matched_panel_diagnostic_plan.md) retains ten existing monitor clips and two paired attack banks (20 cases). Its67 predeclaration checks and280 local CPU mask draws passed before runner implementation. The14–17h repeated-reference scenario is scheduling context only, outside A40 economics.

**Latest verified diagnostic (October 24):** the [university matched attack check](report/checks/2026-10-24-university-matched.md) passes exact prefixes, native CUDA probes and fresh A-only replay on one training clip. It takes about 41 minutes and peaks at 4.89 GiB allocator use on RTX 3070. Step/mask effects depend on the scoring bank; no final attack is selected. Next is offline planning for broader training-only checks. No study admission or new paid Pod.

**Latest implementation (October 23):** the separate [matched attack diagnostic](docs/matched_attack_diagnostic.md) is implemented; 26 new and 91 focused CPU checks pass with zero skips, including exact paired prefixes and a fresh A-only replay. The subsequent university GPU outcome is recorded above. No new rental, primary-method change or study admission.

**Latest offline review (October 23):** [cost attribution and attack comparison](report/checks/2026-10-23-offline-a40-review.md) reproduce the A40 screen with 49 selected checks passing. About85% of timed work is evaluation. The saved PGD20/PGD50 contrast changes masks and starts as well as steps, so a [matched training-only diagnostic](docs/matched_attack_diagnostic_plan.md) was specified as the next control at that review checkpoint. That review did not execute the diagnostic; the later CPU implementation and university result are recorded above. No new rental or study configuration change.

**Latest completed diagnostic (October 23):** the [A40 controls and combined screen](report/checks/2026-10-23-a40-controls-completion.md) verify locally, with 509 fresh-host tests passing and no skips. The unchanged 20-class plan projects **$222–$224** at the quoted $0.61/hour, above the $200 cap. The old timeout remains failed; its six successful phases were reused. The user confirmed termination at 9:26 PM Pacific after verified export; final billing is pending. Next work is offline price/performance review.

**Research question:** During adversarial fine-tuning of V-JEPA 2's native masked prediction loss, does performance on unseen videos deteriorate after initially improving, even while adversarial training loss continues to decrease?

**Latest preparation (October 23):** the [controls-only handoff](report/checks/2026-10-23-a40-controls-preparation.md) is verified offline: 100 targeted local and 33 extracted checks pass. It reuses six completed A40 phases and measures only the missing controls, with a separate guarded cost projection. Fresh quote and host validation come next; no new Pod or research run.

**Latest CUDA outcome (October 23):** the frozen CUDA-host suite passes **509 tests with zero skips**. The [migrated A40 screen](report/checks/2026-10-23-a40-migrated-continuation.md) completed six of seven phases; its final control timed out at the reviewed limit. The export, numerical preflight, both short training checks and feature artifacts verify locally. Both Pods are user-confirmed terminated; billing reconciles to $5.47 cumulative credit used and $9.53 wallet credit. No complete A40 cost projection or research admission.

**Status: bounded shared-forward CUDA recovery passes; the measured study projection exceeds the $200 cap.** The [new budget review](report/checks/2026-10-22-measured-budget-review.md) puts the existing 20-class staged design at about $304–$307 with the retained allowances and reserves. The ten-class alternative is about $219–$221. These are incomplete projections; neither is admitted. The [October 20 proposal](docs/budget_study_plan.md) and its old $193.91 envelope remain historical. Hardware feasibility checks passed on CPU, Apple GPU, and RTX 5090; university RTX 3070 FP32 training ran out of memory. Staged manifests and deterministic preparation of 120 training/exploration clips are verified.

The [October 21 CUDA workflow report](report/checks/2026-10-21-cuda-workflow.md) records the failed exact full-model recovery comparison and a strict deterministic retry stopped after an attack took 16× longer. The repaired suite passes 225 tests on the CUDA host, including a small independent-process GPU replay test; that original-runtime attempt remains rejected. Both historical evidence exports are verified locally.

The [university component profile](report/checks/2026-10-21-university-cuda-components.md) completed all 48 workers: strict replay passes all 12 cases, while default attention gradients fail exact replay. Original Conv3d input backward increased from 1.102 ms to 3.400 s under strict settings on RTX 3070. This isolates a bottleneck on that host; the next gate is a validated execution option for the same Conv3d, followed by RTX 5090/full-model confirmation.

The [first convolution comparison](report/checks/2026-10-21-university-cuda-convolution.md) rejected all 12 workers before timing: the TF32-permitted cuDNN reference fails the fixed FP64 gate. Native matches FP64 but differs from that reference; no option qualified. The subsequent [TF32-off control](report/checks/2026-10-21-university-cuda-convolution-precision.md) passes all 12 university workers and observed replay, with unchanged tolerances. Native execution reduces isolated forward/backward from 3.410 s to 5.897 ms for input gradients and 3.431 s to 7.043 ms for parameter gradients. This qualifies a component candidate; whole-model speed/recovery and study adoption remain pending. The [opt-in native integration](report/checks/2026-10-21-native-integration.md) now binds backend/precision/cache/recovery identities and passes small-model gradients, PGD, exact recovery and an extracted four-process CPU fixture. The first university attempt stopped at a small probe with an incorrect kernel-name check. The [corrected CUDA comparison](report/checks/2026-10-21-university-native-model-comparison.md) verifies native kernel dispatch and exact within-backend replay; clean loss agrees, while input gradients reject the pointwise gate (0.3583% relative L2, 242,613 failed derivatives). The [verified derivative control](report/checks/2026-10-21-university-gradient-attribution.md) completes in 66.915s: all fixed-cotangent backward checks and bitwise saved-gradient reconstructions pass, while both FP32 forward paths reject the FP64 output gate (12 cuDNN / 35 native values). Forward-token differences change incoming tail derivatives; the backend remains unadopted. The [verified shared-forward CUDA control](report/checks/2026-10-21-university-shared-forward.md) passes forward/gradient oracle gates and exact within-backend replay on one training clip; FGSM/PGD-5 outputs also match between backwards. Native PGD-5+score takes 6.324s versus 40.745s with cuDNN (median of two single attempts). The [explicit training integration](docs/shared_forward_training.md) now passes CPU clean/adversarial AdamW recovery, including fresh processes and scheduled evaluation; 434 local tests pass with two CUDA-only skips. The [October 22 pretrained CUDA workflow](report/checks/2026-10-22-shared-forward-cuda-workflow.md) now completes all eleven phases: 436 CUDA-host tests pass; three representative inputs pass independently verified numerical/attack checks; clean and adversarial two-update restart comparisons match exactly. Feature/probe and PGD-50/FGSM/random controls complete on training data. The user confirmed Pod termination after verified export. The new billing increment is provisionally $1.085 from a combined summary, making provisional cumulative spending $2.280. Study calibration, precision/protocol freezing, measured budget admission and research results remain pending.

This project extends our [previous robust-overfitting research](https://github.com/ItsKaiwenDu/Robust-Overfitting) to latent video prediction. It retains the familiar `models/`, `scripts/`, `notes/`, and `report/` organization, along with weekly goals, progress reports, and a Lambda Labs guide.

## Project Overview

V-JEPA predicts representations of masked video regions from visible context. We will attack that prediction objective directly, then study the generalization dynamics of adversarial fine-tuning.

The first study uses **V-JEPA 2 ViT-L/16 with a fixed target encoder**. The context encoder and predictor will be updated; the target encoder will remain frozen. This keeps the target representation stable across checkpoints while retaining the native masked latent L1 loss. Findings will concern this adaptation protocol, not the original EMA pretraining procedure.

The intended contribution is a controlled study of training dynamics and their downstream consequences. Robust overfitting is a hypothesis to test, not an assumed result.

## Start Here

| Document | Purpose |
| --- | --- |
| [Verified thirteen-case sync](docs/reviewed_pair_sync.md) | Two original reviews registered;275 preserved hashes match and thirteen cases accepted |
| [Verified Biking/Archery pair](docs/reviewed_panel_continuation.md) | Thirteen accepted cases and six clip pairs; both original review markers registered |
| [YoYo review registration](docs/yoyo_review_sync.md) | Completed one-review registration; fourteen accepted and Archery time preserved |
| [Archery/YoYo outcomes](docs/reviewed_panel_batch.md) | YoYo verified; Archery startup refused before claim and remains pending |
| [Eleven-case metadata reconciliation](docs/reviewed_panel_reconciliation.md) | CPU recheck and append-only paused progress; legacy failures and STOP preserved |
| [Matched-panel university handoff](docs/matched_panel_handoff.md) | Frozen training-only panel, bounded first case, independent review, whole-case resume and separate scoring-bank analysis |
| [Checkpoint locks and fresh CPU retry](docs/checkpoint_lock_repair.md) | Verified NFS lock fix and the completed free CPU check |
| [CPU fixture repair and university check](docs/cpu_fixture_repair.md) | Verified initial checkpoints and a detached Linux CPU gate before another GPU rental |
| [Lower-cost GPU benchmark](docs/lower_cost_gpu_benchmark.md) | Retained 20-class design and bounded hardware screen; Partial A40 measurements verified; separate controls-only completion prepared |
| [Measured budget review](report/checks/2026-10-22-measured-budget-review.md) | Current unit-cost projections, remaining unknowns and required price/speed improvements |
| [Latest staged study proposal](docs/budget_study_plan.md) | Scoped research question, separate validation panels, staged costs, and decision gates |
| [Experimental protocol](docs/experimental_protocol.md) | Threat model, data split, training conditions, evaluation, and interpretation |
| [Study configuration](configs/study.yaml) | Machine-readable draft settings and unresolved provenance fields |
| [Setup guide](docs/setup.md) | Local development and university GPU preparation |
| [Local model checks](docs/smoke_test.md) | Pretrained checkpoint and native-loss correctness checks |
| [Staged data and evaluation](docs/staged_data_and_evaluation.md) | Verified exploration clips, native attacks, evaluation recovery, and provisional primary selection |
| [Feature probes and scheduled recovery](docs/workflow_diagnostics.md) | Train-only probe/calibration rules, epoch evaluation barriers, and bounded integration commands |
| [Latest shared-forward CUDA outcome](report/checks/2026-10-22-shared-forward-cuda-workflow.md) | Exact bounded clean/adversarial recovery, representative checks, controls and measured costs |
| [CUDA workflow pilot outcome](report/checks/2026-10-21-cuda-workflow.md) | Replay failure, deterministic runtime cost, verified exports, and the next profiling gate |
| [University CUDA component outcome](report/checks/2026-10-21-university-cuda-components.md) | Verified operator timings, strict Conv3d bottleneck, attention replay, and next check |
| [Convolution comparison outcome](report/checks/2026-10-21-university-cuda-convolution.md) | Verified numerical rejection, TF32 precision hypothesis, and unavailable timing/replay |
| [Verified TF32-off outcome](report/checks/2026-10-21-university-cuda-convolution-precision.md) | Native component accuracy/replay and timing; full-model acceptance remains pending |
| [Opt-in native integration](docs/native_execution.md) | Explicit backend/precision identities, first-order limits, and the next university full-model input/attack check |
| [Strict convolution precision control](docs/cuda_convolution_comparison.md) | Explicit TF32-off follow-up, unchanged numerical gates, and detached launch |
| [CUDA component profiler](docs/cuda_component_profile.md) | Bounded synthetic operator timing, exact gradient replay, and detached university launch |
| [CUDA workflow guide](docs/cuda_workflow_pilot.md) | Offline handoff, explicit deterministic policy, and bounded execution instructions |
| [Training and recovery](docs/training_runner.md) | Verified training inputs, optimizer/resume state, atomic checkpoints, and current limits |
| [Data and attacks](docs/data_and_attacks.md) | Grouped subsets, reproducible clips, and pixel-space attacks |
| [Mac GPU pilot](docs/mac_pilot.md) | MPS access, correctness checks, and explicit performance options |
| [Measured Mac feasibility](report/checks/2026-10-19-mac-feasibility.md) | Update times, memory, evaluation cost, and limits of a four-week schedule |
| [RTX 5090 feasibility results](report/checks/2026-10-19-runpod-5090-feasibility.md) | CUDA timings, university memory limit, verified export, and budget implications |
| [Evaluation and compute proposal](docs/evaluation_compute_plan.md) | Proposed checkpoints, complete workload ledger, price checks, and unresolved timing costs |
| [CUDA pilot handoff](docs/gpu_pilot.md) | University workstation status and CUDA feasibility commands |
| [Lambda Labs guide](setup_lambda_labs.md) | Rental workflow and the $200 spending limit |
| [Repository structure](docs/repository_structure.md) | How the familiar folders map to this project |
| [Goals](goals.md) / [Progress](progress.md) | Planned milestones and completed work |
| [Literature audit](notes/2026-10-16_novelty-audit.md) | Prior work, overlap, and limits of the novelty search |
| [Decision log](docs/decisions.md) | Reasons behind the current research scope |

## Experiment Design

The table below preserves the **original larger draft**, which is not admitted within the budget. The [latest proposal](docs/budget_study_plan.md) explicitly changes the data, seeds, evaluation schedule, and claim scope; it has not replaced or frozen `configs/study.yaml`.

| Setting | Original draft protocol |
| --- | --- |
| Model | Official V-JEPA 2 ViT-L/16 and released predictor |
| Target encoder | Fixed, clean-input, gradient-stopped |
| Dataset | UCF101 official split 1; source-group-separated validation holdout |
| Subset | 20 train / 5 validation / 5 test videos per class: 2,020 / 505 / 505 |
| Input | One deterministic 16-frame, 256 x 256 clip per video |
| Conditions | `pretrained`, `clean-finetune`, `adversarial-finetune` |
| Fine-tuning seeds | 0, 1, 2; identical data manifests across seeds |
| Training | 30 epochs; effective batch size 8; encoder and predictor updated |
| Training attack | Random-start PGD-5, L-infinity radius 4/255, step size 1/255 |
| Main evaluation | PGD-20, two restarts, expectation over four masks |
| Budget | Up to $200 total; $160 compute plus $40 overhead and contingency |

The full [protocol](docs/experimental_protocol.md) defines the controls and limitations. The [configuration](configs/study.yaml) records proposed settings; it is not yet a runnable training configuration or a frozen preregistration.

## Setup Guide

From the repository root, with Python 3.11 available:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_setup.py
```

The root dependency file installs only what the scaffold checker needs. For the implemented model integration and CPU tests, follow [local model checks](docs/smoke_test.md) and use `requirements-smoke.txt`. The RTX 5090 pilot passed with PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128; the feasibility report records the tested environment.

Optional checks after installing a suitable PyTorch environment:

```bash
python scripts/verify_setup.py --check-torch
python scripts/verify_setup.py --require-cuda
python scripts/verify_setup.py --require-mps
```

These commands inspect the environment without downloading models, preparing data, launching jobs, or renting GPUs. A passing scaffold check does not establish training readiness.

### Entry Point Status

| Script | Status |
| --- | --- |
| [`scripts/verify_setup.py`](scripts/verify_setup.py) | Implemented: scaffold, YAML, and optional device checks |
| [`scripts/smoke_test.py`](scripts/smoke_test.py) | Passed on the official pretrained checkpoint using CPU, MPS, and CUDA |
| [`scripts/prepare_data.py`](scripts/prepare_data.py) | Implemented: source-group-separated manifests from official split lists |
| [`scripts/prepare_clips.py`](scripts/prepare_clips.py) | Implemented: deterministic training/validation clips; eight pilot clips validated |
| [`scripts/check_attacks.py`](scripts/check_attacks.py) | Real-clip FGSM/PGD sanity checks passed on CPU, MPS, and CUDA |
| [`scripts/check_mps.py`](scripts/check_mps.py) | CPU/MPS comparison of losses, targets, and gradients on one real clip |
| [`scripts/check_patch_embedding.py`](scripts/check_patch_embedding.py) | Isolated correctness/timing comparison of equivalent patch projections |
| [`scripts/pilot.py`](scripts/pilot.py) | FP32 batch-eight MPS and RTX 5090 updates and full attack timing passed; RTX 3070 training ran out of memory |
| [`scripts/estimate_compute.py`](scripts/estimate_compute.py) | Reproduces partial planning projections; missing costs stay unknown and cannot approve a run |
| [`scripts/estimate_measured_study.py`](scripts/estimate_measured_study.py) | Offline shared-forward timing projection; rejects changed inputs, preserves unknowns and never admits a run |
| [`scripts/estimate_staged_study.py`](scripts/estimate_staged_study.py) | Implemented: staged cost comparison, source-hash checks, and metadata-only subset audit; never approves a run |
| [`scripts/train.py`](scripts/train.py) | Bounded training/resume; exact two-update CUDA recovery passes under the explicit shared-forward policy; research admission pending |
| [`scripts/export_staged_manifests.py`](scripts/export_staged_manifests.py) | Implemented: exact replay and export of the proposed subset identities |
| [`scripts/prepare_staged_clips.py`](scripts/prepare_staged_clips.py) | Passed: 100 training and 20 exploration clips decoded twice; confirmation/test pixels closed |
| [`scripts/evaluate.py`](scripts/evaluate.py) | Bounded native evaluation, atomic recovery and saved-anchor loading pass in the shared-forward pretrained CUDA pilot |
| [`scripts/check_workflow.py`](scripts/check_workflow.py) | Bounded feature/probe, calibration and scheduled recovery pass on pretrained CUDA with the explicit shared-forward policy |
| [`scripts/check_native_model.py`](scripts/check_native_model.py) | Original FP32 full-model numerical comparison rejected; the separate shared-forward candidate passes its bounded CUDA checks |
| [`scripts/profile_cuda.py`](scripts/profile_cuda.py) | Passed: all 48 university CUDA workers; strict replay on 12 cases; isolated timing evidence |
| [`scripts/university_profile.sh`](scripts/university_profile.sh) | Implemented: checksummed source handoff and detached, externally bounded university launcher |
| [`scripts/compare_cuda_convolution.py`](scripts/compare_cuda_convolution.py) | Passed: 12 TF32-off university workers and component replay; native timing qualifies a full-model candidate |
| [`scripts/university_convolution.sh`](scripts/university_convolution.sh) | Prepared: checksummed detached university comparison, no model/data downloads |
| [`scripts/check_matched_panel.py`](scripts/check_matched_panel.py) | Implemented and CPU-tested: separate frozen panel, first-case review, serial journal/recovery and complete-only descriptive analysis; CUDA panel pending |
| `scripts/run_report_checks.py` | Planned: attack and implementation diagnostics |
| `scripts/plot_results.py` | Planned: per-seed curves and aggregate summaries |

## Project Directory Structure

```text
V-JEPA-Robust-Overfitting/
├── .github/                 # Repository checks and contribution templates
├── assets/                  # Small README and presentation assets
├── attacks/                 # Bounded FGSM/PGD and fixed-mask/EOT objectives
├── configs/                 # Study settings and local path example
├── data/                    # Local videos/features; tracked split manifests
│   ├── manifests/
│   ├── raw/                 # Ignored
│   ├── processed/           # Ignored
│   └── features/            # Ignored
├── docs/                    # Protocol, setup, decisions, reproducibility
├── datasets/                # Official split parser and deterministic decoder
├── external/                # Ignored upstream checkout; instructions tracked
├── models/                  # V-JEPA adapter, frozen targets, loss, diagnostics
├── notes/                   # Literature notes and dated novelty audits
├── report/                  # LaTeX source and curated results
│   ├── main.tex
│   ├── references.bib
│   ├── pretrained/
│   ├── clean-finetune/
│   ├── adversarial-finetune/
│   ├── checks/
│   ├── figures/
│   └── tables/
├── scripts/                 # Familiar command-line entry points
├── tests/                   # Small CPU model checks; more checks planned
├── checkpoints/             # Ignored model weights
├── runs/                    # Ignored raw metrics and run metadata
├── run_logs/                # Ignored console/job logs
├── AGENTS.md                # Project guidance for coding assistants
├── goals.md
├── progress.md
├── setup_lambda_labs.md
├── requirements.txt
├── requirements-smoke.txt
└── LICENSE
```

Future artifacts retain the familiar condition/seed layout:

```text
checkpoints/<condition>/[diagnostic/]<run-name>/
runs/<condition>/[diagnostic/]<run-name>/
report/<condition>/[diagnostic/]<run-name>/
report/<condition>/overall/
```

Use the seeds from the eventual frozen protocol. The original draft uses `seed-0`, `seed-1`, and `seed-2`; the staged proposal uses exploratory seed 0 and separate confirmation seeds 11, 22, and 33. Repeated runs need distinct names, such as `seed-0-retry-01`; never overwrite a previous run. The pretrained reference uses `pretrained/reference/` because it has no fine-tuning seed. See [artifact conventions](docs/reproducibility.md).

## Results

No research results yet. [Correctness diagnostics](report/checks/2026-10-17-pretrained-smoke.json), [one training-clip attack check](report/checks/2026-10-17-real-clip-attacks.json), and [Mac feasibility measurements](report/checks/2026-10-19-mac-feasibility.md) are available. Planned research outputs include clean/adversarial loss trajectories, best-versus-final gaps, attack-strength checks, and linear-probe accuracy. Report observed overfitting, no observed effect, and inconclusive experiments distinctly.

## Weekly Goals & Progress

The target is a focused report by day 21, with days 22-27 reserved for corrections and review. Runtime and cost must pass the pilot before the full experiment begins. Follow [goals.md](goals.md) and record completed work in [progress.md](progress.md).

## Researcher & License

Researcher: **Kaiwen Du**. This is a separate project from the previous study; collaborators and acknowledgments will be added when confirmed.

Original repository code uses the [MIT License](LICENSE), matching the previous project. Upstream code, model weights, and datasets retain their respective licenses and terms; they are not relicensed by this repository.

## References

- [Assran et al. — V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [Rice, Wong, and Kolter — Overfitting in adversarially robust deep learning](https://proceedings.mlr.press/v119/rice20a.html)
- [Robust CLIP](https://arxiv.org/abs/2402.12336) and [ASTrA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17b08a9de93e2accf13429643e7eafdc-Abstract-Conference.html)
- [Literature map and novelty limits](notes/2026-10-16_novelty-audit.md)
