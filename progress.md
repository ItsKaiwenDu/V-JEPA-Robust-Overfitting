# Research Progress Reports

## 2026-10-21 — Bounded component profiler prepared

Implemented a synthetic component diagnostic using pinned upstream patch
embedding, masking, and rotary attention, plus isolated SDPA and predictor
sorting cases. Twelve cases use representative dimensions and one declared
primary-attack mask draw. Two fresh workers per default/strict policy compare
identical operands and observed output/input-gradient replay. Optional parameter
gradients do not construct an optimizer or alter the scientific implementation.

The detached university launcher verifies a source-only handoff and uses the
existing environment/upstream. Bounds are 20 seconds per worker, 480 seconds
for the diagnostic, and a 540-second external supervisor. A 4 GiB CUDA allocator
limit is explicit and does not guarantee physical-device usage. Atomic progress,
forward/backward stages, worker hashes, and optional operator tables preserve
partial/error/timeout evidence. Incomplete coverage returns nonzero and cannot
pass replay. No CUDA fallback, dependency install, model weights, or data pixels
are used by this handoff.

**Verification:** the complete local suite passed **248 tests**, with one
actual-CUDA skip and 85 upstream warnings in 84.47 seconds, on Python 3.11.16 /
CPU PyTorch 2.14.1. The 24 new tests cover independent gradients, representative
mask shapes, real process-group timeout cleanup, incompatible/tampered results,
precision/source matching, fresh processes, and optional profiling. This is
local fixture evidence; the new CUDA profile is not run or validated yet.

**Next user action:** copy the small prepared archive to the existing university
Desktop research directory and run its detached launcher. See the
[component guide](docs/cuda_component_profile.md). Diagnose the actual slow
operation before proposing an equivalent optimization or another rental.
Full-model recovery, research calibration/admission, and results remain pending.
Reported spending stays $1.195. No new paid compute, commit, or push. The final source-only archive is 19,275
bytes; all eight source-member hashes, shared-home alias verification, and
source-tamper rejection passed. Its restored CLI completed all 48 fresh CPU
fixture workers with observed strict replay on all 12 cases in 60.11 seconds.
[Preparation evidence](report/checks/2026-10-21-cuda-component-preparation.json).

## 2026-10-21 — CUDA replay failure, runtime repair, and cost blocker

Executed the bounded workflow diagnostic from milestone `4576aec` on the
user-launched RTX 5090. Offline restoration, environment checks, the released
checkpoint hash, and all 219 original tests passed. Calibration and initial
features completed. The full-model clean split/resume and uninterrupted
trajectories completed two updates each, but their exact comparison failed.
Contracts, RNG, and target matched; learned weights, optimizer moments, and
metrics differed. Starting-checkpoint attacks already differed before restore.

Added explicit strict CUDA determinism before initialization in each process,
binding warning mode and cuBLAS workspace to runtime provenance. Precision
flags and exact comparison criteria were preserved. All **225 tests passed**
on the Pod, including independent-process small CUDA gradient/update replay.
Local targeted checks passed 28 with one CUDA-only skip. A fresh full-model
retry completed one calibration candidate in 1,261.208 seconds versus 78.769
seconds under the original runtime: 16.011x. Stopped it during the second
candidate after 2,444.721 seconds to preserve the session budget. Full-model
recovery, adversarial trajectories, trained features, and stronger controls
remain unvalidated or unreached. The specific slow operation is not established.

Both evidence archives and all 42/15 member hashes passed local verification.
The user was asked to Stop the Pod; subsequent SSH returned connection refused.
After the delayed console update, the user corrected today's session charge
from $0.501 to **$0.603**, bringing reported spending to **$1.195**. This is a
user-reported ledger update, not independent billing API/invoice verification.
The session reservation is released against that corrected total; $8.805 remains
in the $10 pilot reserve. The user confirmed Pod termination after evidence
verification; its diagnostic checkpoints and volume were not retained. [Outcome and limits](report/checks/2026-10-21-cuda-workflow.md),
[structured evidence](report/checks/2026-10-21-cuda-workflow.json).

**Next:** isolate representative CUDA forward/backward costs and validate any
equivalent optimization before another rental. The small replay test alone
did not predict full-model cost. Near-duplicate review, novelty refresh,
ten-clip calibration, protocol freeze, and full budget admission still precede
research runs. No confirmation/test pixels opened; no robust-overfitting result,
commit, or push was produced in this step.

## 2026-10-21 — Bounded CUDA workflow handoff

Continued from committed milestone `2e04959` with a clean working tree.
Prepared the [CUDA workflow pilot](docs/cuda_workflow_pilot.md) using the
existing scientific implementations. It compares a saved-boundary process
exit/recovery with uninterrupted training for both conditions, checks exact
state and evaluation equality, and measures setup, checkpoint I/O, native
attacks, target caching, features, and fixed probe operations. Reused
evaluation times are separated from fresh computation; nested timings are
explicitly non-additive. Worker deadlines do not stop cloud billing.

The offline handoff contains checksummed source, the pinned upstream Git
bundle, 23 Linux Python 3.12 dependency wheels, and the complete 100-clip
training cache required by the existing loader. Execution selects eight
training clips and one training monitor. Held-out pixels and raw videos are
excluded. The wheels match the previous successful handoff hashes; their
27 active dependency requirements match the previous template plus these
wheels. Current Pod runtime compatibility still requires a live check.

**Verification:** 219 CPU tests passed, with 0 failures/skips and 81 upstream
FutureWarnings. Verification and package identities are recorded in the
[preparation evidence](report/checks/2026-10-21-cuda-handoff.json). The new
small-model CPU checks cover the complete pilot phases, exact recovery,
optimizer corruption, missing/corrupt evaluation evidence, cached timing,
worker timeouts, and archive integrity. The prior 43 recorded scientific
source/configuration/evidence hashes are preserved.

**Limits:** No cloud instance, new paid compute, full-model CUDA recovery,
study calibration, or research trajectory was run. Confirmation/test pixels
remain closed. The $0.592 reported spend, $9.408 remaining pilot reserve,
and $200 total/$160 compute caps are unchanged. No commit or push was made.

**Next user action:** check a current Runpod RTX 5090 quote with 30 GB container
disk and an 80 GB persistent volume at `/workspace`. The proposed diagnostic
session is bounded to at most two billed hours/$3, with a shorter worker
deadline if setup or the live all-in rate requires it. Start only when ready
to connect, export, and stop the instance. The full study remains unadmitted.

## 2026-10-20 — Fixed probes, calibration records, and scheduled recovery

Continued from committed milestone `7b2b7cd` with a clean working tree. Added
[clean probes and scheduled diagnostics](docs/workflow_diagnostics.md): unmasked
mean-pooled context features, training-only standardization, fixed ridge lambda
one under an explicit sum-loss convention, and raw variance/effective-rank
measurements. Calibration compares three primary attack step sizes using only
declared training-monitor clips and independent scoring masks. It records all
candidate evidence and refuses changed weights, source/runtime, or data.

The epoch workflow checkpoints before evaluation and refuses to advance past
unfinished comparisons. It recovers from per-clip interruptions, an interruption
after saving an epoch, and a failed workflow-record write after evaluation.
Matched clean/adversarial runs can share identical pretrained evaluations.
Explicit recovery continues one trajectory directory; reruns require new names.

**Verification:** 197 CPU tests passed, 0 failed/skipped, with 73 upstream
FutureWarnings. The 53 new tests include actual small upstream transformers,
an independent primal ridge solution, split/provenance rejection, real primary
attack calibration replay, and exact scheduled-versus-uninterrupted recovery
for both training conditions (weights, optimizer, history, RNG).

The released full ViT-L checkpoint also passed clean feature extraction and
probe fitting on eight existing training clips from two classes: finite
`8 x 1024` features, centered rank 7 (sample bound 7), and unchanged model and
target weights. Extraction took 23.22 seconds on local CPU, excluding setup.
The probe's training-set score is only a plumbing check; no held-out accuracy
or collapse conclusion was drawn. Raw features/probe tensors remain ignored.
[Evidence record](report/checks/2026-10-20-workflow-checks.json).

**Limits:** No full-model attack calibration, scheduled training, CUDA recovery,
research trajectory, or validation/test probe scoring was executed in this
step. Confirmation/test pixels remain closed. The existing study/proposal,
manifests, measured-pilot evidence, and earlier scientific source hashes are
unchanged. No new cloud instance, paid compute, commit, or push was performed.
The user-reported $0.592 spend remains inside the $10 pilot reserve; $9.408
remains reserved, within the unchanged $200 total/$160 compute caps.

**Next:** prepare the bounded full-model CUDA recovery/cost-measurement pilot,
then measure its missing operations. Near-duplicate review, novelty refresh,
measured budget admission, and protocol freezing still precede the exploratory
pair. No instance is needed while preparing that handoff.

## 2026-10-20 — Staged exploration data and native evaluation checks

Verified user commit `a22c2b9` and began from a clean working tree. Exported the
proposed subset to separate manifests without changing the parent data/config.
Extracted only 100 training and 20 exploration-validation sources from the
verified archive. All 120 clips produced exactly identical pixels and metadata
on a second decode, and saved pixels passed identity/range/checksum verification.
No duplicate source/pixel hashes were found; near-duplicate review remains
pending. Confirmation/test pixels were not extracted, decoded, or evaluated.
[Data evidence](report/checks/2026-10-20-staged-data-preparation.json).

Implemented native routine/primary/convergence evaluation, random/FGSM controls,
independent scoring-bank audits, atomic per-clip recovery, verified anchor
loading, and provisional primary-validation selection. Model/RNG restoration
and strict checkpoint/data/settings binding prevent evaluation from contaminating
training or reusing stale checkpoint results. The diagnostic CLI is limited to
one or two training-monitor clips; full epoch orchestration remains pending.

**Verification:** 144 tests passed, with no failures/skips and 54 upstream
FutureWarnings. The 49 new checks include real small upstream transformers,
interruption recovery, scoring-bank separation, changed/corrupted inputs,
anchor loading, and selection rejection cases. Full pretrained/GPU execution
of the new evaluation command was not performed.
[Evaluation evidence](report/checks/2026-10-20-evaluation-checks.json).

Rebuilt the cleared temporary Python environment in the ignored local `.venv`
with the same pinned scientific versions; the current complete package set is
recorded. No new paid compute, cloud instance, research trajectory, commit, or
push was performed. Prior pilot reports and scientific source hashes remain
unchanged. The $0.592 user-reported spend and $200 cap are unchanged.

**Next:** fixed clean probes/feature diagnostics, training-only calibration
records, and epoch evaluation orchestration. Then validate full-model GPU
recovery and time missing costs before admitting the exploratory pair. No
user action is needed for the remaining local implementation.

## 2026-10-20 — Resumable FP32 training core and recovery tests

Implemented the [training runner and recovery workflow](docs/training_runner.md).
The reusable core supports clean/adversarial native updates, matched data/mask
seeds, warmup and LR drops, and correct partial-batch normalization. The
training-only input loader verifies manifest membership, metadata, shapes,
raw-pixel bounds, and cached-pixel hashes. Fixed clean targets are recomputed
and cached in CPU memory for each attempt.

Checkpoints preserve context/predictor weights, AdamW moments, update position,
scientific history, and Python/NumPy/PyTorch RNG states. Source/config/data,
fixed target identity, and runtime fields are bound to the resume contract.
Atomic manifest commits retain two resume generations and declared anchor
weights. Failed writes preserve prior commits; corrupted latest files require
explicit recovery. Partial updates cannot be serialized. Resumes use a fresh
attempt directory linked to the parent checkpoint.

**Verification:** All 95 local tests passed (29 new), with 35 upstream
FutureWarnings and no skips. Six comparisons using small pinned-upstream
models reproduced uninterrupted clean/adversarial trajectories exactly after
resume, including optimizer states and RNG, across warmup, LR drops, and
partial batches. Additional checks cover invalid optimizer/data/target state,
low disk space, failed commits, checksum failures, and sampler parity.
[Evidence record](report/checks/2026-10-20-training-resume-checks.json).

The CLI currently requires diagnostic mode, limits each attempt to eight
updates on at most eight prepared training clips, and rejects non-diagnostic
research runs. No full pretrained runner execution, CUDA/MPS resume validation,
new GPU rental, validation/test evaluation, commit, or push was performed.
Existing scientific source, pilot reports, and protocol 0.3 hashes are intact.
The new runner's end-to-end timing and I/O costs remain unmeasured.

**Next work:** Export and validate the staged data, integrate primary
attacks/checkpoint selection and diagnostics, then measure missing costs
before admitting the exploratory pair. No new user action is required for
those local implementation steps.

## 2026-10-20 — Staged budget proposal and reported pilot charge

Recorded Kaiwen's reported Runpod session charge of **$0.592**, including
about ten minutes of idle time. This is user-reported, not invoice/API
verification. It is inside the shared $10 pilot reserve, leaving $9.408
reserved for further bounded measurements. The $200 total cap is unchanged.

Prepared the [staged study proposal](docs/budget_study_plan.md), separate
[configuration](configs/budget_study_plan.yaml), and reproducible
[projection/metadata audit](report/checks/2026-10-20-staged-study-projection.json).
The recommended candidate uses 20 classes, 100 training videos, 20 exploration
validation videos, 20 fresh confirmation validation videos, and 60 final-test
videos. One exploratory matched pair precedes three conditional confirmation
pairs. Primary attack strength and model/input shape are preserved; the
reduced update count, sparse anchors, and narrower claim are explicit.

The original validation set has too few groups per class to divide casually
into two independent panels. The proposal withholds exploration groups from
the parent training pool before selecting actual training videos and retains
the original validation partition for confirmation. Metadata audit passes
video/group separation; 100 training videos span 100 groups and 60 proposed
test videos span 59 groups. Exact proposed rows are recorded. No test pixels
were opened; video content integrity and final manifests remain unvalidated.

At the historical $1.20/hour, planning allocations are $33.64 exploration and
$110.27 confirmation, including 25% headroom and respective two/six-hour
allowances for untimed work. Including the $10 pilot and $40 noncompute
reserves gives **$193.91**. This is a conditional envelope, not a complete
measured cost or spending approval. The calculator leaves unknown timings
null, labels proxies, compares alternatives, and always rejects automatic
admission. Full training/resume, complete diagnostics, cost measurements, and
protocol lock remain pending. Original study/configuration and pilot evidence
were preserved.

**Verification:** 15 focused staged-budget and existing compute-plan tests
passed. They check phase/reference counts, no exploratory test evaluation,
fresh confirmation seeds, source-group separation, timing compatibility,
unknown costs, and pilot spending counted once. No GPU was rented, no full
training was run, and no commit or push was made in preparing the proposal.

**Next work:** Implement and test the resumable training runner locally,
export and validate the proposed manifests, then measure the outstanding
operations before admitting the exploratory pair.

## 2026-10-19 — University memory limit and completed Runpod RTX 5090 pilot

The user reported successful CUDA smoke and one-clip attack checks on university
linux10609 (RTX 3070 8 GB). The FP32 optimizer pilot then failed during the
second clean update's backward pass after the first optimizer state allocation.
No adversarial training timing was obtained on that host.

Connected to the user's running Runpod RTX 5090 using a host key independently
verified through its web terminal. Reconstructed the existing verified source
snapshot and eight training clips. Recovered from GitHub/PyPI network timeouts
using the same upstream Git revision and Linux wheels transferred from the Mac.
All 56 tests, model smoke, attack checks, and the complete bounded pilot passed.
Five measured updates averaged 4.026 s clean and 24.425 s adversarial; primary
PGD-20/R2/EOT4 evaluation took 73.442 s/clip. Maximum measured training allocator
peak was 7.551 GiB. Scientific source and study configuration were unchanged.

[Results and provenance](report/checks/2026-10-19-runpod-5090-feasibility.md)
and [updated partial cost projection](report/checks/2026-10-19-runpod-5090-projection.json)
are saved locally. The original training schedule alone projects 179.95 hours
(about $215.94 at the displayed $1.20/hour); the proposed evaluation-inclusive
partial projection is 791.61 hours, about $949.93, with 15 categories still
unmeasured. The full study is not admitted within the $200 budget.

Exported reports, logs and environment metadata and verified the archive hash.
The user confirmed termination after stopping the Pod and verifying local export.
Termination is user-reported; exact launch time and billed cost remain unreconciled. No full study, held-out test evaluation,
commit, or push was performed. Next: reassess workload/implementation costs
before another long GPU run.

## 2026-10-19 — Evaluation schedule, cost ledger, and initial cloud handoff

Prepared a [reviewable evaluation/compute proposal](docs/evaluation_compute_plan.md)
without changing protocol 0.3 or `configs/study.yaml`. It uses one fixed
training/validation clip per class for routine monitoring, nine declared
post-training epochs for primary attacks, and the full 505-video final test
set. Best-checkpoint selection would be restricted to the strong-evaluation
epochs; this limitation is explicit. Three seeds and both conditions remain.
The proposal, subsets, probe details, and exact mask seeds are not locked.

Implemented a read-only workload calculator and recorded its
[partial projections](report/checks/2026-10-19-compute-projection.json).
The proposed schedule totals 1,209.66 Mac hours using measured unit timings,
plus 483.60 hours from rough attack proxies and 15 unmeasured categories.
Unknown costs are represented as null and cannot approve a run. The declared
10-video training fallback and denser primary validation alternative are
calculated separately. These are planning extrapolations, not experiments.

Rechecked official Lambda and Runpod price listings, preserving the
distinction between advertised rates, live quotes, and measured throughput.
Proposed a shared $10 pilot reserve inside the $160 compute envelope, 25%
runtime headroom for full admission, and the existing $40 noncompute reserve.
No cloud performance is inferred from GPU specifications. No account,
instance, filesystem, paid job, or final-test evaluation was accessed.
Kaiwen confirmed an existing Lambda account; one 48 GB A6000 is the first
pilot candidate, pending a live availability and price check.

Prepared the [initial two-hour pilot handoff](docs/cloud_pilot_handoff.md),
including the current source snapshot, existing eight training clips, hashes,
CUDA correctness/timing commands, and export instructions. The initial pilot
uses implemented native checks; baseline/probe timing and resume support
remain work for the full-admission stage. No source was pushed or committed.

**Verification:** All 56 local tests passed, with 15 upstream warnings.
Five planning tests cover missing costs, shared-reference
counts, preserved validation/test work under the training fallback, rejection
of reduced timing evidence, and the denser schedule. Existing scientific
source and its report hashes are unchanged. Source/archive reconstruction
and final repository validation are checked before handoff delivery.

**Next human input:** Check Lambda's live single-A6000 availability and price
for a bounded first pilot. Access is needed for a CUDA measurement;
the $200 full-study budget is not yet demonstrated feasible. Complete
remaining local timing interfaces and resume checks before admitting the
full experiment.

## 2026-10-19 — Mac GPU correctness, performance fix, and timing pilot

Confirmed that the M4 Max's MPS backend is available with normal host GPU
access. The restricted execution environment reported it unavailable with
the same Python interpreter; this explains the earlier local access result.
The machine has 32 GPU cores and 36 GiB unified memory. No driver, allocator,
power, or sleep setting was changed.

The full pretrained smoke check and one-real-clip CPU/MPS comparison passed.
Input-gradient relative L2 error was 0.684%, with cosine similarity 0.999978;
losses, targets, and two selected parameter-gradient tensors passed their
recorded thresholds. The MPS float64 limitation was handled by performing
diagnostic norm reductions on CPU, while model forward/backward remained on
the selected GPU. No CPU fallback was enabled for model operations.

The original MPS Conv3d context projection made input-gradient attacks slow:
one-clip PGD-5 took 121.29 seconds. Added an explicit equivalent linear patch
projection that reuses the original parameters and state keys. The target
encoder and upstream source are unchanged; Conv3d remains the default.
CPU projection tests and full pretrained MPS checks passed. On the same clip,
optimized PGD-5 took 4.84 seconds with matching reported MPS attack losses and
perturbation maxima. The original and optimized evidence is preserved.

**Measured pilot:** FP32, microbatch one, effective batch eight, activation
checkpointing, and cached clean targets. Both conditions restart from the
released weights and have one warmup plus five measured updates. Clean
updates averaged 9.09 seconds and adversarial updates 49.33 seconds. Both
updated context/predictor weights while preserving the fixed target state.
PGD-20 with two restarts, four attack-mask draws, and four independent scoring
draws took 161.09 seconds on one training clip. The largest sampled MPS driver
allocation was 10.20 GiB; live tensors reached about 8.00 GiB. These sampled
maxima are not exact peaks or total system RAM measurements.

**Feasibility:** Six planned 30-epoch runs project to 369.53 hours (15.40 days)
of training alone. One primary evaluation pass over 505 clips projects to
22.60 hours. Up to 13 distinct pretrained/best/final test checkpoints would
add 12.24 days before monitoring, stronger attacks, probes, and other work.
An under-four-week full-study commitment on this Mac is therefore not
supported by these measurements. Keep the Mac for development and bounded
pilots while choosing a measured backend and evaluation schedule. See
[the feasibility record](report/checks/2026-10-19-mac-feasibility.md) and
[raw pilot measurements](report/checks/2026-10-19-mps-linear-pilot.json).

**Verification and provenance:** 51 local CPU tests passed, including
equivalent projection output/input/parameter-gradient checks, with 15 upstream
deprecation warnings. Separate pretrained GPU reports record source hashes,
packages, checkpoint identity, and runtime metadata. The original-projection
source is retained as a patch against base commit
`03e87ddf8f78810ccece272b9ad548c1cf03486e`; replay reproduced the earlier report
hashes. No final-test clips or full training trajectories were evaluated.
No Lambda instance or paid compute was used; the $200 allowance is unspent.

**Next step:** Budget and freeze routine monitoring plus final evaluation;
implement the full training runner with resume checks before long runs.
Compare a measured faster CUDA option or revise scope explicitly if the
four-week target is retained. The university RTX 3070 remains blocked by its
reported driver mismatch. During the local pilot the battery fell despite
reported AC power, so sustained local work also needs a power connection
that maintains charge. No new access or purchase is needed to review these
results and prepare the next implementation step.

## 2026-10-17 — Pretrained checks, real clips, and CUDA pilot preparation

Loaded the complete official ViT-L checkpoint with strict encoder, target,
and predictor state checks. Its SHA-256 is now pinned in protocol 0.3 and
`configs/study.yaml`. The full CPU smoke check passed native/upstream loss
parity, replay, finite nonzero gradients, and target freezing. This used
synthetic input and took no optimizer step; see the
[pretrained report](report/checks/2026-10-17-pretrained-smoke.json).

Implemented native-objective FGSM/PGD, fixed mask-bank EOT, grouped UCF101
manifest generation, deterministic clip decoding, and a small CUDA timing
pilot. The official split-1 annotations produced 2,020 training, 505
validation, and 505 test entries, with no shared video IDs or source groups.
The manifest aggregate SHA-256 is
`bf773c186fcff9ca255896daf601d56f3e6dece3ac404e334afff784b6726ed7`.
Video-content duplication and full dataset integrity remain unchecked.

Downloaded the official dataset archive and extracted eight training clips.
The initial AVI decode exposed missing final B-frame timestamps; FFmpeg's
generated-timestamp option introduced duplicate timestamps. The final
decoder preserves observed timestamps and repairs missing entries only
when all observations match the declared constant frame rate. A regression
test covers this case. All eight clips decoded successfully and reproduced
identical pixel hashes on a second decode. No final-test video was decoded
or evaluated. Sources, hashes, selected frames, and repairs are recorded in
[the data preparation report](report/checks/2026-10-17-data-preparation.json).

On one real training clip with the released full model, clean native loss
was 0.580584; FGSM increased the attack-bank loss to 0.622708 and PGD-5 to
0.693803. Both stayed within 4/255, and all model weights and parameter
gradients remained unchanged. Independent scoring masks were also recorded.
This is an implementation diagnostic, not a robustness benchmark or evidence
of robust overfitting. See the
[attack report](report/checks/2026-10-17-real-clip-attacks.json).

**Verification:** 48 local CPU tests passed, including actual pinned upstream
loss integration, attack behavior, grouped splits, FFmpeg AVI decoding,
and clean/adversarial pilot optimizer updates on small fixtures. There were
15 upstream deprecation warnings. CUDA execution has not been tested; the
local Python runtime exposes neither CUDA nor MPS. GitHub CI checks scaffold
and syntax only and does not run these scientific checks.

**Provenance:** The pretrained smoke report records clean project commit
`56e1417de2100b9e45f6e5fad1163a13f0763ada`; the real-clip attack report records
hashes of the new source files used. All model checks use upstream revision
`204698b45b3712590f06245fbfba32d3be539812`. Checkpoint, archives, extracted
videos, processed clips, and the upstream checkout are stored locally in
ignored directories; small diagnostic reports and split manifests are tracked
research artifacts. The local checks used an isolated temporary Python 3.11
environment, not a validated CUDA environment.

**Compute:** No Lambda instance, rented GPU, or paid compute was used.
The $200 total allowance remains unspent. Confirmatory training, full
evaluation, stronger-attack convergence, and robust-overfitting measurements
have not started.

**Next human action:** Provide access to an allocated university CUDA GPU
or run the environment commands in [the GPU pilot guide](docs/gpu_pilot.md).
The pilot will measure memory, update time, and attack-evaluation time before
choosing a feasible schedule or considering a rental. The full training
runner, resume support, and total cost validation remain upcoming work.

## 2026-10-17 — Repository move and local model integration

Verified the new root README at
`/Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting`, branch `main`,
and remote `ItsKaiwenDu/V-JEPA-Robust-Overfitting`. The move left `.gitignore`,
`.github/`, `.editorconfig`, and `.gitattributes` in the old folder. Restored
those six project files and removed four generated Python cache files from
the working tree; no history rewrite, new commit, or push was performed.

Implemented a pinned upstream adapter with raw-pixel normalization,
fixed clean targets, native masked L1, strict three-component checkpoint
loading, and reproducible smoke-test masks. Added a synthetic full-checkpoint
smoke-test command, CPU tests, and [setup instructions](docs/smoke_test.md).
The code was developed and tested in an isolated temporary environment.

**Verification:** 20 CPU fixture tests passed, including comparison with the
actual upstream loss function, finite differences, masking, target freezing,
parameter updates, activation checkpointing, and rejected malformed
checkpoints. The full constructed ViT-L state names/shapes matched official
checkpoint metadata inspected through a small byte-range request. Details,
environment versions, the initial fixture failure, and limitations are in
[the local check record](report/checks/2026-10-17-local-model-checks.md).

**Experiment status:** No pretrained tensor data loaded, no dataset prepared,
no attack/training/robustness evaluation run, and no paid compute used. Small
random-model fixture tests are not research results. Full pretrained
execution, the complete checkpoint hash, CUDA feasibility, and cost are pending.

**Next step:** Explicitly download the released checkpoint and execute the
full native-loss smoke test on an available suitable device. Then implement
real-video preprocessing and attacks before measuring the GPU pilot. Lambda
comes only after a measured need and the existing budget gate.

## 2026-10-16 — Research planning and repository setup

**Progress report:** Reviewed the prior ChatGPT discussion and current public
literature, then agreed on a focused V-JEPA 2 study with a fixed target
encoder, one small video dataset, three seeds, a report as the initial
deliverable, and a total budget of at most $200. No exact matching public
study was found; this is a bounded search result, not proof of novelty.

Inspected the previous Robust-Overfitting repository and created this local
repository using the same main organizational conventions. Added explicit
homes for configurations, the experimental protocol, data manifests,
provenance, and future scientific checks. The first implemented utility is
the scaffold/environment verifier.

**Deliverables:**

- [README](README.md) and [repository map](docs/repository_structure.md).
- [Draft protocol](docs/experimental_protocol.md), [configuration](configs/study.yaml), and [decisions](docs/decisions.md).
- [Literature audit](notes/2026-10-16_novelty-audit.md) and [goals](goals.md).
- [Setup](docs/setup.md), [Lambda guide](setup_lambda_labs.md), and [reproducibility conventions](docs/reproducibility.md).
- [Setup verifier](scripts/verify_setup.py) and GitHub scaffold-check workflow.
- [Report outline](report/main.tex) and initial bibliography.

**Repository verification:** The scaffold verifier passed from both the
repository root and another working directory using Python 3.14.5. Python
syntax checks passed with Python 3.11 and 3.14. All 50 local Markdown links
resolved; whitespace, YAML, workflow structure, and issue-template
frontmatter checks passed. Checked 15 paths that should be ignored and 15
that should remain trackable. The local branch is `main`, with no remote
configured. GitHub Actions has not run, and the LaTeX outline has not been
compiled.

**Experiment status:** Not started. No model weights or dataset downloaded,
no scientific tests run, and no rented compute used by this project setup.
Training/evaluation scripts and GPU dependency locking remain upcoming work.

**Next step:** Confirm the university CUDA environment and implement the
official checkpoint/native-loss smoke test before any full experiment.

## Progress entry template

```text
Date / week:
Objective:
Completed work and evidence:
Run IDs / commits / protocol version:
Findings, including failed or negative results:
Compute spending and remaining allowance:
Open problems:
Next step:
```
