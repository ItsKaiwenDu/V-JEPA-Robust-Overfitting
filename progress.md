# Research Progress Reports

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
