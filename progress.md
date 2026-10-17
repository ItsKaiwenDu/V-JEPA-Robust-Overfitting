# Research Progress Reports

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
