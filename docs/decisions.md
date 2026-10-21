# Research decisions

## 2026-10-16 — Initial scope

| Decision | Reason and consequence |
| --- | --- |
| Focused research report | A useful first deliverable within three weeks plus a short review buffer; publication quality depends on evidence |
| V-JEPA 2 ViT-L first | Released encoder/predictor; avoids the smaller V-JEPA 2.1 distillation mismatch risk |
| Fixed target encoder | Stable loss reference across checkpoints; conclusions exclude the original moving-target protocol |
| One small video dataset | Preserve budget for repeated training and strong evaluation |
| Three fine-tuning seeds | Expose variation; data split is shared across seeds |
| Up to $200 total | Use university/local resources first and measure rental feasibility |
| 10-20 human hours weekly | Reserve human review for access, unexpected results, and scientific interpretation |

These choices were confirmed in the research planning discussion. UCF101,
the 20/5/5 per-class subset, and numerical training defaults are the proposed
implementation of that scope, subject to the documented feasibility gate.

## 2026-10-16 — Repository setup

Keep the previous project's root `models/`, `scripts/`, `notes/`, `report/`,
`goals.md`, `progress.md`, and `setup_lambda_labs.md`. Add configurations,
protocol documentation, split manifests, and lightweight GitHub checks.
Continue the previous repository's MIT license for original code.

No previous experimental results, model code, collaborator list, or private
conversation exports are copied into this project. Only the organization
and useful documentation conventions are carried over.

## 2026-10-17 — Local model integration (protocol 0.2)

Pin Meta's source at `204698b45b3712590f06245fbfba32d3be539812`. Construct the
upstream V-JEPA 2 architectures directly instead of using the current Hub
loader's localhost URL and non-strict loading. Preserve all ten predictor
mask tokens observed in the official checkpoint metadata. A single sequence
group uses mask token 0, matching the upstream wrapper. Record upstream
pixel-normalization constants in the study configuration.

These resolve implementation details without changing the fixed-target
question, 16-frame input, dataset subset, seeds, or training settings.
Checkpoint metadata inspection is not a successful pretrained load. The
full file hash, full-model execution, CUDA compatibility, and feasibility
are still unverified. See [the smoke-test guide](smoke_test.md).

## Decisions still requiring pilot evidence

- CUDA execution and memory/runtime on the actual experiment hardware.
- Full-dataset decoding, corruption/duplicate checks, and final preprocessing lock.
- GPU microbatch, accumulation, precision, and optimizer implementation details.
- Evaluation step sizes, fixed subsets, attack/scoring mask banks, and probes.
- Measured end-to-end cost and whether 20 or 10 training videos per class fit.

Resolve these before locking the protocol, without using final test outcomes.

## 2026-10-17 — Pretrained/data/attack checks (protocol 0.3)

The official checkpoint passed strict CPU loading, native-loss agreement,
input/parameter gradients, target freezing, and replay. Its SHA-256 is now
recorded. Official split-1 metadata yields 20/5/5 videos per class with
source-group separation and manifest hash recorded in the study config.

Choose the documented centered 4 FPS deterministic clip recipe for the
pilot. Real AVI inspection revealed a missing final timestamp; blindly
enabling FFmpeg's `genpts` also duplicated an existing timestamp. Preserve
observed timestamps and fill only missing entries when all observations
agree with the declared constant frame rate. Record repairs, decoder version,
source bytes, and exact processed-pixel hashes. This is a pilot-tested
proposal; complete dataset integrity and preprocessing lock remain pending.

The PGD implementation keeps the strongest attack-bank candidate per example
over clean input, starts, and all iterates. Independent masks are reserved
for scoring. The feasibility pilot uses FP32, microbatch 1, accumulation 8,
and explicit AdamW defaults; these do not silently fix the final precision
or monitoring/evaluation configuration. No cloud spending is authorized by
a pilot result alone; retain the existing measured cost gate.

## 2026-10-19 — Mac GPU checks and equivalent patch projection

The M4 Max exposes MPS with normal host GPU access. A restricted process
reported it unavailable; that observation did not establish a hardware
limitation. Keep CPU fallback disabled in GPU feasibility checks. The
CPU/MPS loss, target-feature, input-gradient, and selected parameter-gradient
comparison passed on one real training clip. This does not establish
cross-device equality of full training trajectories.

The original context Conv3d projection has an expensive input backward pass
on this MPS runtime. Add an explicit `--patch-embedding linear` option that
reshapes nonoverlapping patches and applies the same projection parameters.
It preserves state keys and parameter identity; the fixed target encoder
retains its original projection. CPU output/gradient tests and full-model
MPS diagnostics passed. PGD-5 on the checked clip fell from 121.29 seconds
to 4.84 seconds, with matching reported MPS attack losses and bounds.

Keep the default projection `conv3d`. The Mac commands explicitly select
`linear`, and reports record that choice. Any experiment using this option
must use it consistently across clean and adversarial conditions. No
upstream source, checkpoint, masking objective, or trainable component was
changed. Numerical reduction order may differ; do not describe this as
bitwise equivalence for arbitrary inputs or platforms.

The FP32, microbatch-one, accumulation-eight pilot is a hardware measurement.
It does not freeze the confirmatory backend, precision, validation subset,
or schedule. The [feasibility record](../report/checks/2026-10-19-mac-feasibility.md)
separates training from evaluation cost. Choose the monitoring schedule
before examining research outcomes; full training and resume support are
still pending. Protocol/configuration version 0.3 remains a draft.

## 2026-10-19 — Evaluation and compute proposal, not a protocol lock

Created a separate [compute plan](evaluation_compute_plan.md) with one
training/validation clip per class for monitoring, stronger primary attacks
at nine declared epochs plus initialization, and full 505-video final testing
at the pretrained and per-run selected/final checkpoints. Three seeds and
both conditions remain. The proposed best checkpoint is selected only by
the primary validation metric among those declared epochs; sparse primary
measurements cannot establish the best of all epochs. A denser alternative
is costed separately. Neither schedule is frozen or used for training.

The calculator retains 15 missing timing categories as unknown and labels
stronger-attack/epsilon estimates as proxies. It never approves a run. The
20-video-per-class proposal projects to 1,209.66 Mac hours from already-timed
operations, plus 483.60 hours of proxies and additional unknown work. A cloud
speedup must be measured; published hourly prices are insufficient evidence.

Reserve up to $10 for pilots within the existing $160 compute envelope,
retain the $40 noncompute reserve, and propose 25% runtime headroom when
pricing the measured full workload. These are planning allowances, not
authorization to launch an instance. Kaiwen confirmed an existing Lambda
account and chose it for the first pilot. The candidate is one 48 GB A6000;
its live quote and availability remain to be checked. Runpod's advertised
rate remains a comparison. No provider account was accessed and no CUDA
throughput was measured in this step.
The draft study file remains unchanged so prior measurement hashes remain
valid. Review scope and evidence before a new protocol version is locked.

## 2026-10-20 — Proposed staged scope, separate from protocol 0.3

The RTX 5090 measurements rule out the original larger schedule within $200.
Prepare a separate [staged proposal](budget_study_plan.md) and
[configuration](../configs/budget_study_plan.yaml), preserving the original
study hash and prior evidence. Recommend 20 classes with five actual training
videos each, one discovery pair, and three new confirmation pairs conditional
on signal validity and measured budget. Preserve the model, loss, 16-frame
256-pixel input, FP32 baseline, and primary PGD-20/R2/EOT4 attack. Narrow
claims to native-loss dynamics; defer direct classifier adversaries, embedding
attacks, and epsilon sweeps. Thirty small-subset epochs are not equivalent to
the original update exposure.

Use disjoint source groups for exploratory and confirmation validation.
The existing validation pool contains only one or two groups per class;
withhold one unused group per class from the parent training pool before
selecting actual training videos. Select classes/groups by identifiers only.
Keep source-group uncertainty and avoid treating three seeds on the same
videos as independent data replicates. Sparse strong-evaluation checkpoints
limit the selection claim to the declared anchors.

Account for the user-reported $0.592 Runpod bill inside the existing $10 pilot
reserve, not in addition to it. The $193.91 candidate envelope includes 25%
headroom, eight untimed hours of allowance, and $40 noncompute reserve. Unknown
costs and proxy attacks still block admission; no unmeasured speedup is assumed.
The proposal does not launch work, lock protocol 0.4, or admit confirmation.

## 2026-10-20 — Training recovery implementation, without study admission

Implement a reusable FP32 native training core and bounded diagnostic CLI.
Reuse the pilot update and preserve its attack settings. Define linear warmup
by optimizer-update position, apply LR drops in the epoch after each declared
boundary, and divide partial-batch loss by the actual number of examples.
Cache the upstream mask sampler once while testing parity against the old
sampler; use explicit epoch/video seeds shared by matched conditions.

Persist AdamW/RNG state and source/config/data identity, with a checksummed
atomic manifest and two rolling resume states. Keep fixed-target identity
without duplicating its weights. Preserve prior attempts when resuming and
require an explicit choice to recover an older generation. CPU tests verify
exact trajectory recovery; GPU equivalence remains untested. See the
[runner guide](training_runner.md). No protocol/configuration freeze or new
cost evidence follows from these implementation checks. Primary research
evaluation and total-cost admission must precede full trajectories.

## Change record template

```text
Date:
Protocol/config version:
Change:
Reason and evidence:
Affected runs:
Effect on comparison/claims:
```

## 2026-10-20 — Staged exports and native evaluation contract

Export the proposed 20-class identities separately, retaining the original
protocol/config and parent manifests. Prepare only the 100 actual training and
20 exploration clips now; confirmation/test pixels remain closed. Two decodes
and saved-cache reloads passed for all 120 clips, without exact hash duplicates.
Near-duplicate review and full protocol freezing remain pending.

Fix evaluation seed serialization explicitly: compact JSON, SHA-256 first
eight bytes modulo 2**63, then modulo 2**31-1 for the upstream signed counter.
Use one fixed attack bank across restarts, independent scoring banks, and fresh
perturbations at every checkpoint. Preserve training modes/RNG around evaluation.
Primary selection requires all declared validation anchors including initialization,
with earliest exact ties; routine PGD and test results cannot choose a checkpoint.

These are implementation details of the staged proposal, not a new admitted
study or changes to the historical pilot. The new CLI remains bounded and
diagnostic. Record them in the eventual frozen configuration after calibration,
full-model validation, and measured budget checks. See
[staged data and evaluation](staged_data_and_evaluation.md).

## 2026-10-20 — Probe convention and epoch evaluation barriers

Implement mean pooling of all clean context tokens and a separate ridge probe
for each representation checkpoint. Fix training-only population normalization,
FP64 solution, unpenalized intercept, and lambda one under **sum** squared error.
Measure variance and covariance-spectrum effective rank before normalization;
report their sample-size limit without declaring a collapse threshold.

Declare the first ten training-monitor manifest rows for eventual research
step calibration. Compare 0.5/1/2 per 255 using the common independent primary
scoring bank, highest mean then smaller exact tie. Keep training PGD-5 step one.
The current CLI's one/two-clip diagnostic is separately marked and cannot
admit or replace ten-clip research calibration.

Require evaluations to complete at saved epoch boundaries before further
training. Checksum-bound in-place continuation is explicit for the new
workflow command; reruns still require fresh directories, and the older
training CLI preserves its fresh-attempt behavior. Reuse shared initialization
only under identical evaluation contracts. Retain primary anchor weights and
reject altered schedules, calibration, source/runtime, data, or mixed receipts.

These details are implemented and locally tested, with a full pretrained CPU
feature check; full-model CUDA recovery/calibration remains pending. Record
these conventions in the eventual frozen configuration after validation and
cost admission. Preserve existing configuration hashes and prior evidence.
See [workflow diagnostics](workflow_diagnostics.md).
