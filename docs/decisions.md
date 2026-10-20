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

## Change record template

```text
Date:
Protocol/config version:
Change:
Reason and evidence:
Affected runs:
Effect on comparison/claims:
```
