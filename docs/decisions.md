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

## Change record template

```text
Date:
Protocol/config version:
Change:
Reason and evidence:
Affected runs:
Effect on comparison/claims:
```
