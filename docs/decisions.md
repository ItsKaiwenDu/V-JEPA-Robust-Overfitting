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

- Full checkpoint hash and successful strict loading/execution; the source
  revision and checkpoint metadata keys/shapes have been inspected.
- Deterministic temporal sampling, resizing, decoding, and short-clip policy.
- GPU microbatch, accumulation, precision, and optimizer implementation details.
- Evaluation step sizes, fixed subsets, attack/scoring mask banks, and probes.
- Measured end-to-end cost and whether 20 or 10 training videos per class fit.

Resolve these before locking the protocol, without using final test outcomes.

## Change record template

```text
Date:
Protocol/config version:
Change:
Reason and evidence:
Affected runs:
Effect on comparison/claims:
```
