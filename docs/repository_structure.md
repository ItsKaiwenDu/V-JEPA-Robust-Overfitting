# Repository structure

The organization follows [Robust-Overfitting](https://github.com/ItsKaiwenDu/Robust-Overfitting)
while giving video data, native prediction losses, and experiment provenance
explicit homes.

| Familiar location | Role in this project |
| --- | --- |
| `README.md` | Overview, status, setup, design, and navigation |
| `goals.md` | Weekly objectives, expectations, and deliverables |
| `progress.md` | Dated execution notes with evidence and next steps |
| `setup_lambda_labs.md` | Cloud workflow and measured spending |
| `models/` | V-JEPA integration, clean target handling, native loss |
| `scripts/` | Verification, then preparation/training/evaluation/plotting entry points |
| `notes/` | Literature summaries and dated novelty checks |
| `report/` | Paper source, curated measurements, figures, and summaries |
| `assets/` | Small documentation visuals |
| `data/`, `checkpoints/`, `runs/`, `run_logs/` | Local experiment assets; large outputs ignored |

New locations:

| Location | Why it is useful |
| --- | --- |
| `configs/` | One readable draft study specification; local paths separated |
| `attacks/` | Raw-pixel FGSM/PGD and native-loss objectives with fixed mask banks |
| `datasets/` | Grouped UCF101 manifests and deterministic video preprocessing |
| `docs/` | Detailed protocol and reproducibility guidance without bloating the README |
| `data/manifests/` | Reviewable, relative-path split definitions and group membership |
| `external/` | A separately pinned Meta checkout with its own license |
| `tests/` | CPU checks for integration, attacks, data separation, decoding, and pilot updates |
| `.github/` | Lightweight repository checks and experiment/review templates |

Reusable code lives in `models/`, `attacks/`, and `datasets/`; commands live
in `scripts/`. The CUDA feasibility pilot is implemented, while the full
training and checkpoint-evaluation runners remain upcoming work.

## Result organization

Replace the old pixel/low-frequency/mixed conditions with `pretrained`,
`clean-finetune`, and `adversarial-finetune`. The condition/seed and
`diagnostic/` conventions remain familiar:

```text
report/
  pretrained/reference/
  clean-finetune/seed-0/
  clean-finetune/overall/
  adversarial-finetune/diagnostic/seed-0/
  adversarial-finetune/seed-0/
  adversarial-finetune/overall/
  checks/
```

The run-specific directories above are created by future experiments, not
prepopulated with pretend outputs. Keep unreviewed/high-volume output in
`runs/`, and promote small verified CSVs, plots, and summaries to `report/`.

`.gitkeep` files preserve empty directory locations in a fresh clone. They
do not indicate that data, results, or checkpoints exist.
