# Report and curated results

[main.tex](main.tex) is a LaTeX protocol/report outline, with references in
[references.bib](references.bib). It contains no results and has not yet been
compiled. A TeX installation is optional for repository setup.

When TeX is available, compile from this directory:

```bash
latexmk -pdf main.tex
```

## Result directories

- `pretrained/`: the common initialization reference.
- `clean-finetune/`: per-seed and aggregate clean adaptation results.
- `adversarial-finetune/`: per-seed and aggregate native-loss PGD results.
- `checks/`: implementation and attack-strength evidence.
- `figures/` and `tables/`: cross-condition, publication-ready assets.

The condition directories remain placeholders; `checks/` contains curated
implementation checks and [Mac feasibility evidence](checks/2026-10-19-mac-feasibility.md).
The [compute projection](checks/2026-10-19-compute-projection.json) is a
partial planning calculation with explicitly unmeasured costs; it is not a
cloud benchmark or research result. Its assumptions are in the
[evaluation proposal](../docs/evaluation_compute_plan.md).
There are no robust-overfitting findings yet. Use the condition/seed structure
described in [reproducibility](../docs/reproducibility.md) when actual runs
exist. Add small, verified CSVs, vector figures, and summaries here; keep raw
logs, per-video caches, and weights in their ignored storage directories.

## Planned figures

1. Clean/adversarial train and validation loss versus epoch, per seed.
2. Validation-selected best versus final held-out performance.
3. Attack-strength, epsilon, and mask-transfer comparisons.
4. Downstream probe performance and representation-collapse checks.

Show uncertainty and failed/inconclusive findings. Do not create illustrative
curves that could be mistaken for measurements. The pretrained reference has
no training trajectory.

The [October 22 shared-forward integration check](checks/2026-10-22-shared-forward-integration.md)
records exact small-upstream CPU optimizer/workflow recovery and artifact
precision checks. It is implementation evidence, not pretrained CUDA recovery
or a robust-overfitting result.

The [fresh shared-forward handoff](checks/2026-10-22-shared-workflow-handoff.md) records package/member/
upstream/checkpoint/data/wheel verification and extracted CPU fixture results.
Its execution is now recorded in the [verified CUDA outcome](checks/2026-10-22-shared-forward-cuda-workflow.md):
436 host tests, representative numerical checks, exact bounded clean/adversarial
recovery and final controls pass. Research findings and study admission remain pending.
