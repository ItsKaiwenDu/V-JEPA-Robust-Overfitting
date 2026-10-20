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
