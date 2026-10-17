# Command-line scripts

Keep this interface familiar to the previous repository. Reusable model and
attack logic should be importable and shared, not duplicated across scripts.

## Available now

```bash
python scripts/verify_setup.py
python scripts/verify_setup.py --check-torch
python scripts/verify_setup.py --require-cuda
```

The first command checks scaffold files, directories, and configuration YAML.
The optional modes inspect an already installed PyTorch runtime. A missing
dependency or a required unavailable CUDA device produces a nonzero exit
code. These checks are read-only and do not establish model correctness.

## Upcoming entry points

| Name | Responsibility |
| --- | --- |
| `prepare_data.py` | Group-separated manifests, deterministic clips, cached clean targets |
| `train.py` | Matched clean/adversarial native-loss fine-tuning |
| `evaluate.py` | Native-loss, attack-transfer, and downstream evaluation |
| `run_report_checks.py` | Gradient, projection, masking, and attack-strength diagnostics |
| `plot_results.py` | Per-seed curves, aggregate figures, and gap summaries |

The planned scripts are not created as nonfunctional placeholders. Add them
with implementation and meaningful checks. Preserve familiar condition,
seed, run-name, and diagnostic-mode concepts, with resolved configurations
saved alongside every run. Document actual command syntax when implemented.
