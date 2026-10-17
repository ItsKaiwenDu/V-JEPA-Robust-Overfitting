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

`smoke_test.py` implements strict local V-JEPA 2 checkpoint loading and a
one-batch synthetic loss/gradient check. It requires an upstream checkout,
model dependencies, and an explicitly downloaded checkpoint. It writes a
new JSON report and refuses to overwrite an existing one. Full pretrained
CPU execution has passed. Run `python scripts/smoke_test.py --help` and
follow [the model-check guide](../docs/smoke_test.md).

`prepare_data.py`, `prepare_clips.py`, and `check_attacks.py` implement
grouped manifests, deterministic training/validation clips, and a one-clip
pretrained attack sanity check. See [data and attacks](../docs/data_and_attacks.md).
`pilot.py` is an explicit CUDA feasibility command; its unit-tested update
helper has not yet been measured on CUDA. See [the handoff](../docs/gpu_pilot.md).

## Upcoming entry points

| Name | Responsibility |
| --- | --- |
| `train.py` | Matched clean/adversarial native-loss fine-tuning |
| `evaluate.py` | Native-loss, attack-transfer, and downstream evaluation |
| `run_report_checks.py` | Gradient, projection, masking, and attack-strength diagnostics |
| `plot_results.py` | Per-seed curves, aggregate figures, and gap summaries |

The planned scripts are not created as nonfunctional placeholders. Add them
with implementation and meaningful checks. Preserve familiar condition,
seed, run-name, and diagnostic-mode concepts, with resolved configurations
saved alongside every run. Document actual command syntax when implemented.
