# Lambda Labs Setup Guide

This guide retains the role of the same-named file in the previous project.
The current repository has no training entry point yet. Use the university
GPU and [local setup guide](docs/setup.md) for initial development.

## Budget and rental choice

The project cap is **$200 total**: $160 planned compute plus $40 for taxes,
storage, and contingency. Count idle time and evaluation as well as training.

As checked on October 16, 2026, Lambda's [official instance page](https://lambda.ai/instances)
listed a single RTX A6000 (48 GB) at $1.09/hour and a single A100 (40 GB) at
$1.99/hour, before applicable taxes. Recheck price and availability at launch.
The $160 compute allowance would buy about 146.8 or 80.4 hours at those rates;
these are arithmetic limits, not measured experiment runtimes.

Start with the A6000 candidate if available and suitable. Select a different
GPU only after estimating the whole experiment's cost from measured
throughput. A lower hourly price alone does not establish lower total cost.

## Prepare before renting

1. Complete local repository checks and prepare code/data access.
2. Confirm where outputs will be copied and how the instance will be terminated.
3. Set a planned runtime ceiling and a billing reminder appropriate to the pilot.
4. Keep SSH keys and account tokens outside the repository.
5. Review the [protocol feasibility gate](docs/experimental_protocol.md#feasibility-gates).

## Connect and inspect

Create the selected instance through the provider's interface when ready.
Follow the provider's displayed SSH command; do not put private keys in Git.
Record the actual GPU, driver, and Python environment:

```bash
nvidia-smi
python3 --version
```

Transfer the repository using its eventual Git remote or your normal SSH
file transfer workflow. Configure a compatible CUDA environment following
[the setup guide](docs/setup.md). Then run:

```bash
python scripts/verify_setup.py --require-cuda
```

This only confirms the scaffold and visible device. The upcoming native-loss
smoke test must confirm model loading, decoding, forward/backward passes,
target freezing, and attack gradients.

## Pilot and admission

Measure representative training steps and every required evaluation mode,
including EOT masks and attack restarts. Include feature preparation,
checkpoint writes, transfer time, and any startup/idle time in the projection.
Save the estimate with run provenance and update [progress.md](progress.md).

Admit the six main runs only if training plus required evaluation fits the
remaining allowance. The first fallback is the predeclared 10-training-videos
per-class subset. Preserve three seeds and the validation/test protocol.
If that fails the budget gate, revise scope before further spending.

## Running jobs

Use the established scheduler or a persistent terminal session for the future
training commands. Store console logs in `run_logs/`, raw run records in
`runs/`, and weights in `checkpoints/`. Track the condition, seed, run name,
protocol version, elapsed time, and expected cost. No launch or background
training commands are provided until those scripts actually exist.

## Export and shut down

Copy and verify required checkpoints, resolved configs, logs, and measurements
before terminating the instance; do not assume its local disk survives.
Promote small reviewed outputs into `report/` and keep large artifacts outside
Git. Terminate idle rentals through the provider and verify that ongoing
billing has stopped. Record final billed compute and any other costs in
[progress.md](progress.md).
