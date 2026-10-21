# Bounded CUDA recovery and measurement pilot

Prepared October 21, 2026 from milestone `2e04959`. This is the next diagnostic
handoff, not permission to run the proposed research study. The earlier
[October 19 handoff](cloud_pilot_handoff.md) measures a different workflow and
must not be substituted for this one.

## Scope and acceptance

The [runner](../scripts/run_workflow_pilot.py) uses eight existing training clips
and one declared training-monitor clip, with the original FP32 Conv3d model,
frozen clean target, effective batch eight, and existing native attacks. It:

1. Runs a **one-clip diagnostic** of the three primary calibration candidates.
   This does not replace the required ten-clip study calibration.
2. Extracts initial clean features and fits/scores the fixed ridge probe on
   the same eight training examples. This is not held-out accuracy.
3. For each condition, saves update one, exits before that epoch's evaluation,
   resumes in a fresh worker process to update two, and runs a separate
   uninterrupted two-update reference. Eight updates total across four
   trajectories. All scheduled evaluations through epoch two must complete.
4. Compares exact weights, optimizer moments, history, RNG, contracts, and
   evaluation metrics. It ignores only evaluation duration and serialization
   file hashes after validating their receipts. Any mismatch stops execution;
   tolerances and precision are not changed automatically.
5. Measures features/probe fitting at the saved adversarial epoch-one anchor,
   plus pretrained PGD-50/EOT-8 convergence, FGSM/random controls, and two
   additional independent scoring banks for the convergence diagnostic.

The interruption is a controlled process exit at a committed boundary. It
does not simulate an arbitrary mid-kernel crash or a provider outage. Existing
unit tests cover other checkpoint/evaluation interruption points.

[Measurements](../models/pilot_measurement.py) record setup, target caching,
updates, CPU state copies, checkpoint writes/reads/restoration, evaluations,
feature extraction, probe fitting/scoring/storage, and worker wall time.
CUDA spans synchronize the device. **Nested spans overlap: do not sum them.**
Cached evaluation reuse reports zero newly evaluated clips/native seconds.
Small-panel timings require cautious extrapolation; full-panel I/O, full probe
costs, near-duplicate review, novelty refresh, research calibration, protocol
freezing, and complete budget admission remain outstanding.

## Instance and spending boundary

Use one RTX 5090 (32 GB), the previously exercised Runpod PyTorch 2.8.0 image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Python 3.12, and CUDA torch
`2.8.0+cu128` / torchvision `0.23.0+cu128`. Setup checks the actual environment.
Other GPUs/templates need a deliberate compatibility review.

Use a 30 GB container disk and **80 GB persistent volume mounted at
`/workspace`**. The runner requires at least 55 GiB free after setup. Four
trajectories can retain roughly 36 GB of rolling optimizer states and anchor
weights, plus a temporary atomic write, the 5.13 GB release, 1.26 GB training
cache, environment, and logs. This pilot therefore needs more disk than the
previous short feasibility check.

Prepare locally before deployment. Check the current console quote; the
October 19 price is historical, not a live offer. Plan **at most two billed
hours and $3 for this session**, inclusive of setup, transfer, idle time,
storage, and applicable tax. Shorten the session if the current all-in quote
requires it. The prior $0.592 reported spend leaves $9.408 of the shared $10
pilot reserve; the $200 total/$160 compute caps remain unchanged.

The runner has a maximum 90-minute worker deadline, leaving a planned
30-minute allowance for setup/export within two billed hours. Track time from
**Pod deployment**, not from Python startup. If setup consumes that allowance,
reduce `--max-seconds` or stop. A worker timeout stops Python, **not Runpod
billing**. No script deploys, queues, stops, or terminates cloud resources.

Runpod says container data is lost on stop; volume-disk data survives a stop
but is deleted with the Pod. Stopped volume storage remains billable.
See [Runpod storage documentation](https://docs.runpod.io/pods/storage/types).
Export and verify evidence on the Mac before termination. A stopped Pod does
not guarantee GPU capacity for a later restart; see
[Pod lifecycle documentation](https://docs.runpod.io/pods/manage-pods). Do not leave an auto-deploy
request unattended.

## Local handoff

[prepare_workflow_pilot.py](../scripts/prepare_workflow_pilot.py) packages a
checksummed working-tree source snapshot, an exact upstream Git bundle,
Linux Python 3.12 wheels, and the original training-cache index/preparation
record plus all 100 training `.pt` files. The existing loader checks all
training paths before selecting eight diagnostic examples, so the full train
cache is included. No raw videos or exploration/confirmation/test pixels are
included. Held-out CSV metadata stays in the source snapshot.

The release checkpoint is separately verified and downloaded on the Pod;
the existing local copy can be uploaded if that endpoint stalls. GitHub/PyPI
are not required on the Pod. Exact source and per-file hashes are recorded,
including explicitly listed new files not yet committed. Package archives,
wheels, model weights, and raw run output remain ignored by Git.

The prepared local package is under
`runs/pretrained/diagnostic/workflow-handoff-20261021/`. Its manifest and
curated [preparation report](../report/checks/2026-10-21-cuda-handoff.json)
record exactly what was verified. An updated source tree requires a new
package and output name; do not silently reuse stale archives.

## On an explicitly selected, running Pod

Transfer the package to `/workspace/workflow-handoff-20261021`, then use its
standard-library restorer. Compare the manifest SHA-256 against the local
copy before running it. Use the existing verified SSH workflow and the new
Pod's current connection details; old terminated-Pod addresses are obsolete.

```bash
cd /workspace/workflow-handoff-20261021
sha256sum manifest.json
python3 restore.py --package . --destination /workspace/V-JEPA-Robust-Overfitting
cd /workspace/V-JEPA-Robust-Overfitting
bash scripts/setup_workflow_pilot.sh
```

Setup verifies the template stack, creates an isolated environment inheriting
its CUDA packages, installs pinned offline dependencies, runs setup checks and
the test suite, and verifies the release checkpoint. It does not train.
Use the available SSH session or a detached `tmux` session for execution;
disconnecting SSH does not stop a detached job or its cloud billing.

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python -u scripts/run_workflow_pilot.py --diagnostic \
  --checkpoint checkpoints/vitl.pt \
  --manifests data/manifests/ucf101-staged-20x5-v1 \
  --clips data/processed/staged-exploration-v1 \
  --output runs/pretrained/diagnostic/cuda-workflow-20261021 \
  --max-seconds 5400
```

Use a smaller deadline if setup took longer than planned. The top-level
session is intentionally fresh-only. A failed session keeps evidence and
checkpoints; inspect its cause and remaining budget before retrying. The
internal split/resume phases explicitly exercise recovery in one session.
No research trajectory is admitted by a successful diagnostic.

## Export and finish

This export works for completed or failed sessions. It includes small JSON,
timing journals, setup package/GPU information, test XML, and worker logs.
Large model, optimizer, feature, and pixel tensors are excluded.

```bash
.venv/bin/python scripts/export_workflow_pilot.py \
  --run runs/pretrained/diagnostic/cuda-workflow-20261021 \
  --output /workspace/workflow-evidence-20261021.tar.gz
```

Copy the archive and its `.manifest.json` sidecar to the Mac, compare the
archive hash, and inspect completion/failure and both recovery comparisons.
Then stop/terminate the Pod through the console and record the actual charge.
Termination deletes its volume-disk checkpoints; retain the Pod only if those
are needed for diagnosing a failure and the remaining storage budget allows.
