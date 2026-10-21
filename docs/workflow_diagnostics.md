# Feature probes, calibration, and scheduled recovery

The [October 21 CUDA workflow outcome](../report/checks/2026-10-21-cuda-workflow.md)
records a failed exact full-model recovery comparison and a strict deterministic
retry stopped for excessive runtime. Small CUDA replay passes; full-model
recovery remains unvalidated. The [university component profile](../report/checks/2026-10-21-university-cuda-components.md)
now isolates a strict Conv3d input-backward bottleneck on RTX 3070 and default
attention-gradient replay failures. The [TF32-off control](../report/checks/2026-10-21-university-cuda-convolution-precision.md)
now qualifies native patch-convolution execution with numerical agreement and
observed replay. Validate an opt-in full-model integration before another rental. The [workflow guide](cuda_workflow_pilot.md) describes
the preserved handoff and explicit runtime policy.

October 20, 2026. These components implement the next local diagnostic milestone.
They do not admit the proposed study or open confirmation/test data. The original
study and proposal YAML hashes are unchanged; details below must be carried into
the eventual frozen configuration before research trajectories begin.

## Clean features and fixed probe

[features.py](../models/features.py) extracts the context encoder's clean,
unmasked final tokens and averages all space/time tokens into one vector per
video. It uses the existing pixel normalization and FP32 model, in evaluation
mode without gradients. It restores model modes and Python/NumPy/PyTorch RNG,
and checks all three component digests before and after extraction.

For each representation checkpoint, fit a separate ridge classifier using
actual training features and labels only. Center each feature dimension using
its training mean and scale by its training population standard deviation.
Dimensions with standard deviation at most `1e-12` use scale one. Keep the
original class IDs; validation cannot change classes, normalization, or lambda.
Solve in FP64 on CPU, with an unpenalized intercept and this exact convention:

```text
minimize sum_over_training_examples ||Z W + b - one_hot(y)||² + 1.0 * ||W||²
```

The sum convention matters: lambda one with a mean-squared loss is a different
regularizer. The dual solution is checked against an independent primal solve.
Source, runtime, recipe, encoder, and target identities must match when scoring.
Held-out groups cannot overlap probe training. Training-monitor rows must be
actual training members. Confirmation/test scoring is currently refused.

Compute population variance and covariance-spectrum entropy effective rank on
raw features **before standardization**. The latter is `exp(-sum(p log p))`,
where `p` is normalized squared singular-value energy after centering. Discard
singular values below the standard FP64 numerical-rank tolerance. Report zero
for a constant representation; report the sample rank bound `min(N-1, D)`.
At least two examples are required for these diagnostics. These measurements
have no automatic collapse threshold and do not replace downstream evaluation.

The released ViT-L checkpoint passed a CPU feature/probe command on eight real
training clips from two classes: finite `8 x 1024` features, centered rank 7,
and unchanged model/target weights. Extraction took about 23.22 seconds in this
local check, excluding setup. The probe was scored on its own eight training
examples; this is a plumbing check, **not held-out accuracy or a research result**.
Feature/probe tensors remain ignored; small evidence is in the
[workflow check report](../report/checks/2026-10-20-workflow-checks.json).

## Training-only calibration

[calibration.py](../models/calibration.py) compares primary PGD-20, two restarts,
four attack masks and four independent scoring masks at step sizes 0.5, 1, and
2 in units of 1/255. It uses the same pretrained weights, ordered training rows,
seeds, runtime, and source identity for every candidate. Choose the highest
mean independent-mask loss; choose the smaller step on an exact tie. Attack-bank
loss selects each adversary, but cannot select the calibration hyperparameter.

The proposed research calibration uses the **first ten rows of the declared
training-monitor manifest**, fixed before any trajectory. Its reusable core
requires all three complete evaluations and records their checksums. A bounded
one/two-clip diagnostic is marked separately and cannot be promoted to research
calibration. PGD-5 training retains step size 1/255. No validation/test metric
chooses this hyperparameter. Full-model calibration has not yet been executed;
the actual study step size is still provisional.

## Epoch boundaries and recovery

[schedule.py](../models/schedule.py) saves initialization and every completed
epoch before evaluation. Routine evaluation occurs at initialization and each
epoch; primary evaluation occurs at initialization and the declared anchors
(proposed epochs 1, 15, 16, 23, 24, 30). Primary anchor weights remain available
after rolling optimizer checkpoints are retired.

A checksummed `workflow.json` records the ordered completed evaluations and
their artifact hashes. Training cannot advance past an unfinished evaluation
boundary. If an evaluation fails, its existing per-clip journal resumes at the
first incomplete clip. If it finishes before the workflow-record write fails,
recovery reuses the completed evaluation. Matched conditions can share the
pretrained reference only when every evaluation contract matches; later
checkpoints generate fresh adversaries. Evaluation preserves the training RNG.

The new diagnostic uses explicit `--resume` **inside the same trajectory
directory**, preserving its checkpoint and evaluation history. A changed
condition, seed, schedule, data, calibration, source, or runtime is rejected.
Reruns require new directories. Saved receipt paths are absolute, so recovery
currently requires the same artifact paths; relocation is not implemented.
The older `train.py --resume-from` fresh-attempt behavior is unchanged.

CPU tests cover interruption inside a panel, after an epoch checkpoint, and
between evaluation completion and workflow-record commit. Clean and adversarial
trajectories match uninterrupted training exactly, including optimizer/history
and RNG. This is small-model CPU evidence; full-model CUDA recovery is pending.

## Bounded command

[check_workflow.py](../scripts/check_workflow.py) accepts `features`, `calibrate`,
or `train`, always with `--diagnostic`. It verifies staged cache bytes and uses
eight training clips: the first two declared monitors, filled to eight in
training-manifest order. One or two of those monitors are evaluated. No held-out
pixels are loaded by this command. Training keeps the proposed optimizer and
30-epoch schedule; each invocation allows at most eight additional updates and
16 panel evaluations. This reduced diagnostic cannot estimate the study effect.

The CPU command already executed was:

```bash
.venv/bin/python scripts/check_workflow.py features --diagnostic \
  --checkpoint checkpoints/vitl.pt \
  --manifests data/manifests/ucf101-staged-20x5-v1 \
  --clips data/processed/staged-exploration-v1 --device cpu \
  --output runs/pretrained/diagnostic/workflow-features-20261020
```

Future CUDA diagnostic commands, after a bounded pilot is arranged:

```bash
.venv/bin/python scripts/check_workflow.py calibrate --diagnostic \
  --checkpoint checkpoints/vitl.pt \
  --manifests data/manifests/ucf101-staged-20x5-v1 \
  --clips data/processed/staged-exploration-v1 --device cuda --limit 1 \
  --output runs/pretrained/diagnostic/calibration-01

.venv/bin/python scripts/check_workflow.py train --diagnostic \
  --checkpoint checkpoints/vitl.pt \
  --manifests data/manifests/ucf101-staged-20x5-v1 \
  --clips data/processed/staged-exploration-v1 --device cuda --limit 1 \
  --calibration runs/pretrained/diagnostic/calibration-01 \
  --evaluations runs/pretrained/diagnostic/scheduled-evaluations-01 \
  --condition adversarial-finetune --seed 0 \
  --max-updates 1 --max-evaluations 4 --max-seconds 600 \
  --output runs/adversarial-finetune/diagnostic/seed-0-workflow-01
```

Repeat the training command with `--resume` to add one update. Use a fresh
output with `--max-updates 2 --max-evaluations 5` for an uninterrupted reference.
Compare saved states, rather than inferring CUDA recovery from exit status.
Run the corresponding clean condition in its own directory. Optional
`features --anchor-store <trajectory>/checkpoints --anchor-epoch 1` loads only
a verified saved anchor and refits the diagnostic probe for that encoder.

The time limit is checked between full updates/panels and excludes setup; it
can overrun during an active operation. Calibration has a finite clip/candidate
bound, not a wall-time cap. Neither command stops a cloud instance or billing.

## Remaining research gates

Measure full-model GPU recovery, calibration, convergence attacks, feature
extraction/probe work, target-cache creation, checkpoint I/O and storage, and
transfer/idle costs. Keep the reported $0.592 inside the total $10 pilot reserve
($9.408 remained reserved at the October 20 preparation checkpoint), and the
$200 total/$160 compute caps unchanged. The latest user-corrected October 21
ledger is $1.195 spent / $8.805 pilot reserve remaining; see the
[workflow outcome](../report/checks/2026-10-21-cuda-workflow.md). The
old $193.91 projection still contains unknown costs and is not admission.

Near-duplicate content review, novelty refresh, full-model validation,
protocol/configuration freezing, and a current quote with a measured remaining
budget precede the exploratory research pair. Confirmation and final-test
analysis remain closed. No new cloud instance or paid compute was used here.
