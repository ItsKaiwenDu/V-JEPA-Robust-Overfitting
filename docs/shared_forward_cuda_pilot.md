# Shared-forward pretrained CUDA recovery pilot

**October 22 execution follow-up:** the [verified outcome](../report/checks/2026-10-22-shared-forward-cuda-workflow.md)
passes all eleven phases, both clean/adversarial two-update exact recovery
comparisons and final controls. The user terminated the Pod after local
export verification. Research admission and settled billing remain pending.
The procedure below records the executed diagnostic and its limits.

October 22, 2026. This is a fresh diagnostic package for the named
`shared-fp64-native` candidate, following the verified university input-only
control and [local optimizer integration](shared_forward_training.md).
It does not admit research trajectories. Previous failed CUDA/FP32 attempts
and their archives remain unchanged.

## What runs and what can pass

Use eight previously opened training clips from the existing staged cache.
No exploration-validation, confirmation or test pixels are included.
The unchanged source-group manifests, original ViT-L/16 checkpoint/predictor,
frozen clean target, native masked L1, raw-pixel 4/255 attacks, effective batch
eight and proposed FP32 AdamW settings remain explicit.

1. **Representative preflight**, at most 20 minutes: the first, middle and last
   of those eight training rows (indices 0, 4, 7), masks 0 and independent score
   masks 100000. Compare shared FP64 patch forward / FP32 tokens with strict
   cuDNN versus native FP32 backwards. Check independent FP64 patch output/
   input/weight/bias references, full loss/input gradients and chain-rule
   reconstruction using the original acceptance limits. Retain actual new
   native CUDA kernel evidence, clean targets, signatures, FGSM/PGD-5 tensors,
   bounds and independent scores. Cross-backend attack equality is descriptive;
   it is not a replacement for numerical gates. Any failure stops before
   calibration. This three-clip check is not comprehensive robustness evidence.
2. **Workflow**, using the remaining allowance: one-clip diagnostic calibration
   across the three primary PGD step sizes; eight-clip initial features/probe;
   two clean and two adversarial trajectories. For each condition, a split run
   saves update one, exits before evaluation, then resumes in a fresh process
   to update two; a separate uninterrupted reference also reaches update two.
   Compare exact weights, optimizer moments, history/sampler/mask/attack seeds,
   learning-rate counters, RNG and complete scheduled evaluation metrics.
3. **Controls and cost**: trained-anchor features/probe, convergence PGD-50/
   EOT-8, FGSM/random controls and extra independent scoring banks. Record
   synchronized setup/cache/update/evaluation/checkpoint/feature/I/O spans and
   allocator peaks. Nested timings overlap and must not be added.

Only complete numerical/replay/recovery evidence can pass. Timeout, OOM,
partial phases or missing kernel evidence remain failures/inconclusive
measurements. Nothing automatically relaxes tolerances or adopts a backend.
Full-study calibration, near-duplicate review, novelty recheck, precision/
protocol freezing and measured budget admission remain separate requirements.

## Package and instance

Local package: `runs/pretrained/diagnostic/shared-workflow-handoff-20261022/`.
Its manifest pins working-tree source, an exact upstream Git bundle, 100
training cache files and their metadata, and 23 existing Linux dependency
wheels. Only eight clips are selected for execution. The original release
checkpoint is separate and must match SHA-256
`5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`.

Use one RTX 5090 with 32 GB VRAM and the previously exercised image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Python 3.12,
PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128. Use a **30 GB container disk
and 80 GB persistent volume disk mounted at `/workspace`**, with at least
55 GiB free after setup. The volume leaves room for retained optimizer
states/anchors, temporary atomic checkpoint writes, inputs and evidence;
a 30-40 GB volume cannot meet that free-space gate.
The earlier university 8 GB full AdamW OOM is not
resolved by a passing input-only preflight. No GPU is launched by preparation.

On October 22 the user reported a current **$1.21/hour** console quote for
this exact configuration and explicitly retained both disk capacities.
Two billed hours at that quote are $2.42, leaving $0.58 within the $3 cap
for any charges outside the displayed rate. The user deployed this reviewed
configuration on October 22; current availability must be checked again
before any future session. The executed name was
`vjepa-recovery-pilot-03`. The planned
session allowance remains **at most two billed hours and $3 inclusive**, with
reported cumulative spend $1.195 and $8.805 left in the shared $10 pilot reserve.
Include GPU, storage, tax, transfer and idle time. Reduce time if the actual
all-in quote requires it; neither the historical $193.91 projection nor this
pilot admits the study.

Runpod documents that stopping clears the container disk while the volume disk
survives until Pod termination; stopped volume storage remains billable.
See [storage options](https://docs.runpod.io/pods/storage/types) and
[Pod management](https://docs.runpod.io/pods/manage-pods). Keep working assets
under `/workspace`. A stopped Pod does not reserve later GPU capacity.

## Restore and detached launch

After the user explicitly selects/deploys a suitable Pod, verify its host key
and current SSH endpoint. Transfer the complete package into a fresh location
under `/workspace`; do not reuse a terminated Pod's address or an old package.
Check the local manifest SHA-256 against the remote copy before restoration:

```bash
cd /workspace/shared-workflow-handoff-20261022
sha256sum manifest.json
python3 restore.py --package . --destination /workspace/V-JEPA-Robust-Overfitting
cd /workspace/V-JEPA-Robust-Overfitting
```

Run `scripts/start_shared_workflow.sh` with an absolute UTC epoch deadline
calculated from the **actual deployment time and reviewed budget**, not from
when Python starts. The exact number will be supplied when that session is
arranged. It verifies every restored source hash, then launches detached:

```text
bash scripts/start_shared_workflow.sh --deadline-epoch DEPLOYMENT_BASED_DEADLINE
```

The launcher has a 25-minute setup cap, up to 90 minutes for pilot workers,
a 20-minute preflight cap within that worker budget, and a four-minute export
reserve inside the absolute deadline. Less remaining time shortens setup and
pilot allowances. A final supervisor also enforces the deadline. No cloud API,
API key, package installation on the Mac, or automatic rental is involved.

Setup verifies the actual Linux/CUDA stack, creates an isolated environment
inheriting the template's CUDA packages, installs existing offline wheels,
runs the scientific suite, and verifies/downloads the original checkpoint.
The pilot starts only after setup succeeds. The deadline includes setup and
worker execution; it does not recover time already billed before launch.
The October 22 execution passed setup and all eleven CUDA phases; see the
[verified outcome](../report/checks/2026-10-22-shared-forward-cuda-workflow.md).
Any future execution requires its own setup, quote and bounded deadline.

The launcher prints the run directory, log command and final export path.
Use `tail -f` to watch; Ctrl-C exits the log viewer. SSH can be closed while
the detached job runs. Reconnect to the same Pod to inspect results.

## Completion or failure

The launcher retains atomic partial JSON and logs and attempts a bounded
export even after failure. Setup failure is explicitly marked as occurring
before pilot execution. The export includes source/setup/session/evaluation/
comparison/timing records and up to six representative tensor artifacts,
limited to 256 MiB each and 2 GiB total. Full optimizer/model checkpoint tensors
are excluded; they remain on the volume for investigation until evidence is
copied and verified.

**Stopping Python does not stop GPU billing.** Copy the evidence archive
and its manifest while the Pod remains accessible, then verify their hashes
and representative numerical/replay checks locally. Stop/terminate the Pod
after the required evidence is safely copied and verified. If export fails,
preserve the volume and inspect raw run/setup evidence before termination;
access to a stopped Pod requires a bounded restart.

These diagnostics can establish candidate correctness and measured feasibility.
They do not estimate robust overfitting, held-out accuracy or the study's final
cost from one small monitoring panel.
