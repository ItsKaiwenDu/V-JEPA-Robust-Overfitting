# Lower-cost GPU benchmark for the retained 20-class study

**Latest offline review (October23):** [the completed review](../report/checks/2026-10-23-offline-a40-review.md)
attributes about85% of timed work to evaluation and preserves the20-class design.
The saved strong-attack contrast does not isolate steps from EOT/start effects.
Prepare the separate [matched diagnostic](matched_attack_diagnostic_plan.md)
before a further paid benchmark. Its status is designed, not implemented/run.
Hypothetical rates are affordability targets, not live offers or GPU forecasts.

**Latest offline preparation (October23):** the [controls-only handoff](hardware_controls_completion.md)
reuses the six verified A40 phases and saved diagnostic calibration. Local100
targeted and33 extracted operational checks pass; no new CUDA timings or rental.
Fresh quote/host review comes next. ProposedNEW$2.00 inclusive hold fits the
reconciled$4.53 diagnostic allowance and is not applied; historical holds are
settled. Both prior Pods stay terminated. The old timeout remains failed and
the eventual combined cost screen is a separate record. Research is unadmitted.

**Current outcome:** the [migrated A40 screen](../report/checks/2026-10-23-a40-migrated-continuation.md)
completed six of seven phases; the control phase timed out. Both Pods are
user-confirmed terminated and evidence is copied/verified. There is no complete
A40 cost projection or research admission. The [reconciled billing ledger](../report/checks/2026-10-23-runpod-billing-reconciliation.md)
records $5.47 cumulative credit used and $9.53 wallet credit. Internal prior
session reservations are settled; $4.53 conservatively remains in the shared
$10 diagnostic allowance. Fresh budget metadata is required before a rental.
The setup, active-host and launch-quote descriptions
below are historical snapshots, not instructions to restart terminated hosts.

Current October 23 A40 retry: the [portability controls](../report/checks/2026-10-23-a40-runtime-portability.md)
record 507 passes/two rejected harness checks from the first setup. The separately
pinned two-test repair passes both same-host controls; the complete 509-case suite
now passes with zero skips on the same Pod. Same $0.61/hour quote, existing $2.80 hold
and 2:58 PM Pacific absolute cutoff. No new hold or deadline reset. Billing
remains active; pretrained economics and study admission remain pending.
The preparation and earlier outcomes below are historical snapshots.

Latest October 23 follow-up: the [university CPU retry](../report/checks/2026-10-23-university-cpu-recovery.md)
is verified: 165 passed, zero skips, exit 0. Four saved CPU recoveries match
exactly and the actual NFS4 probe confirms the checkpoint-lock repair.
The [refreshed GPU package](../report/checks/2026-10-23-hardware-benchmark-refresh.md)
is restored and verified, with six extracted CPU checks passing. Its proposed
$2.80 hold fits the conservative $2.805 available balance. Check a fresh quote
before deployment; do not reuse the October 22 source archive. A40 performance
remains unmeasured.

**October 23 outcome:** the user-deployed A40 at $0.61/hour all-in passed
hardware/transfer checks, then setup rejected generated CPU replay/recovery
tests. The pretrained screen did not start. Both evidence archives verify
locally and the user confirmed termination; final charge remains pending.
The subsequent CPU repair is verified above. See the
[verified rejection](../report/checks/2026-10-23-a40-setup-rejection.md).
The preparation and proposed launch procedure below retain their original scope.

October 22, 2026. Kaiwen chose to **keep the 20-class, 100-training-clip design**
and prepare a lower-cost hardware benchmark. This is an offline-prepared
performance screen, not a rental or an admitted research run.

The [measured review](../report/checks/2026-10-22-measured-budget-review.md)
puts the existing schedule at $304–$307 on the exercised $1.21/hour RTX 5090.
Preserve the original checkpoint, fixed clean target, masked native L1,
4/255 threat model, attack strengths, seed pairs, primary anchors and data
identities. No class reduction, weakened evaluation or study YAML change.

## Candidate order and quotes

Start with **one RTX A5000 on Runpod**, using the existing account and image.
It has the lowest public compute price among the compatible candidates below;
its workload speed and memory fit still require measurement. If unavailable,
check RTX A6000 or A40. Public catalog prices checked October 22:

| Candidate | VRAM listed | GPU price/hour |
| --- | ---: | ---: |
| RTX A5000 | 24 GB | $0.27 |
| RTX 3090 | 24 GB | $0.50 |
| A40 | 48 GB | $0.59 |
| RTX A6000 | 48 GB | $0.59 |

Source: [Runpod Pod pricing](https://www.runpod.io/pricing), whose displayed
update date is October 8. These are public GPU-only listings, not deployable
capacity, the user's cloud-tier selection, disk/tax-inclusive quotes, or
evidence of our workload performance. Obtain a new console quote before a
session. Prefer the existing on-demand workflow; do not create an automatic
deployment subscription while the session is unattended.

Vast is a possible later marketplace comparison. Its
[official pricing documentation](https://github.com/vast-ai/docs/blob/main/guides/pricing.mdx)
describes host-set compute, continuous storage and per-byte bandwidth charges.
A marketplace listing is not an all-in quote or a reliability guarantee. No
Vast account, funds, host, runtime image or instance is configured here.

The prepared screen accepts only an explicitly reviewed **all-in running
rate at or below $0.70/hour**, one allowlisted visible GPU, and the same
Python 3.12 / torch 2.8.0+cu128 / torchvision 0.23.0+cu128 stack. Exact expected
GPU name and available VRAM must match; MIG and unreviewed replacements reject.
This price screen prioritizes plausible candidates; it cannot establish that
a slower GPU fits the final budget. The observed allocated-memory peak is
not a total-VRAM requirement, so every candidate must pass actual execution.

Keep **30 GB container and 80 GB persistent `/workspace` storage**, with at
least 55 GiB free after inputs/setup. Report host RAM in the quote; low host
RAM may also prevent loading, state creation or export even when GPU VRAM fits.
Use the previously exercised image:
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

## Seven-phase screen

The new runner reuses the validated numerical and scientific implementations:

1. Three-input preflight at training indices 0, 4, 7, both shared-forward
   backwards, unchanged oracle/full-gradient limits, FGSM/PGD-5, independent
   scoring and new actual native CUDA kernel evidence. Failure stops timing.
2. One-training-monitor three-candidate diagnostic calibration.
3. Initial eight-training-clip features and fixed diagnostic probe.
4. Two clean reference updates, with the existing scheduled evaluations.
5. Two adversarial reference updates with those evaluations.
6. Features/probe from the verified epoch-one adversarial **reference** anchor.
7. PGD-50/R2/EOT8 with both extra scoring banks, plus FGSM/random controls.

Only eight previously opened training clips and one of their monitors execute.
The package retains the existing complete 100-training-clip cache and metadata;
it contains no held-out pixel cache. Calibration is diagnostic, not the
required ten-monitor study calibration. The feature score is training
resubstitution, not held-out accuracy.

The screen omits the split/resume trajectories deliberately and records
`exact_optimizer_recovery_checked: false`. A completed screen establishes
neither new-hardware exact recovery nor research admission. The original
eleven-phase RTX 5090 recovery evidence and its rejected FP32 predecessors
retain their original scope. A promising GPU needs a separately bounded full
recovery check and the remaining research/data/protocol gates.

## Cost calculation and decisions

Fresh per-operation observations are applied to the **same counts** in the
measured review: 390 updates/run, eight runs, separate validation panels and
up to ten final-test checkpoints. Report both median and maximum-observed
units, plus the historical eight-hour allowance, 25% headroom, full $10 pilot
reserve and $40 noncompute reserve. Neither scenario is a full-study bound.

Clean scores and the two convergence audit banks are already inside their
evaluation times. Count disjoint state/save spans once; exclude enclosing
workflow totals. Missing, failed, reused-only or nonfinite unit timings cannot
become zero. The primary summary explicitly includes calibration-candidate
records, with sample sizes recorded; it does not claim an already frozen
research step size. Actual billed setup/transfer/idle time remains separate.

If the maximum-observed scenario still exceeds either cap, reject that
candidate for this workload before another paid acceptance attempt. If the
arithmetic fits, call it a **promising cost screen**, with exact recovery,
full-panel I/O, feature/probe work, storage/export, calibration and settled
charges still pending. Do not start exploration merely because this screen
finishes or the GPU has a cheap hourly price.

## Offline handoff and launch boundary

Current quote follow-up (October 23): the user reports an A40 available at
$0.61/hour all-in with the requested 30/80 GB disks. Two running hours are
$1.22 at that quote. Match the previous roughly 50 GB system RAM; report a
substantially lower amount before deployment. No Pod has been deployed yet.
Use `vjepa-cost-benchmark-05`, then provide its new SSH/public fingerprint and
deployment time so the absolute deadline starts at actual deployment.

Fresh local package:
`runs/pretrained/diagnostic/hardware-benchmark-handoff-20261023/`.
The October 22 package is historical and retains its original checksums.
Its manifest pins source files, the original upstream Git objects, training
cache files and existing non-torch Linux wheels. The checkpoint is downloaded
only on the selected host and must match the original pinned SHA-256. Earlier
handoff archives are not replaced. No private key, API token or model weights
are included in source archives.

The next retry is **at most two billed hours and $2.80 inclusive**. Both
earlier $3 session holds remain unresolved, leaving a conservative $2.805
unreserved pilot balance. A proposed $2.80 hold fits that balance; it becomes
a reservation only if the next session is deployed. The launcher rejects a
reservation larger than the available balance or smaller than two hours at
the reviewed rate. It retains the existing $0.70/hour quote ceiling and
manual deployment boundary. This lowers the old $3 maximum and does not add
a charge or release an earlier hold. The displayed $3.930 aggregate remains
unreconciled by Pod; do not add earlier partial summaries to it. Refresh this
ledger if spending changes before deployment.

Once the quote and availability are reviewed, use a fresh manual deployment
named `vjepa-cost-benchmark-05`. Verify its new SSH public host fingerprint
before access, copy/verify the package and restore to a fresh `/workspace`
directory. Never reuse a terminated Pod's endpoint or an earlier session's
deadline. With the exact expected GPU name, current all-in rate and an absolute
deadline derived from **deployment time**, the prepared command is:

```text
bash scripts/start_hardware_benchmark.sh \
  --expected-gpu EXACT_GPU_NAME \
  --hourly-rate-usd REVIEWED_ALL_IN_RATE \
  --deadline-epoch DEPLOYMENT_BASED_UTC_EPOCH
```

The detached supervisor verifies the restored source, caps setup at 25 minutes,
caps workers at 110 minutes or the shorter remaining allowance, caps preflight
at 20 minutes, and reserves four minutes for evidence export. These windows
share the same absolute deadline; they are not additive billing allowances.
Timeout, OOM, failed tests, failed numerics or incomplete measurements remain
failures/inconclusive evidence. No threshold is relaxed to finish in time.

Copy and independently verify the export and retained preflight tensors while
the Pod is accessible, then stop/terminate it in Runpod and reconcile billing.
Neither Python exit, timeout nor an SSH disconnect stops provider charges.
The supervisor has no provider API key or lifecycle control. Preserve partial
evidence on failure rather than claiming a passed benchmark.

Preparation uses local small-model fixtures and archive checks. Until this
package is actually run, **there is no new GPU timing, memory-fit, recovery or
whole-study affordability result**.
