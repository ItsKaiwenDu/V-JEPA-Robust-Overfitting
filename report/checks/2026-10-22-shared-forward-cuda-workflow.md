# Shared-forward CUDA workflow: bounded exact recovery passed

October 22, 2026 (Pacific). **All eleven diagnostic phases completed. Both
clean and adversarial two-update restart comparisons match exactly.**
[Structured evidence](2026-10-22-shared-forward-cuda-workflow.json).
This validates the declared `shared-fp64-native` candidate for this bounded
check. Full research trajectories, study calibration and budget admission
remain pending; no robust-overfitting finding follows from it.

The user deployed `vjepa-recovery-pilot-03` (Pod `3fmkak2dyu3vvo`). Its public
SSH fingerprint was matched before connecting. The host had one RTX 5090,
Python 3.12.3, torch 2.8.0+cu128 and torchvision 0.23.0+cu128, with the selected
30 GB container and 80 GB persistent `/workspace` volume. All six package
files and 24 transfer chunks verified before restoration, followed by all
237 frozen source-member checks. Slow SSH transfers were replaced with the
Pod's authenticated HTTPS service; its token remained in memory.

**Setup:** 436 tests passed, zero failures/errors/skips, in 632.46 seconds.
The two CUDA-only cases skipped on the Mac executed here. Setup, including
the original checkpoint download and hash verification, took 758 seconds.
The execution supervisor used the conservative 9:30 PM Pacific deadline,
earlier than two hours from the observed deployment/container start.

**Representative preflight:** training indices 0, 4, 7 passed both backward
arms under the unchanged oracle/full-input gates. The six retained tensor
files, signatures, full-chain reconstructions, bounds, FGSM sign-step/clean
candidates, native CUDA probe and all comparisons were independently checked
on the Mac. A separate CPU FP64 convolution using the rehashed original
trained projection agrees with the saved forward reference within
1.64e-13.
Full input-gradient maximum differences are 1.490e-7, 2.980e-7 and 1.416e-7;
every original pointwise gate passes. Shared tokens are bitwise equal.
Cross-backward attacks are not universally bitwise equal: FGSM differs at
index 0 and PGD-5 at index 7. These descriptive comparisons retain their
original interpretation; exact recovery is assessed within the native policy.

**Training and recovery:** eight original training clips and one training
monitor were used. Four trajectories perform eight optimizer updates total:
clean/adversarial split runs pause after update one, resume in fresh processes,
and reach update two; separate references reach update two uninterrupted.
Each comparison passes all 5,351 logical entries, including weights, AdamW
state, RNG/history, contracts and scheduled evaluation metrics. All five
scheduled receipts through epoch two verify for each trajectory. Validation
of the exported comparison records is local; full optimizer tensors remain
excluded from the small export. The original on-host comparison validated
their hashes and exact logical contents before producing those records.

**Calibration, features and controls:** the one-clip diagnostic chooses
2/255 for primary PGD; training/routine PGD keeps 1/255. This is separate from
the required ten-training-monitor study calibration. Initial and trained
epoch-one anchor features/probes completed on eight training examples;
training resubstitution is not held-out accuracy. PGD-50/R2/EOT8, FGSM and
random controls completed on the pretrained training-monitor clip, including
two extra independent scoring banks for convergence. No exploration,
confirmation or test pixels were opened by this diagnostic.

| Operation | Observed seconds |
| --- | --- |
| adversarial-finetune batch-eight update | 41.173–41.564; median 41.354 (n=4) |
| clean-finetune batch-eight update | 6.953–7.132; median 7.036 (n=4) |
| convergence attack/scoring clip | 631.930–631.930; median 631.930 (n=1) |
| fgsm attack/scoring clip | 6.474–6.474; median 6.474 (n=1) |
| primary attack/scoring clip | 136.781–139.060; median 137.232 (n=9) |
| random attack/scoring clip | 2.761–2.761; median 2.761 (n=1) |
| routine attack/scoring clip | 4.998–5.373; median 5.190 (n=10) |

These short measurements cover selected training clips and checkpoints.
Attack timings include their recorded scoring work; source/runtime/data
contracts remain explicit. Nested spans overlap and must not be added.
Peak PyTorch allocation across phases was 7.586 GiB; this
excludes additional driver/reserved memory. The earlier university 8 GB
AdamW OOM is not resolved by this RTX 5090 measurement.

The complete worker session took 2967.812s,
excluding setup, transfer and earlier idle billing. The copied export is
419,380,347 bytes with 85 verified
members; SHA-256 `866c48f1388d95605537ec57e3123031fe5e07388e76dbaba5c97670b0ece862`. All wrapped JSON, phase hashes,
calibration reconstruction, journals and receipt mappings verify. Raw evidence
and exact verification scripts remain ignored under
`runs/pretrained/diagnostic/shared-workflow-cuda-20261022/`.

**Budget/lifecycle:** the user confirmed full Pod termination after the
export was copied and verified. The reviewed quote was $1.21/hour with a
two-billed-hour/$3 inclusive limit. The user says the $1.688 billing summary
includes the previous confirmed $0.603 session: the derived new increment
is **$1.085**, and cumulative spending including the first $0.592 pilot is
**$2.280**. These derived values remain provisional until a settled
Pod-specific charge is available. Prior confirmed reports total $1.195;
retain the $3 reservation pending reconciliation ($5.805 unreserved in
the shared pilot allowance). No invoice/API or provider shutdown state
was independently queried. Python exit itself did not stop billing.

Next, reconcile spending and replace the old proposal's proxy costs with these
measurements before admitting any exploration. Keep data review, the declared
precision/protocol and research calibration gates. Study/proposal YAML and
all numerical/exact-recovery thresholds are unchanged. Preserve every earlier
rejected original-runtime/FP32 experiment.
