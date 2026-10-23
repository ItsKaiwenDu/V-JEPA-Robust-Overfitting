# Shared-forward CUDA handoff preparation — October 22, 2026

**Outcome:** the fresh offline handoff is built, restored, checksummed and
locally exercised. **434 passed, two actual-CUDA skips, 164 warnings
in 247.61s.** Fourteen new tests cover representative preflight, rejection,
state/RNG/mode restoration, the complete candidate workflow, bounded tensor
exports and refused unreviewable deadlines. Actual new Linux/CUDA setup,
pretrained optimizer recovery and measured GPU costs remain pending.

## Concrete package

- Directory: `runs/pretrained/diagnostic/shared-workflow-handoff-20261022`.
- Manifest SHA-256: `99bbeba481d8dd4e7533e7aabc0122770096bbf096a352f2bb1204ba09e30c47`.
- Total package bytes: 815,075,082; source files: 237.
- All source/archive/member hashes verified by restoration; required precision,
  observer, oracle, training, launcher/export and fresh-process fixture sources
  are present. Exact upstream commit and 32 Python sources match.
- The 100-training-clip and 23-wheel archives are bitwise equal to the earlier
  verified handoff. Active Linux/Python 3.12 dependency checks: 27, all satisfied
  against the previous tested template plus these wheels. No new Linux install
  is claimed. No exploration/confirmation/test cache directories are included.
- Rehashed the separate 5,127,726,842-byte release checkpoint. Verified all eight
  previously opened selected training clips and the monitor by saved pixel
  checksums. No pretrained model was executed locally in this preparation.

A restored copy passes the scaffold check and 6
selected integration cases, including representative oracle/attack checks,
complete pilot phases and twelve fresh CPU optimizer recovery workers.
This is small-model CPU evidence from the actual extracted source tree.
The curated record is written after package construction to avoid a manifest
self-reference; later progress/status documentation is not silently substituted
into the frozen source archive. Scientific source/configuration stays pinned.

## Pilot and failure behavior

An added preflight gates expensive calibration for the shared-forward policies.
It uses training indices 0, 4, 7, mask seed 0 and independent score seed 100000.
Both backwards use the same FP64 patch forward/FP32 tokens and unchanged
FP64/whole-model pointwise limits. It retains raw oracle/input-gradient/attack
artifacts, actual new native CUDA observer evidence, frozen target identities,
nonzero gradients and required native-loss ascent. Cross-backend attack equality
is descriptive. Historical rejected FP32 attempts remain rejected.

The existing clean/adversarial split/resume/reference workflow, calibration,
feature/probe and convergence controls follow only after preflight success.
Exact recovery comparisons, model/loss/attack definitions, execution integration,
study/proposal YAML and all numerical limits are unchanged. New source/runtime
provenance prevents old artifacts being reused for this candidate.

The detached launcher verifies the restored source and requires an absolute
reviewed deadline tied to deployment. Bounds: setup 25 minutes; preflight
20 minutes within a maximum 90-minute pilot; four-minute export reserve;
final supervisor constrained by the deployment-based allowance. It preserves
partial failures and attempts a bounded export, including up to six preflight
tensors (256 MiB/file, 2 GiB total), with optimizer states excluded. A setup or
pre-session pilot failure is explicitly marked. **Python deadlines do not stop
cloud billing.** No rental, API key, cloud lifecycle action or GPU job ran here.

## Current budget and required action

Proposed instance: one RTX 5090 32GB, previously exercised PyTorch 2.8.0 image,
30 GB container plus 80 GB persistent volume at `/workspace`, at least 55 GiB
free after setup. The earlier university 8 GB AdamW OOM is unresolved.

Keep the session to at most two billed hours and $3 inclusive of setup,
transfer, idle time, storage and applicable taxes. Reported cumulative spend
is $1.195; $8.805 remains in the shared pilot reserve. Total cap $200 / planned
compute $160 is unchanged. On October 22 the user reported a current
**$1.21/hour** console quote and confirmed keeping the 30 GB container and
80 GB persistent volume. The quoted running cost for two hours is $2.42;
the remaining $0.58 must cover any charges outside that rate. This is a
reviewed user-reported quote, not an independently queried live offer or
a recorded session charge. Availability is not explicitly confirmed.

Next user action: manually deploy the reviewed configuration when available
and ready, then provide its current SSH endpoint and deployment time.
The absolute execution deadline will be tied to that deployment. No new
Pod has been launched by this preparation. The frozen package is unchanged.
See [the run guide](../../docs/shared_forward_cuda_pilot.md),
[bounded novelty refresh](../../notes/2026-10-22_novelty-refresh.md), and
[structured provenance](2026-10-22-shared-workflow-handoff.json). No study admission or robust-overfitting
finding follows from this handoff.

**Execution follow-up:** the prepared snapshot subsequently completed the
[bounded CUDA workflow](2026-10-22-shared-forward-cuda-workflow.md). Both exact
recovery comparisons and all eleven phases pass; the export is verified and
the user confirmed Pod termination. The preparation facts above remain a
historical snapshot; no frozen archive was replaced.
