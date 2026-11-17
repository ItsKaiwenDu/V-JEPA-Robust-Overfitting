# Repaired PGD attempt refused at setup GPU gate

SSH connected and returned a bounded startup refusal. The saved query identified
`/usr/NX/bin/nxnode.bin`, PID **3958249**, using **174 MiB** on the expected
linux10609 RTX 3070 UUID/driver. Total reported GPU usage was 277 MiB. The two
complete, untruncated query texts match their saved byte counts and SHA256 hashes.
This is the refusal-time observation, not a statement of current availability.
Leave the process alone; do not allowlist it or weaken the foreign-compute guard.

The remote scratch root was created, and source/packet setup reached the parent's
setup GPU preflight. The exact pinned source places that setup checkpoint before
creating its parent work-session claim and before any worker launch. The guard
refused there. **No 2.5 GiB proof, model, gradient or component worker started.**
The earlier setup's source/metadata writes and emergency reserve are not claimed
to be absent: the remote root remains and must be preserved. This result does
not test the repaired Store or monitor during model work and is not a numerical
rejection or new performance/cost result.

The original preclaim-refusal route returned REFUSAL.json without trying to
collect an uninitialized work session. Five local files are verified and joined:
ATTEMPT.json, ROOT-RECEIPT.json, REFUSAL.json, CLIENT-REFUSAL.json and empty SSH
stderr. Actual authorization/control hashes, root suffix and original 40/50-minute
clock relationships match. There is no local saved result tree or collection
receipt, so no second independent remote directory/file audit is claimed.

Saved-only stdlib source/hash/AST review passes **14,941 prior hashes and 13,103
immutable fixture entries unchanged**, with the old FIFO lstat-only. The first
auditor stopped at a late standard-library import under its no-execution hook;
only import ordering and the inherited prepare-method lookup were corrected in
a separate auditor. Both scripts/logs remain. No production source change,
experiment/test/cohort rerun, process/timer, network, new GPU query, proof, model,
array or paid compute occurs during review.

The grant is consumed. Never rerun/rearm/reset the old command, retry collection,
open another target connection under it or clean its root. The next step is an
idle workstation: wait for availability or prepare a separately reviewed
alternative-host availability path. Do not prepare another blind model retry
while assuming this snapshot reserves the GPU. No repeated manual GPU/key paste
is requested here. All original scientific/backend/resource/time/ownership gates,
financial caps, holds and reconciliation remain unchanged. The user commits/pushes.
