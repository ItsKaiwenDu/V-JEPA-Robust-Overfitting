# Result publication: close before committing metadata

The consumed linux10609 attempt failed during the first save_tensor read-back hash
because Store.check rejected changed metadata for pixels.npy. Its expected/observed
rows were not logged, so the exact live field and root cause remain unproved. All
old source, controls, grants, claims, proof/collection records and data stay frozen.
This step implements only a cold candidate repair, with no new target handoff.

The frozen Store.write_bytes records its complete inventory row before its finally
block closes the owned write descriptor. The local reproducer uses the exact saved
store and IO facade, real private files/write/fsync/close/hash operations and opaque
bytes at the same planned first-input path. Its explicit injected inventory view
adds 4 KiB allocated_bytes only while that descriptor is open. The unchanged writer
then raises the exact saved error on immediate read-back, with unchanged file bytes,
identity, extent and mtime. The normal local-filesystem control does not reproduce
the fault. This establishes a software ordering defect under the injected condition;
it does not establish that XFS changed that field in the actual run.

The candidate closes and retires the owned write descriptor once before taking its
final inventory snapshot. It then revalidates parent identity and the published
file's held identity/extent/mtime, runs the unchanged whole-inventory/cap checks,
rechecks every other committed member, and validates the published row again before
recording it. Only the current target row can change through an authorized mutable
replacement. Every recorded field, including allocated_bytes, remains part of later
Store.check comparisons. The allocation comparison is not removed or made advisory.

A close failure poisons publication and does not retry the retired numeric file
descriptor. Post-close mode/mtime/size/inode changes, parent replacement and changes
to prior committed members refuse and preserve evidence. Existing exclusive,
foreign-mutable, symlink/hardlink, owner/thread, recursion, poison, type, namespace,
logical-byte and aggregate allocated-byte guards remain. The allocation ceiling
is tested through the actual Inventory.scan with a fake stat allocation; no large
extent is allocated. Existing OS/snapshot limits are not widened into guarantees.

Store.check additionally retains its original failure condition/message and adds
a bounded diagnostic note (at most 2,048 payload characters) listing changed fields
and expected/observed rows. Actual row values are fixed metadata integers/identity
tuples; this is diagnostic evidence, never an acceptance override. A later injected
foreign allocation change still refuses and reports allocated_bytes explicitly.

The recipe accepts only the exact frozen source hash and edits two reviewed sites
plus a standard-library json import. Its whole-module inverse restores every old
byte. Inventory and other helpers/Store initialization remain exact. Candidate
source is returned and cold-compiled, not installed into any old runtime. The parent
process-monitor exit race is a separate unchanged issue.

Thirty distinct new checks have passing latest results over 31 executions (27/1/2/1).
One first-run test incorrectly expected a pre-close final-name snapshot from the
candidate; only that test expectation was corrected/rerun. Production recipe and
candidate stayed unchanged. The two saved save_tensor variants test the actual
function with explicitly inert tensor/array/NumPy serializer/finite-check stubs:
the old source fails in the same hash path before artifact publication; the candidate
returns matching byte metadata. No real NumPy/Torch import, array/numerical operation,
model, gradient, profiler, process launch, enumeration, signal, timer, GPU, proof,
network or paid compute executes. Actual own-PID/thread identity checks, locks and
ordinary local metadata/file IO run. No passing test, historical test or scientific
cohort repeats. All failed fixtures, logs, JUnit and source snapshots remain.

Complete an independent saved-source/evidence audit before integration. The next
step is a fresh source-bound integration or a small separately reviewed byte-only
XFS probe before another model attempt; do not claim the live cause or skip guards
based on this simulation. Any university connection/write needs its own exact scope
after concrete preparation. No new grant is requested here. All scientific/backend/
ownership/resource/time gates and financial caps, holds and reconciliation remain.
