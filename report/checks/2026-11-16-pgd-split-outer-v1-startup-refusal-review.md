# Startup refusal confirmed; collected evidence verified

The numerical worker stopped before creating its first progress record, building
the model or computing a gradient. Setup had already created
`capture/results/numerical`, and the frozen worker correctly required an exclusive
`mkdir`. The source I prepared treated the allowed directory catalogue as a list
of directories to create. That setup mistake caused this failure.

The original 2.5 GiB write/fsync/readback proof passed in 15.1243 seconds, with a
peak observed allocation of 2,746,728,448 bytes below the 3 GiB setup limit. Its
1,926 event records retain 640 writes and 640 readbacks of the original chunks,
the fsync, and removal of its own proof file. This is an observed successful
storage-proof receipt, not a kernel quota or general live-XFS qualification.

The client completed same-SSH collection and recorded SSH exit status 1. Independent
local review verifies all 186 collected regular files against their recorded
hashes, extents, modes and nanosecond timestamps, plus exact control/authorization/
installation/source joins. The client reports 45,190,076 wire bytes. The complete
wire response was not retained, so its wire hash is reported from the client
receipt rather than independently recomputed. All file bodies were independently
rehashed. No new SSH inspection or retrieval was needed.

Five saved GPU samples show no additional compute rows and no guard refusal. The
traceback identifies the directory collision. It does not support attributing this
failure to SSH disconnection or GPU contention. Runtime imports and the CUDA
availability check were reached; model construction, numerical objectives,
boundary capture and component probes were not reached. Scientific/backend/
accuracy/performance/training admissions remain false.

The same setup loop also precreated the boundary observer and both component
folders, plus eight objective directories. The previous setup tests substituted a
directory table that omitted this ownership conflict. New integration checks use
the actual 172-asset/419-result catalogue and only the frozen workers' individual
`mkdir` statements, without executing scientific functions or importing Torch or
NumPy. They reproduce all four entry-folder collisions.

A new pure local directory-plan candidate selects 26 source-ancestor/supervisor
shell directories and leaves all 12 worker claims fresh. All 13 distinct checks
now pass across 17 executions. Four initial test failures were caused by a missing
root-binding variable in the isolated loop fixture; only that fixture was fixed,
and the nine passing checks were not repeated. All snapshots and failures remain.
The candidate is not integrated into a new runnable handoff and creates no grant.
The consumed v1 source, files and authorization remain unchanged.

The fresh standard-library saved-evidence/source audit passes all 9,712 prior
hashes and 6,758 earlier fixture layout entries. The old FIFO is checked with lstat
only, never opened, read or hashed. One inspector fsync-label assertion was corrected
from its guessed label to the exact frozen `after-data-fsync` event; its first
script/log remain. No production or scientific evidence was changed.

This attempt is consumed. Do not repeat the previous command, delete the existing
worker directory, rearm v1 or make a separate retrieval connection. Next integrate
the minimal directory plan into a fresh corrected handoff, test its real directory
contract and independently review it before requesting one new scoped attempt.
Original bounds, old outcomes/partials and financial limits/holds/reconciliation
remain unchanged.

See the [machine-readable review](2026-11-16-pgd-split-outer-v1-startup-refusal-review.json)
and [local correction candidate](../../models/predictor_pgd_split_directories_v2.py).
