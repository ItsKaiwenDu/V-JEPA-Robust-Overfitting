# V4 expected setup GPU refusal confirmed

The v4 attempt correctly returned the original GPU guard refusal. Its saved
setup query identifies the same process as v3: PID 3418810,
/usr/NX/bin/nxnode.bin, using 191 MiB. Total recorded GPU memory use was 293 MiB.
The original strict guard rejects any other CUDA compute process. These are
observations from the attempted start, not a fresh claim about current availability.
No process or server setting was changed.

The server's refusal type is ValueError, and its top-level message exactly matches
the saved setup guard message. The client wraps that message in RuntimeError as
intended. The previous secondary NoneType.layout masking is absent. This validates
the refusal route on the target; it does not qualify the numerical experiment.

The exact pinned source places that setup checkpoint before parent guard/session
creation and before the proof/worker role loop. Joined with the saved setup refusal,
it establishes that storage-proof/model worker dispatch was not reached. The
packet import correction remains unobserved in a live scientific worker. Initial
root/source/input setup writes had occurred; no claim of zero remote writes is
made. Remote contents were not inspected.

Five local records are independently pinned: ATTEMPT, ROOT-RECEIPT, REFUSAL,
CLIENT-REFUSAL and empty ssh-stderr.log. Their bytes, sizes, private modes and
modification times verify, and control/authorization/claim/root identities join.
Both complete saved GPU query texts match their lengths and SHA256 values and are
identical to the v3 query evidence. No collection body/receipt or final SSH exit
status was retained, so no archive or wire-integrity claim is made.

The fresh standard-library saved-file/AST audit passes on its first execution.
It verifies all 13,445 prior file hashes, 11,081 prior fixture entries and historical
saved byte/mode/time groups, including both earlier failed collections and v3's
five refusal records. The old FIFO is inspected with lstat only, never read or
hashed. No production source changes, tests or scientific cohort reruns were needed.
No further SSH, GPU query, timer, process, proof, model or paid work was performed
by this review. Existing v4 local validation and all earlier evidence remain intact.

The v4 grant is consumed and its root/authorization/claim must stay unchanged.
Do not rerun/rearm the command, reset its claim, force collection, relax the GPU
guard, kill another process or open a new retrieval connection under that grant.
The refusal calls for availability checking, not another launcher code version.

Next prepare and review a narrowly bounded availability-only helper for linux10603,
then seek the exact read-only connection scope before using it. It should read
host/GPU/process status only, with no model/proof launch, university file writes,
model-attempt claim, automatic retries or server changes. Keep that authorization
separate from a later fresh model attempt. Do not request repetitive manual
hostname/nvidia-smi pastes. Do not silently substitute another host or use an old
consumed grant. A future model attempt still needs its own fresh reviewed handoff
and exact approval after availability is known.

Scientific/training/performance/cost admissions, original experiment bounds and
financial caps/holds/reconciliation remain unchanged.

Machine report: 2026-11-16-pgd-v4-GPU-setup-refusal-review.json.
Raw audit and pins:
runs/pretrained/diagnostic/predictor-pgd-outer-incident-v4-20261116/.
