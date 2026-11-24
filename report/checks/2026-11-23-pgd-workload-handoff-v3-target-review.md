# V3 target review: original PGD progressed; whole unit incomplete

The approved v3 attempt is **consumed and failed**. Actual model preparation
reached full-size original PGD and saved31 completed input-gradient journals from
the40-context primary unit. Context31 was started but has no completion record.
At the720-second unit boundary, the recorded parent error was **“Owned child exit
is unconfirmed; no finalization or collection.”** No whole unit completed;
candidate-unit and candidate-replay never started. The full2.5GiB proof passed,
and the same connection collected237 files with SSH exit1.

The independent stdlib saved audit **passes first-run**. This is a successful
review of a failed partial workload, not promotion of the workload to success.
See the [machine-readable review](2026-11-23-pgd-workload-handoff-v3-target-review.json)
and [local shutdown/budget analysis](../../docs/predictor_pgd_workload_deadline_exit_review_v1.md).

## Bound attempt and observations

- Package unchanged:
  `runs/pretrained/diagnostic/pgd-workload-handoff-v3-20261123/handoff-parent/linux10614-primary-v3-final-unarmed`.
- Outer control: `6e44d8bb315e3595333c70278c315a434c1fac2629018c0d2567db76dc02bcee`.
- Supervised control: `0e2f47ad5ff915dfa3b7aad424bee5755cb621596263e5b95195dcc04a3ad82f`.
- Original2019-byte authorization, unchanged:
  `d168c80af82c1e2c83ce009c80e5ce9ca5b161678e63d3b076978cbbb378c9e7`.
- ATTEMPT consumed TRUE, SHA256
  `d272586e133febb93c33b8056cb56b2f942faa55fe576d5d88a27746639a3a91`.

| Stage | Saved evidence |
| --- | --- |
| Full storage proof | PASS:2684354560B written/fsynced/read back; own proof removed;13.704431230s |
| Proof allocation/parent | Peak observed allocation2746851328B/1926events; parent14.105411710s/104RSS samples/42745856B |
| Original worker | PID1694139, original training row `v_BreastStroke_g15_c01`, primary20/R2/EOT8 |
| Gradient progress | 32 start records,31 completion records, indices000–030 complete;031 incomplete; budget40 |
| Completed-context route observations | One native root/scope,1152 calls,0 refusals, root hook removed for each saved completion |
| Recorded context intervals | 17.744957367–17.895007595 seconds, including the observations inside the measured interval |
| Unit completion | None: no unit-complete/work/resource/complete marker or successful original parent observation |
| Failure | Parent RuntimeError from assert_idle; diagnostic child poll status None |
| Later roles | Candidate and replay did not start |
| Transport | 237 declared files verified; SSH1; session failed/complete FALSE, completed roles only `[proof]` |

The target ran real Torch/CUDA model preparation, original EOT8 input-gradient
computation and a real full proof. It is not metadata-only target work. Local
review reads saved byte/JSON/source observations only: no tensors, checkpoint or
pixel values were decoded, recalculated or compared again, and no model, gradient,
proof, test, profiler or old scientific cohort was run. The32nd durable start does
not establish that its differentiation finished or even began backward.

The saved route fields and finite loss/hash schema join each step's start/completion
and source identities. Gradient arrays were not collected for independent accuracy
comparison. These records do not establish original/candidate numerical agreement,
fresh target replay, a completed attack selection/scoring/restoration result or a
resumable recovery checkpoint. The preceding mechanism diagnostic remains closed.

The per-context interval is assigned before its own completion-journal publication
and before timer exit. It is not the complete caller/whole-unit interval. One
partial sequence is not a statistical performance estimate; preparation, scoring,
inter-context and other observation overhead have not been independently measured
or subtracted. No complete-unit throughput or training-progress percentage follows.

## Deadline and exit-confirmation uncertainty

The original role launch starts1070552.318197682, with absolute deadline
1071272.318197682 (exactly720 seconds). The final GPU journal sample is at
1071272.201585825. Collection starts1071272.451153730, about0.132956048 seconds
after the unit deadline. Connection origin1070530.355382395 retains work deadline
1072930.355382395 and connection deadline1073530.355382395. Collection deadline is
1071872.451153730; no clock reset is recorded.

The diagnostic ends in the coordinator's assert_idle at line542, and its note
says the original child exit status was None. The child log contains only
deprecation warnings, with no traceback. The first triggering exception and final
child exit code/death/reap timing were not retained. A deadline-associated shutdown
race is a source-and-timing inference, not a saved primary timeout diagnosis.

The saved monitor performs bounded termination/reaping and closes leases in its
finally block. The outer wrapper skips its additional wait when monitoring ran
and the lease is already closed, then its except block calls assert_idle before
re-raising the primary exception. If the process has not yet reported exit, that
assertion can replace the primary error. This explains a possible masking path;
the missing original exception prevents establishing its precise trigger.

The actual source also requires assert_idle before collection. The later guarded
collection is consistent with that check returning after the early failure; it
does not rewrite the earlier diagnostic, reveal the terminal code, authenticate
exhaustive cleanup or establish current remote process state. No follow-up query,
cleanup, retrieval or new connection was made. All grants are consumed.

No original-worker RSS peak/sample count, Torch allocator peak or successful
parent resource record was published. No new RSS refusal is saved, but the lost
primary exception means a memory/resource trigger cannot be ruled out merely from
its absence. The74 GPU samples peak at5404MiB process-used and5486MiB device-used
in nvidia-smi. These are not host RSS or Torch allocator peaks and do not qualify
instantaneous limits or full-model/unit memory fit. Preparation got past the v2
failure point; a measured RSS reduction or complete memory qualification is absent.

## Transport and preservation

All27 input pins,17 local source pins,55 uploads/896039B,47 runtime files/450548B,
nine supervisor installation records and original packet/99 member descriptors
join. The audit reconstructs the saved priority header, ordered file frames,
payloads/references, prefix and terminal; all237 modes/mtimes/hashes and journal
rows/offsets/hash links match the recorded client digest.

- Expanded45365115B; payload45054308B; deduplicated310807B.
- Reconstructed wire45186134B, SHA256
  `a778d8bafa8fbbee2f0f7c3b0c7fc83a8e3a3773eaf5ec21db8ba9ad11a3dee6`.
- Journal metadata plus progress257902B; sampled local allocation49029120B.
- Inventory239 files;237 transported. Original emergency reserve and the2355B
  generated `tmp/tmpnfjbxde4/_remote_module_non_scriptable.py` cache file are
  explicitly omitted by the header. Their inventory/allocation accounting is
  retained; no separate retrieval is attempted.

Raw network wire/terminal was not separately saved. This is independent
reconstruction matched to the client's saved full-wire digest, not a direct
second scan of the original stream. Session/work collection/partial/journal FALSE
and original pending flags remain unchanged. Separate transport TRUE does not
authenticate computation or promote failed/incomplete science.

The real proof result,1926-event sequence, proof parent PID1693692/start ticks
107053819/exit0, session and receipt hashes join. This review does not recreate
removed random proof data or recompute its2.5GiB digest. Original proof FALSE
qualification/authentication fields remain. Sampled observations are not an
instantaneous quota, reservation or exhaustive process proof.

All53292 prior hashes,58605 prior fixture entries and275 new attempt/session entries
pass the standalone audit. Old FIFO LSTAT only, never opened. Local device16777233
matches its prior conditional observation; old and target device mappings remain
unchanged. The five original package files retain complete recorded identities
and hashes, including AUTH inode44967366. Package inode44965638/owner/group/mode
stay fixed; only the approved attempt/session lifecycle changes tracked APFS
nlink7 to9 and directory metadata. Parent and all earlier lifecycle exceptions
remain unchanged. Prior uncommitted scope/arming work, all source/auditor/test
versions, failures, logs and claims are preserved. User alone stages/commits/pushes.

Raw review: `runs/pretrained/diagnostic/pgd-workload-handoff-v3-20261123/target-review-20261123`.
Seal: `sealed-workload-v3-target-review-v1.json`. Live mutable indexes are excluded;
before/after snapshots are retained. The saved audit invokes no project/test/
scientific callbacks, tensor decoder or process/timer/signal/GPU/proof/network work.

## Next separate local work

Derive and test bounded delayed-exit/primary-error preservation with explicit fake
OS ports. Keep idle/ownership gates and all caps, deadlines and scientific paths.
Review complete-unit clock/overhead feasibility before another scope: this attempt
did not finish40 contexts within its window, and candidate/replay costs are unknown.
Do not retry, resume/promote partial journals, shorten PGD/EOT or disable guards for
timing. No new scope/handoff/approval request or production change occurs here.

Actual complete primary/convergence/fresh replay, matched timing/explicit overhead,
chosen-hardware memory/exact recovery, economics,30 training-only calibration
evaluations/10 clips and data/protocol adoption remain. Research/training FALSE,
trainingPGD5/EOT1/routineEOT1/optimizer backward/registry/study YAML and financial
200/160/40/single10 caps/holds remain unchanged. Old financial snapshots remain
historical; no measured percentage or time-to-training is established.
