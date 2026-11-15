# Split profiling design passes local review

The new local design keeps all eight original/candidate matched EOT8 gradient
objectives and their full numerical/native/root/checkpoint/restoration checks.
It proposes two additional serial attention-only backend probes on a new native
boundary captured inside the first original perturbed objective. The probes share
actual Q/K/V and incoming cotangent values and original layout provenance. One
uses native forward/backward; the other preserves native forward and uses public
math in its owned custom backward. They are explicitly additional VJP evaluations,
not hidden work inside the previous eight-gradient allowance.

A new identity would admit only representative backend evidence, not the old full
EOT8 kernel counts, component VJP accuracy, replay, performance/cost or training.
No full-model detailed trace, FP64 reference, CPU gradients, warmups, untimed
controls or completed-cohort reruns are proposed. The old worker/reader and all
prior evidence remain unchanged; the old timeout's exact cause is still unknown.

The exact proposed inventory contains 419 result files. Joint raw accounting,
including all old numerical/rollout returns, new boundary/probe arrays and a full
partial, is 1,852,641,328 bytes below the 1,792 MiB cap. Metadata, both raw traces,
provisional snapshots, duplicate native proofs, journals, logs and mutable partials
total 60,702,720 bytes below 64 MiB. Traces stay inside that shared metadata cap.
Logical/category/allocation reserves fit the 2.25 GiB capture / 3 GiB setup
ceilings; actual allocated/free/inode conditions still require live guards.

All work phases fit 40 minutes: setup 180 s, proof 60 s, numerical worker 1,680 s,
two component workers at 120 s, finalization 120 s and overhead 120 s. The numerical
allowance is 28 minutes rather than 32; actual completion in it is unknown.
Same-SSH collection stays at 10 minutes. Proposed component wall/RSS monitoring
must be implemented; post-materialization event/byte caps are not instantaneous
memory or profiler-stop guarantees. See the
[complete design and resource table](../../docs/predictor_pgd_split_profile_v1.md).

**Validation:** 47 distinct new cold design checks are covered. First run: 45 pass,
one test-expectation failure, no errors/skips. The calculator's proof sum was
correct; the test literal was corrected from 3,028 to 2,928 MiB. Only that failed
check and one new predecessor-reservation check ran next; both pass. Both source
snapshots and logs remain. No passing check or scientific cohort was rerun.

Independent standard-library saved-file review passes 4,510 prior hashes and
242/280/135/245/5/3 historical descriptor groups with byte/mode/nanosecond-mtime
checks. It independently recomputes raw/metadata/time bounds, checks all 419
inventory entries, validates the exact test-only correction and preserves both
consumed claims. No model, gradient, profiler, GPU, SSH, proof, retrieval, cleanup,
paid work or new financial hold/release occurred.

The existing component reader assumes matching forward/backward families and
cannot certify the proposed hybrid. Its association policy also rejects math
forward inside the backward phase. Next build a new metadata-only hybrid reader
with explicit recorded cross-thread and inner-task identity. New component
ownership, numerical receipts, progress lifecycle, inventory/parent supervision
and source packaging need local integration afterward. No target runner, helper
or authorization is ready. Financial caps/holds/reconciliation, old rejections,
frozen successful cohorts and all training gates remain unchanged.
