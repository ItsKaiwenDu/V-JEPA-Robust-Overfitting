# University matched-panel handoff

October 24, 2026. The runner implements the [frozen plan](matched_panel_diagnostic_plan.md)
in a separate diagnostic namespace. Preparation and small upstream CPU checks
do not establish a ten-clip CUDA outcome. The planned panel has not run.
No cloud deployment, dependency installation or checkpoint download is part of
this handoff.

## What is packaged

`scripts/prepare_matched_panel_handoff.py` creates a source snapshot, the pinned
upstream Git bundle, exact plan, ten declared training tensors, training-cache
metadata, a standalone verified extractor and SHA256SUMS. The released campus
checkpoint is reused from the existing project; it is not uploaded again.
No exploration-validation, confirmation or test pixel tensor is packaged.
The full training index is metadata, not a promise that all 100 tensors exist
in this package. Only the ten frozen monitor clips may be loaded.

The source snapshot is the trust reference for later export review. Its
PANEL-HANDOFF.json binds the new execution sources separately from the immutable
plan. The plan's 51 project and 34 upstream source hashes and all 198 original
non-Markdown scientific pins must match. Later documentation bookkeeping does
not change this prepared snapshot.

The new reusable modules are `models/matched_panel_case.py`,
`models/matched_panel.py`, `models/matched_panel_io.py` and
`models/matched_panel_runner.py`. `scripts/check_matched_panel.py` is the thin
CLI; `scripts/university_matched_panel.sh` handles the detached Linux launch.
Canonical EvaluationSettings, PGD and study YAML remain unchanged. CPU fixture
records explicitly identify their generated inputs and cannot qualify as
university CUDA evidence.

## First case, review, then the queue

1. Obtain a fresh university hostname, GPU name/UUID/driver/memory and free-disk
   check before transfer or launch. The frozen reference is linux10609 and the
   previously observed RTX 3070 UUID; another GPU, driver or software stack
   rejects. An active CUDA compute process also blocks dispatch. Check shared
   filesystem free space separately from any personal quota.
2. Transfer archive, member manifest, SHA256SUMS and verify_extract.py into a
   fresh dated directory. Verify the three external file hashes before running
   the extractor. Use the existing `$HOME/.venvs/vjepa/bin/python`; no install
   or random-weight fallback is allowed. The launcher restores upstream from
   the local bundle and verifies package contents before and after execution.
3. From the extracted package, `bash scripts/university_matched_panel.sh first
   EXPECTED_HOST "EXPECTED_GPU_NAME"` launches only clip-00-bank-0. It runs
   matrix A/B/C/D, then a separate fresh A-only replay, exports evidence and
   stops. The command prints the panel directory and exact launcher log path.
   Closing SSH or pressing Ctrl-C in `tail -f` stops neither the detached job
   nor its deadline. Do not start another command while it is running.
4. Download the first case's unique tar.gz and matching .manifest.json, plus
   its launcher log and exit_status.txt. Keep the SHA printed in the launcher
   log as the external archive pin. Independently review that export locally
   against the trusted handoff with CLI `review`, supplying --archive,
   --manifest, --expected-sha256, --trusted-handoff, a fresh --output and
   --acceptance-output. Review checks the retained inputs, traces, contracts,
   native probe, bounds, worker completion and exact fresh A replay; it does
   not execute the pretrained model again. The acceptance record binds the
   plan, execution source, case receipt and archive hash.
5. Only after that independent review, upload the acceptance JSON and use
   `bash scripts/university_matched_panel.sh remaining PANEL_DIRECTORY
   FIRST_CASE_ACCEPTANCE`. This serially dispatches the remaining 19 cases.
   A missing or changed acceptance, failed worker or damaged saved case stops
   the queue. No case is silently skipped and no failed case retries itself.

The complete first-case command is at most 80 minutes plus a 15-second outer
kill grace. Its matrix worker has 3,600 seconds, replay 900 seconds, and parent
4,620 seconds inside that outer limit. Child process groups receive TERM then
KILL after ten seconds. Every CUDA worker uses the smaller of 7 GiB and 90% of
VRAM. The initial panel needs 12 GiB free disk; remaining work checks space
again, allowing for its evidence plus reserve. Exported files are limited to
256 MiB uncompressed per case, with a small separate index.

The old one-case runtime repeated twenty times gives 13.65 hours (17.07 hours
with a 25% margin). These are scheduling scenarios, not bounds, and do not
predict the other clips. Twenty 80-minute case allowances sum to 26h40m before
kill graces or the deliberate first-case review pause. Do not interpret these
limits as permission to share the GPU concurrently or as a calendar promise.
The remaining launcher has a finite outer deadline covering its nineteen
case allowances, kill graces and coordinator reserve.

## Evidence, interruption and resume

Each case writes matrix and fresh-replay records, retained pixels/targets/
starts/masks/adversaries, observed gradients and native CUDA probe, session,
read-back receipt and a bounded export. Runtime observations are instrumented
diagnostics and cannot enter A40 cost pooling. Mask draws are generated from
the frozen seeds in the required campus PyTorch 2.8.0+cu128 runtime. The local
PyTorch 2.14.1 mask audit is a reference, not a cross-version hash gate.

Journal pointers commit atomically only after whole-case and export verification.
An interrupted pointer write can recover completed evidence without another
GPU case. Resume rechecks plan/source/runtime/weight/pixel identities, receipts
and every saved export. Paths are relative to the panel root, so transferring
the whole root preserves receipt meaning. A single live supervisor holds the
panel lock. There is no PGD-iteration resume.

Preserve failed attempts. After inspecting the failure, an explicit `retry
PANEL_DIRECTORY CASE_ID ACCEPTANCE_OR_DASH` may start a new attempt for the
next incomplete case only. Its remaining allowance subtracts time already
recorded and coordinator reserve; an unknown/exhausted allowance rejects.
Use `-` only when retrying the first case before acceptance. A numerically
complete case with a missing/damaged export needs offline export/pointer
repair, not another GPU run. Never delete the failed attempt or edit its
elapsed time to reset the limit. A failure is not a pass, and no precision,
tolerance or backend fallback is automatic.

Case archives use unique `<case-id>-attempt-<NN>.tar.gz` names in exports/,
with matching sidecars. The small `panel-index-<count>-<id>.tar.gz` contains
journal, progress/analysis and receipt/manifest references, rather than a
second copy of every large case. Download every case archive and its sidecar,
the latest index and logs. A failed case is exported where possible but is
never committed as successful. `latest-index.json` identifies the newest
index; earlier indices remain historical snapshots.

Incomplete collections expose progress and `scientific_aggregate: null`.
Only all twenty verified cases produce B-A, D-C, C-A and D-B contrasts for
main, audit_0 and audit_1 separately. Retain per-case losses/contrasts, average
the two technical banks within each clip, then average the ten clips equally.
Relative contrasts use each lower-setting adversarial loss; zero denominators
remain undefined while absolute differences are retained. No audit pooling,
case selection, sample-size-20 claim, primary selection or convergence pass
is automatic. Research calibration, protocol/precision adoption and measured
budget admission remain separate future gates.

## Verification status

The preparation report records the exact local and extracted test receipts and
archive hashes. Small upstream CPU cases cover both bank replicates, fresh A
replay, independent scores, retained evidence, failure export, journal recovery,
locks and supervisor stop paths. Pure analysis tests use synthetic loss tables;
they are not twenty executed model cases. The university panel and first-case
memory/runtime/replay remain unexecuted until a later reviewed launch.
