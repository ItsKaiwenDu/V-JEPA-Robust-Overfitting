# Strict inventory refusal observation and the missing replay plan

The guard's original decision is retained. A separate local observer adds bounded
filename/stat evidence only after the frozen inventory function has already raised
ValueError. Nineteen new local checks pass in their first and only suite. The three
verified CUDA workers and partial fourth remain unchanged; no new model, GPU,
university proof, SSH, cleanup, installation or paid work occurs.

## What the local reproduction establishes

The real v2 run stopped with `Unexpected special file or external hardlink.`.
Its log omitted the offending path and inode metadata. The last recorded cache
use and later empty cache cannot identify the cause. An actual unsafe file/link
is still possible; the historical university event remains unresolved.

A controlled local filesystem interleaving reproduces the same frozen refusal:

1. A regular JSON file and its `.partial` publication name share one inode with
   link count two, an atomic pair already recognized by the existing guard.
2. Immediately after the guard's real `lstat`, a scheduling hook lets the writer
   finish that publication, replace the JSON with the next generation and begin
   another partial generation. An open read descriptor retains the old inode so
   reuse cannot make the test ambiguous. All stat results come from the filesystem.
3. The guard then sees a partner from a different generation. Its original snapshot
   still reports two links, while both current entries are regular single-link
   files. It refuses. A separate test-only stable scan accepts the final entries.

The observer captures the original rejected inode/link count and the later entry/
partner metadata. This demonstrates a possible temporal failure mechanism on the
local macOS filesystem, with explicitly controlled scheduling. It does not establish
the target filename, an uncontrolled race frequency or Linux/XFS behavior. The
production observer never performs that second test scan or changes the refusal.

## Observer interface and bounds

`models/scratch_inventory_observer_v1.py` pins the frozen
`oracle_boundary_operator_v1.py` source and exact inventory function/code identities.
It calls the actual frozen scanner exactly once, with the same hashing, byte-cap
and running arguments. Successful results pass through unchanged. Frozen source
and acceptance predicates are not patched or copied into a replacement policy.
Source/identity drift refuses before scanning.

After a ValueError, the observer reads the matching scanner traceback frame and
captures only the rejected relative path, original inode/device/mode/type/link
count/UID/size/allocation/mtime and root device. At most two additional `lstat`
observations inspect that same path and the existing atomic partner name; they
do not follow a final symlink or open file contents. Follow-up stats cannot change the
already failed decision. Optional scanner hashing still reads exactly the data the
original caller requested; the added diagnostics do not read experimental data.
A source-integrity check also reads the pinned guard's source bytes.

Compact context stays at most 1,536 ASCII JSON bytes. Displayed relative paths stay
within 384 JSON bytes, with a digest and truncation flag for the full original path.
Optional follow-up snapshots may be dropped to meet the cap; the original observed
stat/path digest is retained. Context is attached to the new ValueError and included
in its message, fitting the existing 2,048-character parent error field. The original
exception remains its cause. Diagnostic failure reports a bounded unavailable status
and still refuses. Other exception types propagate unchanged. There is no scan
retry, new acceptance scan, error waiver, file modification or automatic continuation.
This observer is not integrated into any existing live helper or authorization.

## Local verification and saved evidence

The nineteen checks cover exact successful-scan parity with/without hashing; the
existing valid atomic-pair and physical-allocation accounting; genuine external
hard links, symlink, FIFO and Unix socket refusal; the controlled writer interleaving
through both old and observed scans; no added file-content reads; unavailable or
disappeared diagnostic observations; bounded deep Unicode paths; the existing
aggregate byte refusal; missing roots; source drift; and single-delegation/no-return
structure after refusal. Unsafe file/link classes remain refused.

The actual 280-file collected snapshot produces exactly the same frozen and
observed inventory, including 704,377,869 logical bytes. That stable post-run
snapshot does not resolve its earlier live failure or admit the incomplete replay.
No old successful scientific calculation or test suite is repeated.

Independent standard-library saved review checks the nineteen results, real saved
stat witnesses, source/AST constraints and evidence bounds without importing the
observer or executing any scan/reproduction. All 2,366 prior result/source pins,
280 latest target byte/mode/nsmtime descriptors, 135 startup-failure descriptors
and 245 earlier component descriptors remain unchanged. The original three-worker
CUDA numerical result stands; final candidate replay remains open.

## Plan only the missing scientific work

`configs/predictor_missing_replay_plan_v1.json` is inert: no capsule, worker,
operator, helper, target authorization, attempt or upload is created. It plans one
fresh full EOT8 candidate replay, using complete original-1 (worker 2) for the
same-replay original comparison and complete candidate-0 (worker 1) for exact
fresh-process replay. All three verified workers remain in the original audit;
none needs another calculation. Partial worker 3 remains failed evidence.

The plan pins every original reference artifact, header, scope/proof and final
marker. Source control identities must remain distinct from any future execution
control; old headers/markers must never be rewritten or a new result presented as
work under the consumed grant. A typed lineage reader and new result record need
separate local verification before any target handoff. A completed replay must
pass old anchors/original input bounds and exactly reproduce candidate-0 loss,
input gradient, predictions, native cotangents, targets and masks.

The two references contain 72 logical array files totaling 404,104,200 bytes.
They have 37 unique array blobs totaling 214,635,140 bytes; reference proof records
add 2,666,936 bytes. A future content-addressed capsule can store shared values
once with explicit per-reference mappings, regular files and no hard/symbolic links.
It must preserve the old metadata byte-for-byte. No capsule is constructed now;
compressed extent, extraction accounting, code and bootstrap sizes must be measured
before finalizing resources. Existing resource/financial caps are not expanded.
The prior 477.362277 seconds and consumed claim remain recorded.

Next local step: build and verify that deduplicated immutable capsule and exact
lineage reader, then prepare a concrete unarmed scope for the single missing
candidate replay using strict refusal observation. A future attempt can complete
replay or produce precise refusal metadata; it cannot bypass a refusal. Any new
remote work needs a new exact scoped approval only after preparation and review.
Do not rerun the cohort or ask repeated manual GPU checks for this local stage.
The old 3,724-entry rejection and all backend/optimizer/attack/performance/budget/
research/full-training gates remain unchanged.
