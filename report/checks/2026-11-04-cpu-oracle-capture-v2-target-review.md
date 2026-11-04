# V2 target review: storage passes, launcher import fails — November 4, 2026

The complete saved response is independently verified. The real 2.5 GiB
allocation/write/fsync/readback proof succeeds within the original 3 GiB setup
ceiling. The packaged reference parent then exits on a missing import path
**before any original worker or reference call**. The full CPU probe remains
incomplete. This attempt is consumed and must not be repeated.
[Machine-readable review](2026-11-04-cpu-oracle-capture-v2-target-review.json).

## Evidence and storage

All **149 streamed members**, 40,534,908 logical bytes, verify independently
against both the response archive and copied files, including exact modification
times. All 129 original packet members, seven overlay sources, control,
authorization and claim bindings match. Response size is 40,676,995 bytes;
SHA256 is `7266b2d4a0d8be40fe9fbb03c654aaaafd1b58db6feab8346d5eb7ba4682f15c`.

The actual OS allocator is used without a test override. The proof records
2,684,354,560 bytes written and read back, successful fsync/hash agreement,
and exactly 2,684,354,560 allocated proof-file bytes before close. The setup
peak is **2,701,754,368 allocated bytes (about 2.52 GiB)**, below the unchanged
3,221,225,472-byte ceiling. All 1,926 allocation events verify, with 640 write
blocks and 640 read blocks. The proof takes **9.487 seconds**; the supervisor
records 97 storage samples and exit zero. The whole run phase lasts 14.102 seconds.

Only the successful proof and its new operational reserve are removed by their
own operation. The random proof file was intentionally not retained, so its
bytes cannot be independently rehashed on the Mac; this review authenticates
the exact executed source, receipts and event sequence. Observations are not a
continuous kernel quota, exclusive reservation, future-capacity guarantee or
qualification of the complete capture/export workflow.

## Packaged-launch failure

`reference.log` ends with `ModuleNotFoundError: No module named 'attacks'`
while importing `run_workers`. The overlay makes the original `models`
directory visible through `models.__path__`, but omits the original packet root
from `sys.path`. Sibling top-level packages are therefore unavailable in a cold
launch. The previous local tests imported from the checkout, masking this
packaging error. This was a launcher/test-coverage defect.

The reference parent exits 1 after 3.316 seconds. No `results` directory, worker
authorization, runtime session, computed array or new evidence export exists.
The outer `reference_process_started=true` refers to that parent process; it
does not mean an original worker ran. Its null technical-error field is not an
acceptance decision. This review records the import failure and rejects the full
probe. No numerical finding, CUDA qualification or explanation of the historical
approximately 1e-10 discrepancy follows from this attempt.

## Local correction and new regression coverage

Four new isolated regression checks pass with zero failures, errors or skips.
They restore only bound source files, use a neutral working directory and cold
Python, reproduce the exact original missing-`attacks` failure, and verify a
candidate that places the original sibling packet root after the overlay on
`sys.path`. The candidate resolves `attacks` and `datasets` from the original
packet and retains overlay `models` precedence. Wrong or symlinked packet roots
refuse.

The harness imports the real runtime, then blocks worker dispatch. It explicitly
denies subprocess creation and original oracle/worker calls; no tensor fixture,
model or numerical result is generated. These checks qualify the local import
correction only. A candidate source/diff is retained beside the raw review; it
is not an armed remote operator. Frozen v2 entrypoint, runtime, supervisor,
packet and reports remain unchanged.

## Preserved state and next step

All 98 immutable bindings and original sources/arrays/reports reverify. Target
control remains `64364f84133035c6fff3d5fbed87c64f7a8ec92203e202ec20d70456fe300484`;
its new `ATTEMPT.json` exists and is consumed. The remote directory
`/var/tmp/vjepa-cpu-capture-v2-20261104-linux10603-95smn4j9` remains; no scratch
cleanup is authorized. Old failed captures and grants remain preserved.
Linux10609 stays unarmed, with no accepted target-review bridge or new helper.

This review makes no SSH connection, new university calculation, allocation
test, remote deletion, paid launch or financial release. Conservative diagnostic
encumbrance stays $9.8116, leaving $0.1884 in the single $10 envelope. The $2 hold,
$0.0792 gap, $6.56 last reported wallet, incomplete reconciliation and
$160 compute/$200 project/$40 other-cost limits remain unchanged.

Next prepare and test a separately identified packaged launcher correction.
Only after that concrete operator is reviewed may another target resource
exception be considered. Neither the consumed v2 helper nor a control-host run
should start now.
