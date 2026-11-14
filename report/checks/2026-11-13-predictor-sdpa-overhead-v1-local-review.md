# Separate overhead and GPU-range metadata policy verified offline

78 focused checks pass with zero failures, errors or skips. The new verifier is
separately versioned; all frozen validators, scientific workers and consumed
handoffs stay unchanged. It retains every original raw row/index and adds roles
in a separate projection. The complete backend/phase/autograd validator body is
AST-identical to the frozen version. No numerical tolerance, reference, runtime,
startup, device, kernel-family, no-fallback or replay rule changes.

The narrowly supported CPU buffer-request signature retains device index -1 and
cannot own links or parent operations. A GPU range must carry its annotation flag,
match one of our phase names and its real CPU annotation's recorded correlation,
and satisfy exact runtime/device/field predicates. It cannot count as a kernel.
Unknown or malformed records still refuse; no negative-ID blanket allowance or
name-only dropping is used. Source details and exact predicates are in
[the policy document](../../docs/predictor_sdpa_overhead_v1.md).

The unchanged 106-row university trace passes a separate saved-metadata assessment.
Row 10 remains overhead; row 20 remains a GPU range. Twelve distinct positive raw
CUDA witnesses resolve to genuine CPU owners, with the frozen cross-thread
producer-chain check. Row 9, a real attention operation, remains the owner for
correlation 10 even though overhead row 10 carries that same number. The raw
SHA256 stays `d1b2006c583c6cfc4d8dc0bb0b12e823293222079c740d6d78acb78a72382686`.
This is metadata validity under a new offline policy, not completed CUDA execution,
resource/replay evidence or adoption. The historical worker's original rejection,
false completion/flags/input/profile markers, missing complete/resource receipts,
and unrun components 1/2/3 remain unchanged. The old validator still refuses.

Official versioned source supports both metadata cases: Kineto overhead uses
negative device identity; PyTorch may inherit its parent's correlation number;
and Kineto creates separate GPU user ranges from CPU annotation identities and
GPU device/stream fields. The raw annotation flag distinguishes such ranges.
This does not prove the installed binary source commit or authenticate fabricated
metadata. No unavailable native-kind witness is claimed. Source receipts are
preserved, with direct versioned links in the policy document.

An initial local assessment rejected a shared overhead/frontend correlation number;
its exact sources and refusal are saved. The unnecessary global exclusion was
removed: only real frontend roles enter the ownership map. The first test run
then had 61 passes and 3 failures. One synthetic same-name CPU fixture mistakenly
shadowed an autograd producer; two real-trace assertions exposed the separate GPU
phase annotation. The fixture was corrected and narrow GPU-range support plus
regressions added. The final 78 checks are not accumulated with earlier counts.
All failed logs, XML, test artifacts and tested source snapshots remain.

Independent standard-library saved review blocks scientific imports, profiling,
processes and network. It rechecks 66,754 prior hashes and all 192 current target
byte/mode/nanosecond-time descriptors, compares the actual/synthetic saved proofs,
verifies final test counts and unchanged backend/autograd code, and confirms all
historical claims remain consumed. It starts no model, derivative, SDPA, profile,
CUDA or reference work and repeats no tests. No SSH/upload/retrieval, cleanup,
installation, paid compute or held-out pixels are used. The rejected candidate,
financial caps/holds/pending reconciliation and full-training gates stay.

Next integrate this separate policy into a separately versioned component worker,
keeping capture-before-validation and every numerical/reference/startup/resource/
input/flags/fresh-replay gate. Test the complete local integration before any new
university handoff. There is no new target control, grant or attempt at this stage.
