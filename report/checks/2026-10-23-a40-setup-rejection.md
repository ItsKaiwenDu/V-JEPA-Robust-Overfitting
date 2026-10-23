# A40 cost screen stopped at the CPU setup gate

October 23, 2026 (Pacific). **Setup rejected; the pretrained A40 benchmark did
not start.** The full suite reports **458 passed, one failed, 11 errors, zero
skips**, across 470 cases. The 11 errors share one failed generated CPU
replay fixture. No new A40 speed, full-model memory-fit, recovery or economic
result is available. [Structured evidence](2026-10-23-a40-setup-rejection.json).

The user selected an available A40 after A5000/A6000 capacity was unavailable
and confirmed cancelling the A5000 auto-deployment. The reviewed running quote
was $0.61/hour including a 30 GB container and 80 GB persistent `/workspace`
volume ($0.59 GPU-only). Public fingerprint verification preceded SSH access.
One NVIDIA A40 with 44.4 GiB visible VRAM, Python 3.12.3, torch 2.8.0+cu128 and
torchvision 0.23.0+cu128 passed the hardware/runtime contract. The host CPU was
an Intel Xeon Gold 6342; the container memory limit was about 50 GB. Shared
filesystem free space is not a measurement of the selected volume quota.

The six immutable package files, 24 transfer chunks and all 254 restored
source files verified before launch. The original 100 training clips were
transferred; no held-out pixel cache was included. Authenticated HTTPS was
used for transfer, with its token retained only in process memory. The
deployment was approximately 10:06 AM Pacific, corroborated by the observed
container start and the user's approximate deployment age. The supervisor's
11:55 AM cutoff retained a buffer before two billed hours and was not reset
at worker launch. The two-hour/$3 inclusive session limit was unchanged.

## What failed

`tests/test_native_diagnostic.py::tiny_run` completed four fresh CPU workers
on generated pixels and a small upstream model. Every worker finished, but
their aggregate exact replay check rejected. Within each backward policy,
the initial model-state hashes and clean-target hashes differ across the two
fresh processes despite the seeded construction. The saved gradients and
PGD-5 outputs consequently differ. This is evidence of nonidentical fixture
inputs; it does not establish nondeterminism of A40 pretrained execution or
identify the low-level initialization cause.

The second failure is the fresh-process adversarial `shared-fp64-native` CPU
recovery test. Its resume worker correctly rejected a changed resume
contract before restoring state; no `resume.pt` result was produced. The
retained uninterrupted-reference and split checkpoint contracts differ at
`target_sha256`. Their checkpoint hashes verify. This is a failed recovery
test, not a completed recovery comparison with a near-match tolerance.

Setup took 1,028 seconds, including 969.44 seconds reported by pytest. The
supervisor exited with status 1 at 10:34:31 AM Pacific and exported the
failure. The checkpoint download comes after the test gate and never ran.
None of the seven hardware-screen phases started. No attack, precision,
exactness limit, training plan, study YAML or seed count was changed to
continue past the rejection. Earlier successful and rejected GPU results
retain their original scopes.

## Evidence and next action

Both archives were copied locally and independently verified before the user
was asked to terminate the Pod:

| Archive | Bytes | Verified members | SHA-256 |
| --- | ---: | ---: | --- |
| Setup export | 46,476 | 10 | `78618873312fd736aaaeb5d82fbf01e19c1e110dd0cbeba55ca3fac051304e49` |
| Generated CPU fixtures | 11,979,409 | 17 | `453cb170cb21ea7b0cb0af86b087001ecb51b17a7cc5170a20189ae1fc6cd328` |

Local verification checks every archive member, the handoff pin, 189 unchanged
non-Markdown source files, JUnit counts, wrapped failure payload, 17 native
diagnostic source files and 32 upstream files. All four saved worker tensor
hashes/signatures verify. Re-running the summarizer on those saved files
reproduces the exact replay rejection. The small generated AdamW checkpoint
also passes its manifest check. Raw logs, generated tensors, manifests and
verification scripts remain ignored under
`runs/pretrained/diagnostic/a40-cost-screen-20261023/`.

**Lifecycle and billing:** the user confirmed termination after verification.
No provider API was queried. The final session charge remains unreported;
keep the new $3 hold and the prior unresolved $3 hold. Prior confirmed reports
total $1.195, with a separate provisional derived cumulative $2.280; these
are not reconciled with a settled invoice. Conservative unreserved shared
pilot balance is $2.805. Do not substitute the hourly quote for a final bill.

**Billing screenshot follow-up (10:43 AM Pacific):** the visible rows are
$1.644 (October 23), $1.688 (October 21), and $0.598 (October 20), summing to
**$3.930**: $3.878 GPU cloud plus $0.052 storage. This aggregate observation
is not added to earlier partial reports or assigned wholly to this A40.
The earlier $2.280 remains historical provisional arithmetic, not the current
account total. The A40-only charge, finalization, display timezone and any
reporting delay cannot be established from this screenshot. Keep both holds
pending attribution. Runpod's [Pod billing API](https://docs.runpod.io/api-reference/billing/GET/billing/pods)
supports time buckets and grouping/filtering by Pod ID. Its console has also
[documented local timestamp conversion](https://blog.runpod.io/runpod-weekly-15-new-referral-program-community-changelog-blogs/);
do not assume these date labels establish a specific UTC or Pacific day boundary.

Next, reproduce and repair the generated CPU fixture initialization/replay
locally, retaining exact comparisons and the failed evidence. Validate a
fresh package before another rental. A later A40 attempt would still need its
own quote, identity verification and deployment-based deadline; this setup
failure establishes neither affordability nor a reason to reject A40 speed.
The unchanged 20-class study remains unadmitted.
