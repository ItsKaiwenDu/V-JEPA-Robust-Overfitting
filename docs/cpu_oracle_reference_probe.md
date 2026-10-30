# CPU reference reproducibility probe

Latest update: [approved archive cleanup is complete and verified](../report/checks/2026-10-30-duplicate-archive-cleanup-review.md).
Twenty-four duplicate compressed exports were removed, with verified Mac backups
retained. Small temporary writes work again; full-capture quota headroom remains
unknown. Both reference probes and the cleanup claim are consumed. Prepare a
separate storage-aware recovery plan; never repeat the old launchers. Earlier
entries below document historical preparation, failures and recovery.

Latest update: both interrupted captures are backed up and independently
reviewed. Available complete saved portions meet unchanged bounds, but both
full probes remain incomplete and the earlier discrepancy remains unresolved.
See [the backup review](../report/checks/2026-10-30-cpu-oracle-salvage-review.md).
An exact 24-archive storage cleanup proposal is prepared, awaiting explicit
approval; no deletion or new probe has started. Entries below are historical.

Latest status: both CPU captures are incomplete. A saved 10603 traceback
reports disk quota exceeded; 10609 stops during a save without a preserved
traceback. The read-only metadata is verified, but calculated tensors await a
copy-only backup. See [the storage-quota status review](../report/checks/2026-10-30-cpu-oracle-status-review.md).
Both launch claims remain consumed. No deletion or retry has started.

Current status: both new Mac launch claims are consumed, but neither local
download contains a verified archive. Use only the prepared read-only status
helper after the [pending retrieval review](../report/checks/2026-10-30-cpu-oracle-retrieval-pending.md).
Numerical outcomes and failure causes remain unknown; never repeat a launch to
recover files. The sections below document the frozen plan and preparation.

The previous fixed-input encoder-1 run on linux10603 rejected CPU FP64 reference
bookkeeping before transferring inputs to the GPU math backend. The passed and
failed workers had identical saved inputs, but the failed calculated FP64 arrays
were not saved. Their reported error cannot identify an operation or establish
a cause. This separate forensic probe preserves those arrays and intermediate
values before numerical checks. It changes neither the original reference math
nor its 1e-12 relative / 1e-14 absolute bounds.

## Fixed input and observation

The packet copies the original encoder-1 fixture and eight NumPy arrays byte for
byte: Q/K/V/cotangent and fixed FP64 output/Q/K/V references, shape
`[1,16,400,64]`. It includes no pixels, pretrained weights, trained activations,
dependency wheels, private keys or financial ledger. Workers never regenerate
inputs or fall back to another fixture. The original 61 scientific source pins,
33 upstream bindings and eight prior consumed claim/plan files remain unchanged.

Each fresh worker calls the original pinned `fp64_oracle` three times: unobserved
before, observed, then unobserved after. All three outputs and derivatives are
saved before numerical acceptance checks. The observer uses Python line tracing
on that function alone and restores the prior trace state. It refuses to replace
an existing trace hook. The original function and tensor calculations are not
rewritten or monkeypatched in the university packet.

For the observed call, it saves converted FP64 inputs and all 64 query chunks
(16 heads, query chunks of up to 128 tokens). Each chunk contains centered
scores, probabilities, output, probability cotangent, score cotangent, Q
derivative, accumulated K derivative and accumulated V derivative. Partial files
are retained if capture is interrupted. Saved inputs, outputs and intermediates
have bounded NumPy headers/extents, SHA256 hashes and logical signatures. A
separate NumPy calculation rechecks every captured stage; fixed references check
all three final calls. A first captured mismatch is an observation boundary,
not a proven faulty operation or hardware diagnosis.

## Predetermined scope

Linux10603 is the target and linux10609 the control. Each new, separately claimed
attended plan allows exactly this ordered sequence of fresh processes:

| Worker | CPU threads | Replay label | Reference calls |
| --- | ---: | ---: | --- |
| 0 | 4 | 0 | before, observed, after |
| 1 | 1 | 0 | before, observed, after |
| 2 | 4 | 1 | before, observed, after |
| 3 | 1 | 1 | before, observed, after |

Each host has a 600-second allowance from remote extraction, a 90-second limit
per worker, five-second final kill grace, no automatic diagnostic retries and
a 1 GiB uncompressed evidence-export cap. Worker rejection completes the three
already planned calls and saves their evidence, then stops before any next
worker. Technical failure or timeout also stops and preserves partial evidence.
No repeat-until-pass loop or original-claim reset exists. Capture integrity,
complete task coverage and numerical agreement are separate result fields.

`CUDA_VISIBLE_DEVICES` is empty before importing the probe. The launcher runs
only CPU calculations, never queries `nvidia-smi`, initializes CUDA, executes
SDPA, loads a model or starts an optimizer. Existing university Python 3.11.17,
PyTorch 2.8.0+cu128 and NumPy 2.4.6 must match; there is no install/fallback.
Each worker records CPU/library build details, thread counts, affinity read-only,
CPU location and floating-point rounding mode when available, precision flags
and whether CUDA was initialized. It changes no affinity or server settings.

This deliberately omits the earlier GPU worker's CUDA initialization/context.
Observation and file writes may also perturb execution. The three phases and
their bitwise equality are recorded rather than assumed. A short clean CPU-only
sample cannot explain the historical discrepancy, establish reliable hardware,
qualify GPU attention or waive any existing refusal. Numerical disagreement
does not by itself identify whether Torch, NumPy, a library, hardware or another
cause is responsible.

## Attended university handoff

The ignored local directory is
`runs/pretrained/diagnostic/cpu-oracle-university-20261030/`. Its two host folders
contain **new** `run-attended.sh` helpers and unarmed proposals. Execute each
helper once from its own **local Mac Terminal**, keeping Terminal/SSH open,
the lid open and the Mac awake through download. Both CPU probes may run in
parallel on the different hosts. There is no GPU status check needed here.

Explicit manual helper invocation activates only that new host's CPU plan.
The helper verifies the known university public host key, refuses an existing
remote directory, uploads the source/fixture packet, and keeps the foreground
SSH connection attended. It has one upload-connection recovery before launch
and one download-connection recovery afterward, neither of which repeats CPU
work. Download uses ordinary SSH with up to ten minutes per password/transfer
step; enter passwords promptly. Results may be about 640 MB per full host run,
so download time is additional to the ten-minute CPU allowance.

If transfer fails, preserve partial folders and report the result. Use only the
new host's `download-only.sh` after review, never rerun `run-attended.sh`, remove
a claim or restart an old fixed-attention job. Downloaded capture is not accepted
as a result until its archive, calculations and host/library context receive
independent review. Earlier failures and their evidence remain frozen.

## Offline verification

All 33 focused CPU checks pass with zero failures, errors or skips. They include
four fresh tiny workers, partial/multiple-head query chunks, trace restoration,
changed evidence/domain/history refusal and an injected mismatch that retains
all three calls and its actual trace. The full 400-token local CPU rehearsal
completes all four fresh workers: 12 calls, 256 captured chunks, 2,048 stage
comparisons and 48 final comparisons against fixed references. No CUDA context
was initialized. Its independently restored 423-member archive is 639,754,844
bytes; its uncompressed evidence remains below 1 GiB. This is a Mac rehearsal,
not a university-host qualification. Both hosts' four simulated transport
scenarios pass without network connections. The final packet restores 129
members and matches the CPU-tested new sources.

No real university connection, upload or CPU/GPU attempt starts during
preparation. Financial state stays unchanged: single $10 diagnostic allowance
inside $160 compute / $200 project caps, $40 noncompute reserve; $9.8116 remains
conservatively encumbered and $0.1884 unreserved. The historical $2 hold and
$0.0792 unassigned gap remain; account reconciliation is incomplete. Last
reported wallet is $6.56. No new paid resources or research execution starts.

See [preparation evidence](../report/checks/2026-10-30-cpu-oracle-preparation.md)
and [the prior mixed review](../report/checks/2026-10-30-fixed-attention-review.md).
