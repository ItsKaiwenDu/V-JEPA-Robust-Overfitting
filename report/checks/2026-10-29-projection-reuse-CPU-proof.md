# Projection-reuse CPU proof and cost target — October 29

The separate CPU-only prototype passes **32 new checks**. Together with 77 original
candidate checks and 31 original precision checks, **140 focused checks pass**
with no final failures/errors/skips in their recorded process groups. All original
61 scientific source/config/data pins and existing runtime guards remain unchanged.

One identical-input mask bank computes the original FP64 projection once while
retaining every per-request FP32 convolution backward and original normalization
graph. Exact tokens/layouts, input/parameter derivatives, actual tiny upstream
PGD20/R2/EOT8 adversarial pixels/losses/RNG and synthetic two-update AdamW recovery
pass. The tiny upstream bank has 16 projection requests; one forward replaces
16 repeated forwards. No draw, iteration, restart or scoring sample is removed.
This is a work-count proof, not a measured end-to-end speedup.

The initial mixed-process regression run retains 137 passes and one runtime
mismatch refusal after precision fixtures. The new tests restore their own flags,
RNG and threads. The original precision checks run in a separate process because
the legacy parent/fresh-child fixture boundary expects consistent settings. No
test is omitted, no runtime guard/tolerance changes and no repository-wide
fixture-order fix is claimed. The original failure XML and parent contract remain.

The SE timing-based cost model requires about **1.46–1.53×** faster primary work
if every other measured category and the unvalidated eight-hour placeholder stay
fixed. Ideal 16-way reuse would require 33.6–36.7% of primary time in eligible
repeated forwards; actual time share and clone/guard overhead are unknown. This
does not establish affordability, a full-study bound or GPU qualification.

The prototype is not connected to any runner and rejects CUDA. A test-only model
adapter is rejected by the existing model-provenance gate. No weights/study pixels,
SSH, university GPU, paid work, protocol adoption or research trajectory were used.
Next prepare a separately identified/source-bound correctness and timing diagnostic
for free university hardware, then measure before any paid benchmark decision.

The RTX 3090 charge is now settled to the user-reported **$1.1328** and available
wallet **$6.56**. Internal release is $0.1672, leaving $0.1884 unreserved in the
single $10 diagnostic envelope. The older $2 hold and historical gap remain;
$200 project/$160 compute caps are unchanged. Full account reconciliation is pending.

See the [prototype and qualification limits](../../docs/projection_reuse_cpu_prototype.md),
[machine-readable proof](2026-10-29-projection-reuse-CPU-proof.json), and
[separate billing settlement](2026-10-29-rtx3090-billing-settlement.md).
