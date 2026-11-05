# CPU control passes; target and control saved arrays agree exactly

The new linux10609 control **passes independent review**. Its four unchanged
original workers complete and exit 0. A fresh local audit verifies all 2,048
stage comparisons and 48 final output/VJP comparisons at the original
`rtol=1e-12`, `atol=1e-14`. The complete work phase takes 62.677 seconds.
The one-attempt grant is now consumed; never repeat its attended helper.

The matched linux10603 target and linux10609 control have **2,112 bitwise-equal
array pairs**, with zero cross-host absolute error. This is agreement for the
fixed CPU-only reference capture. The original GPU-initialized setup is not
reproduced, and the historical approximately 1e-10 discrepancy remains unresolved.

## Numerical and cross-host evidence

The host sequence is serial: the accepted target finishes and is independently
reviewed before the separately authorized control. Both use the same frozen
19,042,363-byte packet and fixture `[1,16,400,64]`, with workers in order
`(threads,replay) = (4,0), (1,0), (4,1), (1,1)`. No worker regenerates inputs.
All three reference calls per worker save values and preserve their CPU-only
context. The control's CUDA contexts remain uninitialized before and after.

| Independent control check | Result |
| --- | --- |
| Original workers / captured calls | 4 / 12 complete |
| Captured chunks / logical signatures | 256 / 2,112 verified |
| Stage comparisons | 2,048 pass |
| Fixed final output / VJP comparisons | 48 pass |
| Within-worker phase byte comparisons | 32 equal |
| Cross-worker byte comparisons on control | 1,584 equal |
| Maximum independent absolute error | 2.498001805406602e-16 |
| Maximum error-to-tolerance ratio | 0.007506069824549864, below 1 |

Matched target/control comparisons load and validate both hosts' actual arrays,
then compare bytes and compute agreement at the unchanged bounds. Byte equality
is an observed finding; numerical acceptance bounds are not altered.

| Matched cross-host array group | Pairs | Bitwise equal |
| --- | ---: | ---: |
| Converted FP64 inputs | 16 | 16 |
| Observed intermediate stages | 2,048 | 2,048 |
| Final outputs and VJPs across three phases | 48 | 48 |
| Total | 2,112 | 2,112 |

Both hosts record Python 3.11.17, PyTorch 2.8.0+cu128, NumPy 2.4.6,
AVX512 and an Intel Core i9-11900 CPU. For corresponding workers, recorded
thread/interop counts, CPU count, read-only affinity, task environment,
precision settings, selected CPU information and library configurations match.
Full contexts are not identical: hostname, observed scheduling CPU and kernel
patch version differ. Linux10603 records `5.14.0-687.54.1.el9_8`;
linux10609 records `5.14.0-687.58.1.el9_8`. These differences are preserved;
their effect on the historical failure is not established. Whole-launcher
durations are descriptive and do not constitute a controlled speed benchmark.

The remote immediate and final audit decisions agree with the new local audit.
Their context-dependent diagnostic statistics remain separate. The remote
parent differs from each one-thread producer in 354 stage statistics, while
the Mac review has its own recorded numerical-statistic differences. None
changes the comparison decisions, bounds, array coverage or saved tensor bytes.
No exact nested-statistic equality requirement or original refusal is waived.

The independent local review uses Python 3.11.16 / PyTorch 2.14.1 / NumPy 2.4.6
on arm64, with process and network creation denied. It reads saved arrays;
it starts no original reference worker, SDPA, model or GPU calculation.

## Storage, transport and provenance

The real XFS proof writes, fsyncs and reads back all 2,684,354,560 bytes in
9.230 seconds. All 1,926 events verify in order. The successful proof removes
only its own proof file, as authorized. The proof and reference supervisors
record 95 and 484 samples, exit 0 and no descendant requiring termination.

| Observed control allocation | Peak | Original ceiling |
| --- | ---: | ---: |
| Setup | 2,701,770,752 bytes | 3,221,225,472 bytes (3 GiB) |
| Capture | 1,805,107,200 bytes | 2,415,919,104 bytes (2.25 GiB) |

The per-file kernel limits remain 3 GiB for the proof child and 1 GiB + 8 MiB
for the capture child. Events and supervisor samples establish observed
accounting, not a continuous aggregate kernel quota, exclusive capacity
reservation or future headroom guarantee. The original component's
`live_XFS_qualified: false` flag stays untouched. This review records that this
particular full-size attempt meets its checks. The removed random proof bytes
are not independently reread on the Mac; their reported digest and consistent
source/event/receipt chain are preserved in the JSON review.

All 454 outer regular members, their exact nanosecond modification times and
1,353,594,571 logical bytes verify against both the response stream and saved
tree. The full response is 1,353,955,857 bytes. All 431 numerical export members
verify against their sidecar and saved files; that export is 639,941,082 bytes.
The unchanged original packet has 129 members. All eight overlays (109,322
bytes), 61 scientific source bindings, 33 upstream bindings and 131 immutable
history bindings verify. Control saved files are rehashed after numerical
review. The accepted target's 453 saved files, exact timestamps, complete
response, source and audit artifacts reverify before and after the comparison.

| Record | SHA-256 |
| --- | --- |
| Control | `002e0dabbd39a7871ea998f4b19eae0f578dfa5225c3021753c382dbcfab0136` |
| Authorization | `c4858a2e49e1cf0d8fda22b43a85eee86564dea0622daa112638c4c9340f7b40` |
| Consumed attempt | `6ed5c4941a3eba87cd70a5aa5381234acfc494b77f3044b8d42a866630996b79` |
| Complete response | `953d5117dec2fd2a5882ec183a933d1e4f9b1fad69be0c45ee0f52ef281f6dbf` |
| Numerical export | `b2c4213c61aaac2f3dd3371e7cd5f9fa3fb84ead961f1c61be735603c2cd804f` |
| Accepted target review | `783a390b9eef87b1fe9e7d700ea97bb39677330e07d3f551652e662246514c3c` |
| New independent evidence review | `7c3b33aa86e26e29c6a6857de1dfff1d0877a2e2bd7e7d940aca0729bb39c296` |

Raw evidence, executed source, array audits and detailed cross-host/context
comparisons stay outside Git under
`runs/pretrained/diagnostic/cpu-oracle-control-v1-20261104/`;
the new review directory is `independent-control-review-20261104/`.
Remote scratch remains at
`/var/tmp/vjepa-cpu-control-v1-20261104-linux10609-v307_rnw`.
No remote cleanup is authorized or performed by this review.

## Interpretation and next step

Both current CPU-only captures pass and agree exactly under the recorded
conditions. This does not retroactively accept the original v1 parent protocol
refusal, v2 startup failure or original fixed-input GPU-context worker. Their
sources, claims, reports and arrays stay unchanged. The original failed CPU
derived arrays were not saved, so the later passing values cannot be compared
directly with that failed calculation. The current CPU-only probes deliberately
omit that worker's CUDA initialization; no cause is assigned to this difference.

Next perform an **offline comparison of the preserved historical worker setup
and these passing CPU-only captures**. Identify the still-unobserved setup
differences before proposing any separately scoped reproduction. No new
university attempt, GPU job or paid compute is prepared or authorized here.
Both current grants are consumed; both hosts can remain idle. Old control
authorization/claim and old acceptance bridge stay absent at their old paths.
Do not repeat helpers, reset claims, extend allowances or delete remote scratch.
General host reliability, GPU/backend qualification, optimization adoption
and research admission remain unestablished.

Financial state remains unchanged: $200 project / $160 compute / $40 other
reserve, one $10 diagnostic envelope, $9.8116 conservative encumbrance and
$0.1884 unreserved diagnostic allowance. The historical $2 hold, $0.0792 gap,
last reported $6.56 wallet and incomplete reconciliation remain. No hold is
released. User commits themselves; no staging, commit or push is performed.

See the [machine-readable review](2026-11-04-cpu-oracle-control-v1-review.json),
[accepted target review](2026-11-04-cpu-oracle-capture-v3-target-review.md),
[historical fixed-input review](2026-10-30-fixed-attention-review.md),
[control authorization](2026-11-04-cpu-oracle-control-v1-authorization.md)
and [operator documentation](../../docs/cpu_oracle_control_v1.md).
