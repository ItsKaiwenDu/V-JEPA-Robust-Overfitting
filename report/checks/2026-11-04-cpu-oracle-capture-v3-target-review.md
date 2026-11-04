# V3 CPU target capture: independent review passes

The new linux10603 attempt **passes the controlled v3 CPU capture protocol**.
The parent exits 0, all four unchanged original workers complete, and an
independent Mac review recomputes every saved stage and final comparison at
the original `rtol=1e-12`, `atol=1e-14`. The actual full-size storage proof and
observed capture allocation also stay within their original ceilings.

The entire remote work phase takes **48.732 seconds**, including setup, proof,
workers, audits and export. Evidence collection follows on the same attended
SSH connection. The one-attempt grant is consumed; do not rerun its helper.
This review makes no SSH connection and starts no original reference worker,
model, SDPA, GPU task or paid compute.

## Numerical evidence

The original full fixture has shape `[1, 16, 400, 64]`. The planned workers run
in order `(threads, replay) = (4,0), (1,0), (4,1), (1,1)` on linux10603, with
Python 3.11.17, PyTorch 2.8.0+cu128 and NumPy 2.4.6. Their before/after receipts
confirm CPU-only execution and unchanged runtime settings. The new local
review uses Python 3.11.16, PyTorch 2.14.1 and NumPy 2.4.6 on arm64; process
and network creation are explicitly denied throughout that review.

| Independent check | Result |
| --- | --- |
| Original workers / captured oracle calls | 4 / 12 complete |
| Captured chunks / logical array signatures | 256 / 2,112 verified |
| Stage comparisons | 2,048 pass |
| Final output and VJP comparisons | 48 pass |
| Within-worker phase byte comparisons | 32 equal |
| Cross-worker byte comparisons | 1,584 equal |
| Maximum absolute error in independent checks | 2.498001805406602e-16 |
| Maximum error-to-tolerance ratio | 0.007506069824549864, below 1 |

The immediate and final remote v3 audit **decisions** agree with the new local
array audit. Diagnostic floating-point statistics are preserved with their
computing context. The four-thread remote parent differs from the one-thread
producer in 354 recorded stage statistics for each one-thread worker; the Mac
review differs in 843 or 844 statistics from each producer and 844 from each
remote parent audit. These statistic differences do not change any decisions,
coverage, shapes, bounds or captured tensor bytes. Exact nested diagnostic-float
equality is not substituted for checking the saved arrays.

## Storage and transport

The XFS scratch directory is on device 66307, separate from shared home storage.
The proof writes, fsyncs and reads back all **2,684,354,560 bytes (2.5 GiB)** in
9.499 seconds. All 1,926 allocation events verify in order: 640 writes, 640
readbacks and the surrounding preallocation/fsync/close/removal boundaries.
The successful proof removes only its own proof file as specified.

| Observed allocation | Peak | Original ceiling |
| --- | ---: | ---: |
| Setup | 2,701,758,464 bytes (2.516 GiB) | 3 GiB |
| Capture | 1,805,094,912 bytes (1.681 GiB) | 2.25 GiB |

The proof and capture supervisors record 97 and 352 samples, respectively,
with exit 0, no retry and no descendant requiring termination. Their kernel
per-file limits remain 3 GiB and 1 GiB + 8 MiB. The checked event and supervisor
records are observed accounting; they do not establish a continuous aggregate
kernel quota, exclusive capacity reservation or future free-space guarantee.
The proof's component-level `live_XFS_qualified: false` field remains untouched.
This report records that this particular full-size XFS attempt passed its checks.
The removed random proof data is not independently reread on the Mac; its
success follows the bound executed source and consistent saved event/receipt
chain. The proof's reported data hash remains in the JSON review.

All **453 outer members**, their exact nanosecond modification times and their
1,353,583,842 logical bytes verify against both the response stream and saved
tree. The complete response is 1,353,945,364 bytes. All **431 numerical export
members** verify against the export sidecar and saved files; the export is
639,941,016 bytes. The unchanged input archive has 129 members and 19,042,363
bytes. All seven uploaded overlay files, 61 scientific bindings, 33 upstream
bindings and 115 immutable local history bindings verify. Every saved file
is rehashed again after numerical review.

## Provenance and preserved limits

| Record | SHA-256 |
| --- | --- |
| Target control | `51f8c3215411d2347d58895db3cce18be6b9ac977098c9c3197217d173b509b7` |
| Authorization | `e51f2925c6461ccd771f4cb10839270f189b2a77b39f0bd9fc79a4900335dd26` |
| Consumed attempt | `6396015bdd55750d18ba2f44bde79d9f18becaa35ff2d1270b7c937dbd7af704` |
| Complete response | `f467312d9135fd474e7b1717ba2455f4dcb90df44d473e5e623e7ed1b0fdce15` |
| Numerical export | `4597b30308a3ca1b61748a75643ea1716b8aa0fcbcd7d28695e47743fc51e287` |
| New independent evidence review | `33422c0d7c29af28880508c6dd2852db3219bed9bb1d44c61961cc07a99f0263` |

Raw transport, captured source, runtime records, arrays, independent audits and
the review script remain under
`runs/pretrained/diagnostic/cpu-oracle-capture-v3-20261104/` outside Git.
The independent review directory is
`independent-target-review-20261104/`.
The retained remote scratch path is
`/var/tmp/vjepa-cpu-capture-v3-20261104-linux10603-3k0ulh7w`;
no deletion is authorized or performed.

The original [v1 protocol refusal](2026-11-04-cpu-oracle-scratch-target-review.md)
and [v2 startup failure](2026-11-04-cpu-oracle-capture-v2-target-review.md), their
source, claims and arrays remain unchanged. The original v1 parent is not
invoked or retrospectively accepted. The historical approximately 1e-10
discrepancy is still unresolved. This result establishes neither general host
reliability nor GPU/backend qualification, optimization adoption or admission
of a research result.

No control-host helper or grant is prepared. The old linux10609 authorization,
attempt claim and acceptance bridge remain absent. Both hosts can stay idle.
Next prepare and test a **separately identified, unarmed control-host capture
handoff**, then review its concrete resource scope before seeking a separate
attempt authorization. Do not reuse old grants or automatically start a run.

Financial state is unchanged: $200 project cap, $160 compute cap, $40 other
reserve, one $10 diagnostic cap, $9.8116 conservative encumbrance and $0.1884
unreserved diagnostic allowance. The historical $2 hold, $0.0792 gap and last
reported $6.56 wallet remain; account reconciliation is incomplete. No hold
is released and no paid work starts.

See the [machine-readable review](2026-11-04-cpu-oracle-capture-v3-target-review.json),
[preparation record](2026-11-04-cpu-oracle-capture-v3-preparation.md),
[one-attempt authorization](2026-11-04-cpu-oracle-capture-v3-authorization.md)
and [operator documentation](../../docs/cpu_oracle_capture_v3.md).
