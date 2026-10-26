# Second-host reference replay independently verified

Both fresh linux10603 PGD20/EOT4 arm-A replays match the fixed accepted
linux10609 Swing bank-1 reference exactly. The pinned external CLI review
and a separate saved-evidence audit pass without another model execution.
This is a scoped one-clip/one-arm bridge result, not a full-panel case or
automatic parallel-execution admission.

The evidence archive is 142,338,272 bytes with 33 regular members, SHA-256
`6543557a2a75cc9284698d58b666e19a5bdbab31e8d96c3a3d58bab1a65c5162`.
The complete 142,366,720-byte SSH transport and all member hashes verify.
The first copy connection closed with exit 255 and zero bytes; its failed
output is retained. The download-only retry succeeded without repeating
the GPU operation. The connection-close cause remains unknown.

| Observation | Independently verified value |
| --- | --- |
| Candidate | linux10603, RTX 3070, distinct UUID e54fae90-feeb-8106-d8c0-1974fe48555f |
| Reference | linux10609, accepted clip-02-bank-1, v_Swing_g20_c03 |
| Scope | Two fresh PGD20/EOT4 arm-A workers, two restarts each |
| Exact retained candidate tensors | 85 per worker, 170 total |
| Exact finite nonzero input-gradient observations | 40 per worker, 80 total |
| Attack times | 197.203 / 197.294 seconds |
| Peak allocator | 3.674 / 3.673 GiB, below 6.882 GiB limit |
| Actual campus CPU guards | 67 passed, zero failures/errors/skips, 2.733 seconds |
| Launcher window, October 25 Pacific | 10:55:57 PM–11:04:29 PM |
| Whole launcher time | 8 minutes 32 seconds |
| Parent session | 497.309 seconds |

All clean pixels, targets, starts, masks and selected adversaries match
bitwise. All PGD losses/gradient hashes/maxima, strict candidate selection,
independent main/two-audit scores and actual native CUDA probes verify.
The candidate records its actual host/GPU under its own distinct execution
identity; the reference is not relabeled. Full runtime/model/checkpoint
contracts and all 198 original pins, 58 qualified sources, 34 upstream files
and four new shadow source/fixture pins match.

Both worker exits, final session/launcher/SSH exits, JUnit and actual timing
records pass. The one-attempt twenty-minute limit and fifteen-second grace
are respected, with no automatic GPU retry or old recovery-grant reuse.
The local review acceptance binds plan, archive, comparison, identities and
independent report, and authorizes no further computation or dispatch.

The frozen v1 plan, old cases, receipts, partial evidence and historical
FP32 failures remain unchanged. The panel stays at seven of twenty verified
cases, thirteen remaining; TennisSwing bank 1 is next on its original host.
No partial-panel aggregate, all-clips/EOT8 qualification, new FP64 oracle,
research precision adoption or A40 cost pooling is claimed.

Both of our project jobs have ended and their evidence is verified locally.
The user requested stopping after this verification to sleep. No further
work is launched. On return, prepare an explicit two-host execution revision
and test coordinated workers/journals before any additional panel dispatch.
Keep whole clip pairs on one host and preserve all original artifact identities.

[Machine-readable verification](2026-10-25-cross-host-shadow-verification.json).
