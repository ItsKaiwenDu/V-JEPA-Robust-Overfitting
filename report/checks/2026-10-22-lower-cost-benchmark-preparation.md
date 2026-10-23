# Lower-cost GPU benchmark: offline preparation verified

October 22, 2026. **Ready for the user's reviewed manual deployment; no new
GPU is running.** [Structured evidence](2026-10-22-lower-cost-benchmark-preparation.json).
The user chose to keep the 20-class study and supplied a **$0.29/hour** quote
for one RTX A5000 with 30 GB container and 80 GB persistent `/workspace`.
Availability and host RAM were not explicitly reported. The public listing
and its limits are in the [catalog check](2026-10-22-lower-cost-gpu-catalog.json).

The [benchmark guide](../../docs/lower_cost_gpu_benchmark.md) preserves the
original data, losses, threat model, attacks and seed/epoch schedule. A new
seven-phase screen reuses numerical preflight, calibration, two clean and two
adversarial reference updates, features/probes and PGD-50/FGSM/random controls.
The trained feature check explicitly loads the reference anchor; original
split-anchor defaults remain unchanged and are covered by the related tests.
Split/resume recovery is **not** part of this screen; its result cannot admit
the study or replace the full RTX 5090 recovery evidence.

All **72 targeted CPU tests pass** (101.33 s); the fresh restored snapshot
separately passes **three selected CPU cases**, including the complete real
small-model seven-phase path (29.31 s). Both shell scripts pass syntax checks.
These are not new pretrained CUDA performance, memory-fit or recovery results.

The package contains **254 source files**,
102 training-cache members (100 clips plus metadata), 23 existing Linux wheels,
and the original pinned upstream Git objects. All restored source/member hashes
verify. Size: 815,135,208 bytes. Manifest SHA-256:
`5865a2dcc1a5eaae2c7989f5b56b4a00e41aeb850d86d64f50578c73547f6544`.
Earlier archives are unchanged; no held-out pixel cache, credentials or weights
are included in source. The checkpoint remains separately hash-verified.

At the displayed rate, two billed hours are $0.58. Keep the existing **two-hour,
$3 inclusive maximum**, deployment-based absolute deadline and verified export
before termination. The conservative $5.805 pilot balance can cover the proposed
$3 hold; it leaves $2.805 unreserved while the prior charge remains provisional.
No new spending or provider state is claimed by this preparation.

At $0.29/hour, the existing maximum-observed timed units could be about
**2.50× slower** and still meet the planning arithmetic, retaining the
old unverified overhead allowance and reserves. This is a requirement to test,
not an A5000 speed prediction or complete-study cost bound. The eventual screen
uses fresh measurements and retains all research admission gates.

Next: manual Pod `vjepa-cost-benchmark-04` when available and attended, then
verify the new public SSH host fingerprint, bind the deadline to deployment,
transfer/restore the frozen package and run the bounded screen. Copy and verify
the export while accessible, then stop/terminate and reconcile charges.
