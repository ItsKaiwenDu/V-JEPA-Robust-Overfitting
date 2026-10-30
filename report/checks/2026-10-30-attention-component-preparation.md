# FP32 attention component preparation — October 30, 2026

**Prepared and unrun.** The next diagnostic tests synthetic post-RoPE attention
forward and Q/K/V derivatives on the shapes found in the reviewed evaluation
profile. Predictor work is assigned to linux10609; the two encoder shapes are
assigned to linux10603. The independent workstations can run in parallel, with
each backend comparison confined to one GPU.

All 54 focused CPU checks pass, with zero skips. Six fresh tiny CPU workers
replay exactly, satisfy the independent FP64 oracle and retain actual math
backend observations. Full-size synthetic captures reproduce the 2048-token
predictor and 656/400-token encoder geometries, including original mask-position
signatures and upstream layouts. CPU fixtures do not qualify a CUDA backend.

The final source-only archive is about 291 kB and restores all 103 regular
members. Its production manifest equals the final tested manifest. All 61
original scientific pins and 33 upstream source bindings remain unchanged.
There are no pixels, pretrained weights, trained activations or dependency
installations in this diagnostic. The [machine-readable preparation record](2026-10-30-attention-component-preparation.json)
contains exact hashes, limits, test receipts and financial state.

Eight local transport scenarios pass across the two helpers: ordinary execution,
one prelaunch upload connection retry, one download-only connection retry and
refusal of an existing remote directory. Exclusive local claims prevent repeat
launches. The standalone download helper reads saved files without model work.
No real SSH connection, upload or university GPU work occurred in preparation.

Each user-invoked helper permits one attended attempt of at most 20 minutes,
120 seconds per fresh worker, a 4 GiB allocator cap and five-second outer kill
grace. Identity and foreign-compute checks repeat every five seconds. Keep the
Mac awake and SSH/Terminal open. A failure preserves evidence and stops that
host; it does not authorize retries, tolerance changes or interference with
another process. The current two-host status supplied by the user is clear;
live checks repeat before work starts.

Component screening requires both algorithms to pass the predeclared output/
QKV derivative bounds against a CPU FP64 analytical oracle. The math arm also
passes a same-case efficient-reference comparison before timing. Within-arm
fresh and repeated outputs/derivatives must be bitwise equal. Retained device-
typed events must show the requested backend before timed samples are accepted.
Saved operand, oracle, runtime, geometry, timing and memory claims are checked
independently. Download review can use a different local checkout path only
after its original source bindings verify.

The upstream block enables its own SDPA choices internally. The new code therefore
captures its CPU synthetic operands at the attention boundary and forces each
standalone algorithm separately; the original full model is unchanged. No
component timing or screening tolerance qualifies full-model integration,
PGD/scoring/optimizer work, a faster training trajectory or an affordable study.

Preliminary local CPU fixtures and unlaunched packets remain preserved. Fixture
argument names and a singleton rotary subspace were corrected without editing
upstream. A NaN refusal test was aligned with the existing atomic writer, and
portable independent-review support was verified before final packaging.

Financial state is unchanged: $200 project / $160 compute / $40 non-compute,
one $10 diagnostic allowance inside compute, $9.8116 conservatively encumbered
and $0.1884 unreserved. The historical $2 hold and $0.0792 reconciliation gap
remain. All known Runpod Pods are terminated; no paid hold or research admission
is created. The user continues to make their own commits.

Next: run the two separate attended Mac commands once, collect both exports,
then independently review numerical, backend, replay, clock and memory evidence.
See the [diagnostic design](../../docs/attention_component_diagnostic.md).
