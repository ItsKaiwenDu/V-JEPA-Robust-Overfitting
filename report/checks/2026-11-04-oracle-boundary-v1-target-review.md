# Original CPU return captured after CUDA setup; independent review passes

The controlled linux10603 capture succeeds. The unchanged original worker
initializes CUDA, computes one CPU FP64 oracle return, and passes all four
original bookkeeping checks at `rtol=1e-12, atol=1e-14`. The observer saves the
actual output/dQ/dK/dV before that gate and stops at the first input-transfer
call before its body. The original full GPU worker remains `complete=false,
passed=false` with the expected `BoundaryReached` stop. GPU input transfer,
attention, benchmarking and research admission are not performed.

| Check | Independently verified result |
| --- | --- |
| Saved response | 200,382,839 bytes; 166 members / 200,221,415 logical bytes |
| Original source packet / overlays | 139 unchanged packet members / nine unchanged overlay files |
| Prior evidence | 946 preserved bindings; 1,120 total sealed input files |
| CPU numerical gate | Four checks pass; maximum absolute error 2.498001805406602e-16 |
| Prior CPU-only values | All 96 matched array comparisons bitwise equal |
| Original inputs/common contract | Identical to both historical encoder-1 passing and failing records |
| Storage proof | Full 2.5 GiB write/fsync/readback passes in 9.462 seconds; 1,926 events |
| Observed setup / capture peak | 2,701,959,168 / 217,427,968 bytes, below 3 GiB / 2.25 GiB ceilings |
| Work / supervision | Whole phase 49.005 seconds; original observer call 0.752 seconds; worker supervisor 4.502 seconds |

Independent review denies process/network creation. It rehashes the complete
response and every saved file, verifies exact modification times, authenticates
the original packet and overlay sources, reproduces the original gate decisions
from actual arrays, and compares with every saved final array from the accepted
CPU-only target and control (four workers × three phases × four arrays × two
hosts). No worker or oracle is rerun. Metadata and record checks preserve all
original incomplete/error fields, strict tolerances, prior claims and refusals.

The computed-return NPZ is 13,108,174 bytes with SHA
`c7da1f108881a005432443180ffe10f814bf67aa541abae109add84276a8a7f8`.
The input snapshot has the same bytes as both historical snapshots,
`1c4dc71ab5e5de65cd6a47747dedc24261aa95db801cb1b8c67aaef74e51dc6e`.
This time the computed arrays themselves are present; fixed canonical references
are not being relabelled as missing historical computed results.

Post-gate context records CUDA initialized, PyTorch 2.8.0+cu128, Python 3.11.17,
NumPy 2.4.6, four intra-op/eight inter-op threads and matching recorded affinity.
Compared with the CPU-only captures, Torch build introspection includes five
additional CUDA/NVCC/cuDNN/Magma detail lines. Existing build settings, parallel
text and NumPy configuration match. These display differences do not establish
that libraries changed or explain the old numerical refusal. The new metadata
is collected after the gate; older CPU probes collected it before their first
oracle call. This remains an observed capture rather than a single-variable
causal experiment.

The historical approximately 1e-10 refusal is **not reproduced or explained**.
Its actual computed arrays remain unavailable. This successful observed setup
does not justify waiving old failures, changing bounds, qualifying GPU attention
or adopting an optimization. GPU monitoring persisted one periodic row during
the short worker; it does not prove continuous exclusivity. Storage accounting
is not a kernel aggregate quota or future reservation.

The new attempt claim is consumed:
`5e06811a7c9aeb781addd3e6f3f90baa85641fe7502619f1f5de1f3e8465cda3`.
The original control/authorization/source bytes remain frozen. Never repeat the
launch or reset its claim. Remote scratch
`/var/tmp/vjepa-oracle-boundary-v1-20261104-linux10603-js7zusqn` is retained;
no deletion or further remote work is authorized. Both university hosts can
stay idle. Financial caps/holds and pending reconciliation remain unchanged.

Next prepare an **offline proposal** for the next original GPU-attention
boundary, retaining the original CPU gate and saving downstream actual values
before their acceptance checks. This review prepares no next operator, resource
grant or launch. The 79-test preparation remains the unchanged operator baseline.

Raw independent review:
`5d63984c5ff94f94b536236f255337dbb09ce2434ac12a8c7935d2102154a2f9`.
The [machine-readable review](2026-11-04-oracle-boundary-v1-target-review.json)
contains source, transport, evidence, limits, comparisons and limitations.
