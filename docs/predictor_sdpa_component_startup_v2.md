# Component startup v2: local correction and frozen native inputs

The accepted native capture from the consumed November 7 v2 attempt is retained.
The next component execution must use a separate operator and resource grant.
This checkpoint fixes and tests startup locally; it contains no new university
launcher, authorization, attempt, connection or model capture.

## Single configuration owner

The old target wrapper configured deterministic execution, initialized CUDA while
setting its allocator fraction, then entered an executor which configured again.
The strict `Configure deterministic execution before CUDA initialization` guard
correctly rejected that second call. The old source and failed evidence stay frozen.

The new `models/predictor_sdpa_component_target_v2.py` handles plan, input,
predecessor and output checks. It leaves configuration and allocator setup to
`models/predictor_sdpa_component_v2.py`. At its original startup boundary that
executor calls `models/predictor_sdpa_component_startup_v2.py` exactly once, before
runtime/operand work. Startup requires the exact target runtime, invokes the
original strict configuration, checks device availability/capacity, applies the
unchanged 7 GiB allocator fraction and resets peak statistics. Its saved receipt
binds the process, device, precision flags and allocator extent.

A pre-existing CUDA context remains a refusal. The original runtime module and
configuration function are unchanged; neither is made idempotent or allowed to
configure after initialization. All arithmetic, saved operand layouts, canonical
FP64 builder, backend evidence, numerical bounds, serial predecessor checks,
fresh replay and deadlines remain as before. An AST regression checks the math
source; this correction cannot adopt the previously rejected full-model candidate.

## Local checks and their limits

The startup suite passes 71 tests. The regression enters both real wrapper and
executor, simulates CUDA context creation at the device-property call and stops
before operand work. The old chain reproduces two configuration calls and the
original refusal. The corrected chain configures once, applies the 7/8 fraction
on simulated 8 GiB hardware, and reaches the subsequent boundary. Pre-initialized
CUDA and a wrong target runtime refuse before allocator operations.

The suite also executes four fresh small CPU component workers against saved
native CPU inputs. One FP64 reference serves all four; actual return arrays,
original agreement, kernel-event associations and fresh replay verify under the
unchanged bounds. Rehashed startup receipts and unsafe/mutated records refuse.
These are local CPU results and simulated CUDA startup, not real target CUDA,
backend, memory-capacity or performance qualification.

The first two suite runs retained 69/2 and 70/1 passing/failing results. Their
failures were in the Mac CUDA shim/lazy import and overbroad test mocks. The
corrected test harness and earlier source snapshots are preserved. The final
count is 71, without accumulating retries or rerunning older scientific suites.

## Frozen native-input capsule

`scripts/prepare_native_sdpa_capsule_v1.py` makes a deterministic local archive
from reviewed, pinned inputs. The final artifact is at
`runs/pretrained/diagnostic/predictor-component-startup-v2-20261107/capsule-final/`.
It has 118 regular members, 48,627,833 logical bytes and 36,042,589 compressed
bytes; SHA256 `b8ad88e9a34ea55460177d996d40f915efe9d4284e16e9616923a46eeca6391c`.

The members are 97 unchanged code/metadata packet files, all 18 accepted native
capture files and three parent metadata records. The original training-pixel
file and redundant prior full-gradient packet file are excluded. The capsule
still contains saved native tensors, the captured full input gradient, loss and
targets. Every archive copy has mode 0600; original source bytes, modes and
nanosecond modification times remain unchanged. The initial variant, with a
0644 copied public report, is preserved but inactive.

All 18 packaging checks pass: actual member/source coverage, archive identity,
per-member content/size/mode/time, duplicate/unknown/unsafe paths, links, missing
members and refusal to overwrite an existing capsule. Saved-data review checks
all eight full native tensors again, maps an inert v2 plan to their local paths
and verifies 25,548 prior hashes and the original target metadata. It blocks
model/autograd/SDPA/CUDA/process/network and reference execution. No full native
FP64 reference or component is executed during that review.

## Next task

Integrate a separate component-only university operator/client using these saved
inputs, retaining strict live GPU checks and the existing scientific/resource
limits. Test cold authorization, exact payload/import paths, serial worker
startup and same-SSH collection before requesting one new exact scoped grant.
The consumed v2 grant and claim cannot be reused. Do not repeat its Mac helper,
connect for retrieval, change old clocks or delete old evidence.

Full training remains gated on full-model correctness/policy, chosen-hardware
optimizer/evaluation feasibility, budget and the research calibration/protocol
checks. The 3,724-entry candidate rejection and all financial holds/caps remain.
[Saved local review](../report/checks/2026-11-07-predictor-component-startup-v2-local-review.md).
