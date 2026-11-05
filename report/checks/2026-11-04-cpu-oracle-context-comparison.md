# Historical CPU oracle discrepancy: offline setup comparison

The saved evidence narrows the problem to **CPU reference bookkeeping before
GPU input transfer and before attention backend selection**. The historical
passing and failing encoder-1 workers have identical input snapshots and recorded
common/runtime contracts. Their reported results differ, but neither saves the
actual `local_oracle` return values. Later CPU-only captures pass on both hosts.
The cause of the historical discrepancy remains unresolved.

This is an offline source and evidence analysis. It starts no oracle, worker,
CUDA context, SSH connection or paid resource, and changes no original source,
numerical bound, failed record, claim, financial hold or accepted result.

## Verified evidence and original gate replay

The accepted target and control reviews are rehashed and their saved sources /
arrays / timestamps reverify. All 317 historical archive members (151 on
linux10603 and 166 on linux10609), their sidecars and unpacked files verify.
All 131 immutable control history bindings remain unchanged. The detailed raw
audit seals and rehashes 935 input files after completing its checks.

The two historical encoder-1 input snapshots have the same SHA256:
`1c4dc71ab5e5de65cd6a47747dedc24261aa95db801cb1b8c67aaef74e51dc6e`.
Eight actual tensor comparisons confirm their logical FP32 values match the
fixed fixture. Sixteen additional file comparisons confirm the four input and
four reference NPY files on both current hosts match that historical fixture.
The original 61-source study binding, fixture loader, execution helpers and
`fp64_oracle` implementation remain unchanged. The oracle explicitly converts
its operands to CPU FP64 and performs the same query-chunked calculation.

| Check | Outcome |
| --- | --- |
| Historical passing/failing common contracts | Identical |
| Historical four-input snapshots | Same serialized bytes and logical values |
| Fixed NPY inputs and references across old/current captures | 16 file identities match |
| Original Torch gate expression on saved current final values | All 96 comparisons pass |
| Original gate bounds | `rtol=1e-12`, `atol=1e-14`, unchanged |
| Maximum error in that gate replay | 2.498001805406602e-16 |
| Current matched cross-host values | Previously verified 2,112 bitwise-equal array pairs |

The 96 comparisons cover both hosts, four workers, three captured calls and
four output/VJP tensors. They use the literal original `torch.allclose` gate
and maximum-absolute-error expression, reading already-saved computed values;
they do not rerun `fp64_oracle`. The reviewer is Python 3.11.16 / Torch 2.14.1 /
NumPy 2.4.6 on arm64, rather than the historical Linux Torch 2.8.0 runtime.
This confirms that current saved values pass both auditing expressions under
the same bounds. It cannot certify the missing historical values or the old
runtime's computation, and it does not retrospectively accept that failure.

## Where the original worker stops

Inspection of the captured, hash-verified source establishes this order:

| Historical worker step | Source line |
| --- | ---: |
| Shared policy and four Torch threads requested | 77 (separate column positions verified) |
| Hardware gate and CUDA availability check | 82–83 |
| CUDA device properties and allocator fraction setup requested | 85 |
| Load the fixed fixture | 89 |
| Capture the simplified runtime contract | 94 |
| Compute `local_oracle = fp64_oracle(...)` | 104 |
| Reject failed CPU bookkeeping | 107 |
| Transfer input leaves to the GPU | 108 |
| First `iteration(...)` call | 115 |

The `iteration` function enters `force_backend` at line 127 of
`models/attention_component.py`, then calls SDPA at line 129. Backend selection
is downstream of the CPU gate. The failed worker has no device-input snapshot,
GPU math-attention outputs, backend profile or timing. Its `math` label is a
planned arm, not evidence of an executed GPU math backend. The parent requests
fresh subprocesses for each arm; the efficient and math workers are not two
successive computations in one reused Python process.

These are source-order facts consistent with the saved stop stage, not a
dynamic historical call trace. The raw audit records both line and column
positions because setup statements share a source line. A preliminary local
source-order assertion used only line numbers; it is preserved separately and
corrected without changing any numerical or historical evidence.

## Differences and missing observations

| Aspect | Historical fixed-input worker | Passing CPU-only probe |
| --- | --- | --- |
| Fixed input/reference values and oracle math | Same frozen values and implementation | Same |
| CUDA setup before CPU gate | Requested in **both** historical pass and failure | Hidden CUDA; uninitialized state recorded |
| Torch thread count | Four recorded | Four and one tested; only four-thread cases are direct matches |
| Precision/runtime fields present in both records | Identical between historical pass/failure | Match for the four-thread workers |
| CPU interop count, BLAS build, affinity and rounding details | Not recorded at the failed boundary | Additional context available |
| OMP / MKL / OpenBLAS thread variables | Launcher source requests four; effective worker values not saved | Requested and recorded per worker |
| Pre-oracle metadata collection | Simplified `runtime_contract` | Detailed Torch / NumPy build and thread introspection through `context()` |
| Other prelude work | Hardware polling, CUDA setup, CPU input `.pt` snapshot and its I/O | Different imports/metadata/I/O path and fresh scratch environment |
| Oracle calls per worker | One | Unobserved before, observed with line tracing, unobserved after |
| Actual computed CPU return values | Missing, including in the passed worker's saved `values.pt` | Saved for all three calls |

The historical `values.pt` fields called `oracle_output`, `oracle_dq`,
`oracle_dk` and `oracle_dv` are copied from the **fixed NumPy reference**, not
from the independently computed `local_oracle`. Their bytes match the canonical
references. The source assigns that artifact from `oracle`, while the separate
`local_oracle` is used only to calculate bookkeeping metrics. Those artifact
fields therefore cannot recover the earlier CPU calculation or identify its
first erroneous element/operation.

Both historical paths performed the same declared CUDA setup before this gate:
one passed and one failed. CUDA setup alone is not an established explanation;
an interaction or hidden state remains untested. The newer CPU probe also calls
`torch.__config__.show`, `torch.__config__.parallel_info` and NumPy build
introspection **before its first oracle call**. The older simple runtime record
does not. Observation/library-introspection effects are untested. The two
programs must not be described as an experiment that changes only CUDA
initialization. The original 656-token encoder-0 success on linux10609 is also
not a matched 400-token encoder-1 control; the current fixed captures supply
that later matched comparison, without repairing the historical missing values.

The reported failed maxima remain approximately 1.2846e-10 (output),
8.9826e-11 (dQ), 3.7739e-11 (dK) and 9.2847e-11 (dV), against errors at most
2.50e-16 for the preceding passed worker on the same inputs. These are retained
failed receipts, not independently reconstructed failed tensors. No claim of a
harmless rounding difference, faulty operation, defective host, CUDA cause or
library defect follows from these data.

## Next: prepare a capture at the missing boundary

Prepare a **separate, tested observer for the original pre-SDPA call path**.
Preserve the original worker, fixed inputs, setup ordering and bookkeeping
gate; capture the actual `local_oracle` output and dQ/dK/dV immediately when
the original call returns, before the original acceptance check can discard
them. Record the original gate outcome and full runtime context. Avoid adding
new detailed library introspection before the first computed result; collect
it afterward or treat it as an explicit observation control. Preserve any
refusal and stop before GPU input transfer / SDPA.

This would fill the specific missing evidence rather than repeat more pure
CPU-only runs. The observer can affect execution and must be checked on local
fixtures, against the restored package and for source/receipt integrity before
any university proposal. CUDA setup is outside the prior CPU-only domain; no
old grant authorizes this follow-up. Its operator, exact resource bounds and
new authorization must be reviewed separately. **No executable handoff, new
authorization or attempt is prepared by this design note.**

Both university hosts can remain idle. Both current grants are consumed. No
remote cleanup, paid work, financial release, GPU/backend qualification,
optimization adoption or research admission occurs. Project $200 / compute
$160 / other reserve $40 / single diagnostic $10 caps and all historical
financial holds and reconciliation gaps remain unchanged. User commits themselves.

See the [machine-readable comparison](2026-11-04-cpu-oracle-context-comparison.json),
[historical refusal](2026-10-30-fixed-attention-review.md),
[accepted target](2026-11-04-cpu-oracle-capture-v3-target-review.md),
[accepted control and cross-host comparison](2026-11-04-cpu-oracle-control-v1-review.md),
and [next-step design](../../docs/cpu_oracle_context_comparison.md).
