# CPU oracle setup comparison and next capture boundary

The [offline comparison](../report/checks/2026-11-04-cpu-oracle-context-comparison.md)
confirms that the historical refusal occurred in CPU bookkeeping before input
transfer or attention backend selection. The historical pass and failure have
identical saved inputs and common contracts. All 96 newer saved final values
also pass the original Torch comparison expression at unchanged bounds.

The cause is still unknown. Neither historical worker saved its actual computed
`local_oracle` arrays; `values.pt` contains the fixed canonical references.
CUDA setup was present in both the old pass and failure. Newer CPU-only probes
also differ in pre-oracle metadata collection, imports, I/O and observation,
so their passing results do not isolate CUDA initialization as a causal variable.
The detailed review preserves source ordering, runtime gaps and actual evidence.

## Design only: observe the original CPU return before the original gate

The next preparation should build a separately named observer around the
unchanged original fixed-input worker path. Its purpose is to save actual CPU
outputs and Q/K/V derivatives immediately after `fp64_oracle` returns, together
with the original gate decision and relevant execution context. It should stop
before GPU input transfer or SDPA and preserve all failed results and old claims.

Preserve the cold setup sequence as closely as possible. Detailed BLAS/thread
introspection should follow the first captured result, or become an explicitly
separate observation control; calling it first would add another prelude
difference. Record instrumentation entry/exit, trace/profile restoration and
source identity. The observer must neither replace the oracle nor alter input
values, tolerances, thread requests, numerical acceptance or bookkeeping flags.
Passing this capture would not constitute a GPU attention benchmark.

Preparation needs meaningful local checks: original function identity,
unchanged input/return values, capture before a rejection can lose data,
preservation of original pass/refusal status, restoration on exceptions,
failure collection, no dispatch beyond the intended boundary, cold restored
package imports and explicit separation of test fixtures from live grants.
All actual resource bounds must be determined and reviewed before proposing
a new university attempt. The prior CPU-only grants are consumed and cannot
authorize CUDA setup or a new run. No operator or launch helper exists yet.

This note grants no resources, launches no work and modifies no captured source.
Both hosts can stay idle while the separate observer is prepared. The historical
discrepancy and GPU/research qualification gates remain unresolved.
