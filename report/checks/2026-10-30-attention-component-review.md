# Attention component review — October 30, 2026

**Mixed result.** Both exports are downloaded and independently byte-verified.
The predictor has a promising saved-input component result. The encoder run
stopped on an input-identity mismatch and did not start its second shape.
Input generation and the original cross-platform reconstruction check remain
unqualified. No backend is adopted and no retry starts from this review.

The predictor export contains 122 regular members (214,268,862 bytes, SHA256
`1cede5502127193fa06816b5c6f0f42c2f4c407ed636e9d9cffed7a70e625be3`).
The encoder export contains 120 regular members (137,315,607 bytes, SHA256
`d27d74f1d83bd9bab8255ce54537de889b6451d2441e8a62e065b519a8200366`).
All 242 members match both their archive manifests and downloaded files. Each
export includes the original 103 source/metadata members. All 61 scientific
pins, 33 upstream source bindings and captured diagnostic sources are unchanged.

The attended commands began at 11:27:43 and 11:27:44 AM PDT on linux10609 and
linux10603. Reported session times before archive export were 42.14s and 20.51s;
the exits were zero and one respectively. Every worker stayed below its
120-second limit. Twenty-two monitor observations contain no foreign compute
process; sampling does not prove continuous GPU exclusivity. Both launch claims
remain consumed and unchanged. Completed SSH sessions may close.

## Predictor result on identical saved operands

The 2048-token, 12-head, 32-dimension component completed four fresh workers.
Both algorithms satisfy the predeclared elementwise output/Q/K/V derivative
bounds. Actual CUDA events identify FP32 CUTLASS efficient forward/backward in
the reference and SGEMM plus softmax forward/backward in the math arm. Saved
operands match exactly across algorithms; all 12 saved tensors replay exactly
within each algorithm's two fresh workers. The worker also records successful
within-process signature checks before, during and after profiling/timing.

| Median over six timed passes per arm | Efficient | Math | Efficient time / math time |
| --- | ---: | ---: | ---: |
| Forward | 2.218 ms | 4.728 ms | 0.469 |
| Backward | 23.318 ms | 7.364 ms | 3.166 |
| Forward + backward | 25.537 ms | 12.123 ms | 2.106 |
| Peak allocated memory | 75.2 MiB | 899.0 MiB | — |

Math is slower in forward but faster over the complete isolated component,
with substantially higher peak allocation. These are synthetic attention
operands, two fresh samples and six short timing observations per arm, all on
one RTX 3070. They do not measure a trained predictor, full attention block,
EOT8 objective, attack, checkpoint recomputation, optimizer or training run.
There is no confidence interval or full-study cost replacement.

## Encoder refusal

The 656-token encoder shape completed efficient replay 0, math replay 0 and
efficient replay 1. The fourth worker, math replay 1, passed its *reported*
untimed FP64 oracle check but refused its reference comparison before profiling
or timing. Its saved common contract differs from the efficient replay 1
contract at exactly one field: `/operand_signatures/q/sha256`.

The expected Q hash is
`b7139f2ab8c27b104500b3af439af233dae36e14e27664f1572b1dc6a80bf82a`;
the failed worker records
`fea51af6138a80bb5c6d2c83523cad0b0661657b52d4348f4a1d3aa017f1e27f`.
The reference worker and tensor-file checksums match their authorizations.
Other input signatures, shape, seed, hardware and runtime fields match.
This is an input-identity refusal in the saved record, rather than an SSH
disconnect or a recorded foreign-process refusal. Its cause is not established.

The failed worker saves no operand/output tensor artifact, so the magnitude
and element locations of that Q difference cannot be recovered. Its oracle
report cannot be independently recomputed. The 400-token shape never starts.
Three earlier workers are inspectable evidence, but there is no complete,
fresh-replayed encoder comparison and no accepted encoder speed estimate.

## Independent review and its limit

The unchanged original reviewer also refuses the predictor export on the Mac:
regenerating FP32 synthetic inputs does not yield the saved Linux signatures.
All four operand tensors differ, with maximum differences between 1.31e-6 and
2.18e-6; shapes, positions and other shape metadata match. The review machine
uses ARM64/PyTorch 2.14.1, while the run used x86_64/PyTorch 2.8.0+cu128.
These environment differences do not establish the cause of either this
regeneration difference or the encoder's within-host Q mismatch. The original
reconstruction refusal remains a failure and is preserved without changed
inputs, thresholds, sources or captured summaries.

A separately identified `attention-saved-operands-numpy-audit-v1` checks the
values that actually reached each completed worker. Its independent CPU FP64
softmax and analytical Q/K/V VJPs reconstruct the saved oracle within the
existing FP64 bookkeeping bounds (rtol 1e-12, atol 1e-14), then verify actual
CUDA tensors at the original component bounds. This establishes a saved-input
numerical result; it does not waive or certify input-generation replay.

Across seven completed workers, the audit verifies 84 tensor signatures,
28 actual-versus-NumPy comparisons and 12 same-input cross-backend comparisons.
Forty-eight fresh/cross-arm tensor byte comparisons pass. Fourteen CPU reader
tests pass with zero skips, covering independent derivatives, finite differences,
elementwise bound refusal and both observed CUDA softmax-backward variants.
The 54 original preparation tests remain a separate earlier check.

The [machine-readable review](2026-10-30-attention-component-review.json) binds
the raw audit, original-reviewer refusals, source snapshots and test receipt.
The old packets, worker outputs, summaries and launch claims remain unchanged.
No real SSH connection, upload, GPU work or paid session started during review.
Financial holds and the $200/$160/single-$10 caps remain unchanged.

Next prepare an immutable, checksummed synthetic-input fixture that all workers
load, and save input evidence before any refusal. Keep generator verification
separate from same-input backend comparison. A new encoder attempt requires a
new reviewed bounded plan; the old launchers must never be repeated. Full-model
correctness, memory, speed, optimizer recovery, study affordability and research
admission remain pending.
