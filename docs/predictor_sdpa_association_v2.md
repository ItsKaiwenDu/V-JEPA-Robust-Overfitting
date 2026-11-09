# Offline SDPA event association v2

The separate standard-library verifier in
`models/predictor_sdpa_component_events_v2.py` assesses saved profiler records.
It does not execute a model, derivative, SDPA call, profiler, reference builder,
CUDA operation or university command. The original collector, v1 verifier,
scientific workers, failed records, consumed claims and clocks remain unchanged.
This version is not yet integrated into a new component worker or handoff.

## Association rule

Ordinary parent-contained traces retain their recorded CPU-parent phase evidence.
The new bridge applies only to supported first-order autograd nodes outside the
backward label's parent tree. It requires all of the following:

- Ordered forward/backward user labels from one calling thread and process.
- Exactly one supported forward SDPA invocation.
- A backward node with a matching forward-thread ID and nonnegative sequence
  number, plus exactly one matching recorded autograd engine wrapper.
- An intact engine/node parent relationship, matching process/thread/identity
  fields, and a host interval contained within the backward label.
- A known forward producer under the selected SDPA invocation. Dispatch wrappers
  sharing an identity must form one ancestor chain with a unique deepest producer;
  unrelated producers or duplicate backward identities refuse.
- An unchanged parent graph for the backward operator and its kernel-launch
  descendants. The verifier adds a separately recorded association region; it
  never changes an event's original parent, thread, name, time or sequence.

PyTorch documents that forward sequences are unique within a thread, so both
sequence and forward-thread identity are needed. Its backward `apply` ranges
carry the useful stashed sequence; low-level operations inside backward do not
supply that identity. The implementation therefore reads the autograd node's
metadata rather than treating the backward ATen operator's sequence as its key.
[PyTorch 2.8 profiler integration](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/profiler/README.md#autograd-integration),
[forward/backward correlation](https://docs.pytorch.org/docs/2.8/autograd.html#torch.autograd.profiler.emit_nvtx).
The recorded node scope follows the versioned
[RecordScope definition](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/record_function.h).

Fused efficient/CPU-flash nodes use their specific forward producer. Math VJPs
use their recorded inner BMM/MM or softmax producer identities, which need not
match the outer math wrapper's sequence. Unknown producers, missing metadata,
ambiguous nodes/engines/chains and out-of-scope backend operators refuse. The
supported backend families, device-domain restrictions and no-fallback policy
remain unchanged.

## Kernel evidence and limits

A required forward/backward operator still needs positive device-0 activity
linked from its recorded CPU descendants. Names alone do not qualify a phase.
The new verifier requires a unique matching CUDA row by name, device and duration
rounded to nanoseconds; this is an identity consistency check, not a speed metric.
Competing unrelated host claims, missing links, duplicate device witnesses,
annotation-only activity and mismatched names/devices/durations refuse.

GPU intervals can outlive their launching CPU labels. The verifier does not force
asynchronous CUDA execution into CPU-label time containment, select the nearest
kernel, infer a backend from flags or force a fallback.

The old trace schema contains no unique launch correlation IDs. Name/device/
duration matching cannot recover missing identifiers or authenticate arbitrary
fabricated traces. Assessments therefore bind the reviewed trace's hash and
preserve this limitation in their output. Ambiguous matches refuse rather than
inventing an identity. The proof concerns one recorded component profile; it
cannot infer worker completion, final allocator/flags/input receipts, fresh replay,
full-model correctness, capacity, throughput, cost or research admission.

## Validation and preserved outcome

72 tests pass. They cover the actual 106-row target trace, four saved CPU traces,
synthetic parent-contained and cross-thread efficient/math cases, math inner
producer identities, malformed parent graphs, identity ambiguity, unrelated or
unlinked kernels, device/domain errors and cold execution with scientific imports,
processes and network blocked. Synthetic cases are parser fixtures, not new CUDA
measurements. Cached CPU traces preserve the prior classification and phase proof.

The actual saved trace yields one cross-thread anchor and 12 linked CUDA witnesses,
including the required CUTLASS forward and backward kernels. The frozen v1 verifier
still reproduces its original refusal. The old component remains incomplete;
missing final receipts and unexecuted math/fresh-replay cells stay missing.
No result or event is relabeled and no numerical bound changes.

The first run's 67 passes/one failure and source snapshot are retained. That
fixture put a competing link outside the qualified operator subtree; it did not
make a valid proof ambiguous. The corrected fixture places the competing claim
inside the qualified subtree, with separate tests ensuring unrelated activity
cannot replace a missing required link. Additional malformed-parent and duration
overflow cases bring the final count to 72; counts are not accumulated.

Saved-only review independently checks the actual thread/sequence/parent links,
all 12 device witnesses, source hashes and 41,489 preserved file hashes. Only
metadata is read; no tensors or references are regenerated. Next integrate the
unchanged v2 verifier into a separate worker/handoff version and test complete
startup, numerical and receipt paths before any new scoped university grant.
No new target authorization or attempt is created at this checkpoint.
