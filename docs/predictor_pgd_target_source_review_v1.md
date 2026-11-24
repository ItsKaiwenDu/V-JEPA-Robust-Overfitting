# Official PyTorch 2.8 source review for snapshot ownership

**The versioned upstream source supports retaining the existing snapshot copy.**
Replacing `.tobytes()` with a read-only view of an arbitrary CPU tensor does not
establish a private, stable snapshot. Copy elimination remains deferred; no
adapter, API choice or runtime change is adopted. Every fresh hash and the
unconditionally closed budget gate remain fixed.

See the [review report](../report/checks/2026-11-24-pgd-target-source-review-v1-local-review.md)
and [decision config](../configs/predictor_pgd_target_source_review_v1.json).

## Provenance and scope

The user's November 24 “Continue” starts the separate official-public-source
evidence task identified in the completed local inventory handoff. The earlier
local-only inventory and its negative finding remain intact. This task retrieves
public source text through the GitHub connector and consults the official
repository through the web tool. That network work is real; no university
connection, query, source upload or target work is included.

The official [v2.8.0 tag observation](https://api.github.com/repos/pytorch/pytorch/git/ref/tags/v2.8.0)
resolves to commit
[`ba56102387ef21a3b04b357e5b183d48f0afefc7`](https://github.com/pytorch/pytorch/commit/ba56102387ef21a3b04b357e5b183d48f0afefc7).
Thirteen complete UTF-8 source/license files are retained as inert `.txt`
snapshots. Their locally calculated Git blob identities match the connector's
reported identities; SHA256, byte/line counts and capture stat observations are
also recorded. The license is retained. GitHub's reported commit-signature
verification is an API observation, not an independent signature check here.

The tag's [version.txt](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/version.txt)
contains `2.8.0a0`. A release tag and its source-version template do not
authenticate the deployed `2.8.0+cu128` wheel, generated version file, CUDA build,
dispatch, NumPy2.4.6 or CPython3.11.17 consumers. Those build/runtime joins remain
missing. Newer local2.14.1 documentation is not substituted.

## Findings anchored in the captured source

Each observation has a frozen file hash, exact line range, range hash and source
markers in `SOURCE-CLAIMS-v1.json`. These are source observations; physical
ownership, concurrency and deployed dispatch are not tested.

| Boundary | Versioned observation and consequence |
| --- | --- |
| Detach | The documented result retains storage sharing. Detaching does not establish private bytes. [Source](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/_tensor.py#L791-L808) |
| CPU transfer | The ordinary Python wrapper passes `non_blocking=false` and `copy=false`; dispatch releases the GIL. The wrapper supplies no exclusive-writer guarantee. [Source](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/tools/autograd/templates/python_variable_methods.cpp#L406-L449) |
| Matching device/options | The native conversion fast path can return its input. An already-CPU tensor is not necessarily copied. Explicit copy bypasses that reuse path; inspected dense branches allocate and copy. [Source](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/TensorConversions.cpp#L344-L447) |
| Contiguous layout | An already contiguous tensor is returned directly; the other branch clones. Contiguity is not ownership. [Source](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/TensorProperties.cpp#L115-L125) |
| Flattening and byte view | Reshape can alias, while dtype view reuses storage and adjusts metadata under size/stride/offset rules. Preserve reinterpretation and logical byte order. [Reshape](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/TensorShape.cpp#L1997-L2048), [dtype view](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/TensorConversions.cpp#L872-L959) |
| NumPy export | The array uses the prepared tensor's data pointer and is writable. A tensor base and disabled resizing attempt retention; the source itself identifies a storage-replacement lifetime limitation. [Source](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/csrc/utils/tensor_numpy.cpp#L123-L201) |
| CUDA completion | The inspected device-to-host branch distinguishes asynchronous transfer from a helper that synchronizes the current CUDA stream. This supports a source-level completion argument only for its branch, not ordering of every possible writer. [Copy branch](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cuda/Copy.cu#L384-L424), [helper](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/c10/cuda/CUDAFunctions.h#L78-L99) |
| Copy/export requests | Ordinary clone allocates/copies, and `to` documents an explicit copy option. NumPy's force option can still share storage. These API descriptions do not select an implementation or prove a caller's lifetime discipline. [Clone](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/TensorFactories.cpp#L2155-L2178), [versioned docs](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/_tensor_docs.py#L3708-L3734) |

Inference from these sources: layout conversion, graph detachment and a read-only
export flag cannot collectively substitute for a private-snapshot contract.
A proposed copy would also need ordered capture, no surviving writable access,
retained exporters/views through hash completion and exception cleanup, and the
exact original byte/refusal contract. The reviewed files do not prove NumPy or
CPython buffer-consumer behavior, the actual target build, absence of other
writers, or complete-unit cost and memory fit.

## Unchanged project contract

Six prior source pins and four AST anchors cover `tensor_hash`, `Snapshot.check`,
the separate ordered-name `state_digest`, and the unconditional budget refusal.
The tensor hash still detaches, transfers to CPU, makes logical contiguous
layout, flattens, reinterprets as uint8, exports to NumPy, materializes bytes and
computes SHA256. Every original observation and guard remains required.
`state_digest` retains its own ordered-name and dtype byte contract.

No Torch/NumPy import, binary package download/install, source execution, tensor
decode, model/gradient/worker lifecycle, FileSink/ledger, benchmark, workload
clock/timer/signal/PIDFD, GPU/proof or paid work occurs. No previous witness,
encoding comparison or scientific cohort is rerun. Ordinary source reads,
UTF-8/JSON/AST parsing, filesystem copies and hashing are real local work.

All actual grants stay consumed. No operational scope, package, grant, approval
request or launch command is created. The original runtime/client graph,
scientific calculations, hashing/ownership/durability/shutdown controls and
continuing clocks/resource caps remain fixed. Research/training are false;
source availability does not admit execution or establish a training estimate.

## Saved review and next task

One initial source-path lookup returned404; its response and the following
capture TypeError are retained. The corrected `tools/autograd/templates` path
was fetched separately. The first provenance reader completed all13 source
identity joins before its local AST lookup refused a wrong helper path. A v2
local-contract reader corrected only that lookup, inheriting the saved source
manifest without fetching/reemitting it. All versions and logs remain.

The independent saved audit checks prior preservation, source/claim/AST joins,
45 new fixture entries and the decision. It executes no inspected source,
capture/claim builder, project/test/scientific callback, or network operation.
The historical FIFO remains LSTAT-only. Device16777229 references the frozen
shutdown-binding observation; old device/target gates and four narrow directory
lifecycle exceptions remain unchanged. Any further device change requires a
new record/review. Prior actual v3 proof/31-of40 partial failure and consumed
claims remain unchanged.

Raw evidence: `runs/pretrained/diagnostic/pgd-target-source-review-v1-20261124`.
The prior inventory remains uncommitted and preserved at starting HEAD828e7fa.
Only the user stages, commits and pushes. Mutable indexes retain exact
before/after snapshots and remain outside the preservation seal.

Next is a separate local ownership/lifetime/byte-contract design and explicit
fake-port validation plan at the existing hash call sites. This is requirements
work only; it selects no adapter/API, emits no runtime, imports no libraries and
repeats no old witnesses/cohorts. Target build provenance and actual complete-unit
timing/memory/replay evidence remain required before any operational admission.
Budget stays `REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE`; the v5 gate refuses before
protocol clock, ATTEMPT or connection. No target query or launch follows.
