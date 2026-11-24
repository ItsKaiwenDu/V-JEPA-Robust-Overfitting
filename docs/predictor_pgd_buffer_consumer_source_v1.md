# Pinned NumPy and CPython buffer-consumer source review

**Both required release sources are available and reviewed; copy elimination
remains deferred.** This source evidence does not authenticate the target build,
selected hash backend/provider, private capture, writer ordering or performance.
Keep the original `.tobytes()` copies, every fresh hash and the closed budget gate.

See the [decision](../configs/predictor_pgd_buffer_consumer_source_v1.json) and
[saved review](../report/checks/2026-11-24-pgd-buffer-consumer-source-v1-local-review.md).
The earlier [symbolic harness](predictor_pgd_snapshot_symbolic_v1.md) and its
32 first-pass cases remain closed and are not rerun.

## Source provenance

The user expressly requested the previous nine-file commit and `git push`,
overriding the earlier user-only Git instruction for that operation. Commit
[`61011366b2c125bbf6b66426c291743936b07dc1`](https://github.com/ItsKaiwenDu/V-JEPA-Robust-Overfitting/commit/61011366b2c125bbf6b66426c291743936b07dc1)
uses the exact title and numbered description and was pushed to origin/main.
The working tree was clean and HEAD matched origin/main before this separate
official-public-source task began. This review introduces new uncommitted work;
automatic commit-message delivery remains the user's standing preference.

| Official release | Observed annotated tag | Resolved commit |
| --- | --- | --- |
| [NumPy v2.4.6](https://github.com/numpy/numpy/tree/v2.4.6) | `65daff092ee0f3d92f166630d32d6b9c81d99343` | `b832a09cf2a169c833dd2371e7c07aa00b293242` |
| [CPython v3.11.17](https://github.com/python/cpython/tree/v3.11.17) | `8e4c525dc9bcd1885bbf06f72c611124a7d26713` | `e6392eb68a39cef49c9fe6431253050caa17a426` |

Six saved API observations join refs to annotated tags and commits. Seventeen
complete source/license files,577297 bytes, are retained as inert `.txt`
snapshots. All17 calculated Git blob identities match the connector's reported
identities, alongside SHA256 and line counts. GitHub's reported signature status
is an API observation; no independent signature verification occurs.

The pinned NumPy project [declares2.4.6](https://github.com/numpy/numpy/blob/b832a09cf2a169c833dd2371e7c07aa00b293242/pyproject.toml#L8-L10).
Its Meson file delegates version spelling to a Python helper that reads this
declaration and contains Git/build logic. These files are inspected as text;
the helper is never run. The CPython
[version header declares3.11.17](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Include/patchlevel.h#L18-L26).
Release declarations and source identity do not establish the deployed binary,
ABI/build flags, actual OpenSSL/provider or runtime dispatch. No package binary
is downloaded, installed or imported, and no newer version is substituted.

## Buffer and bytes paths

Twenty source observations have exact file/range hashes and markers in
`SOURCE-CLAIMS-v1.json`. The following findings concern inspected branches only.

| Boundary | Finding |
| --- | --- |
| NumPy default order | The tobytes method defaults to C order and delegates to PyArray_ToString. [Method](https://github.com/numpy/numpy/blob/b832a09cf2a169c833dd2371e7c07aa00b293242/numpy/_core/src/multiarray/methods.c#L601-L611) |
| Contiguous bytes | The C-contiguous/C-order branch passes logical byte count and array data to the Python bytes constructor. Other layouts allocate/copy or delegate to ravel; their full downstream behavior is outside this review. [Conversion](https://github.com/numpy/numpy/blob/b832a09cf2a169c833dd2371e7c07aa00b293242/numpy/_core/src/multiarray/convert.c#L335-L414) |
| NumPy buffer export | A buffer points to existing array data, reports logical extent and access/layout information, and retains the array. Readonly status follows flags; acquisition creates no independent byte snapshot. [Export](https://github.com/numpy/numpy/blob/b832a09cf2a169c833dd2371e7c07aa00b293242/numpy/_core/src/multiarray/buffer.c#L776-L842) |
| Python bytes construction | The ordinary nonempty path copies input memory. Empty and single-byte paths can reuse immutable objects, so compatibility cannot require distinct result identity. Source capture atomicity is still unproved. [Constructor](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Objects/bytesobject.c#L119-L145) |
| Readonly wrappers | A readonly memoryview adds a view on the existing managed buffer and changes its readonly property. Its obj accessor exposes the retained exporter. Writable aliases still require separate exclusion. [Readonly](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Objects/memoryobject.c#L1425-L1438), [exporter](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Objects/memoryobject.c#L3023-L3034) |
| Lifetime guards | Memoryview exports retain/count views, and explicit release can refuse while exports remain. Generic buffer release invokes the exporter hook and drops the reference. These protect described lifetimes without proving immutability of all underlying aliases. [Memoryview](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Objects/memoryobject.c#L1035-L1058), [buffer release](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Objects/abstract.c#L755-L770) |

The existing flattened uint8 tensor-hash stream and ordered-name state-digest
stream remain separate. A buffer consumer may describe accepted layout differently
from the original chain; its acceptance alone does not establish identical
logical bytes, dtypes, refusals or offset handling. No original tensor case or
operator is executed here.

## Hash backend and consumption lifetime

The common [hashlib macro](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Modules/hashlib.h#L8-L58)
requests a simple buffer and rejects remaining multidimensional descriptions.
It supplies no private-copy guarantee. Its synchronization helpers concern the
hash object's mutable state, and define a2048-byte threshold used by selected
thread-release paths.

The reviewed OpenSSL
[constructor](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Modules/_hashopenssl.c#L856-L925)
can release the GIL while consuming sufficiently large initial input; its
[update](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Modules/_hashopenssl.c#L579-L607)
can also release it under the hash-object locking branch. The acquired buffer
stays held until the synchronous call returns, including the selected exit
paths. The helper forwards the input pointer to OpenSSL in chunks. The linked
OpenSSL/provider implementation and deployed build are not reviewed or joined.
Neither this hash lock nor exporter retention excludes unrelated input writers.

The reviewed builtin SHA256
[update](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Modules/sha256module.c#L500-L510)
and [constructor](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Modules/sha256module.c#L590-L623)
also acquire, consume and release buffers. These bodies contain no explicit
thread-release macro. The
[selection logic](https://github.com/python/cpython/blob/e6392eb68a39cef49c9fe6431253050caa17a426/Lib/hashlib.py#L126-L178)
can select OpenSSL or a builtin path according to availability/policy. Actual
target selection remains unknown; no behavior of one branch is generalized to
every backend, input size or build.

Inference: a readonly exported view can remain writable through another alias
during a hash. A future copy-elision proposal still needs proven private bytes,
ordered capture/transfer, contained exporters and retained lifetime through all
consumer uses and failures. The current bytes result protects its completed
contents without newly proving that source writers were excluded during capture.
No adapter/API choice or runtime bridge is selected by this review.

## Preservation and remaining evidence

The standalone saved audit checks71347 prior hashes,77867 prior fixture entries
and54 new entries, all17 source identities, two ref/tag/commit chains,20 claim
ranges and six local pins/four original AST anchors. It does not execute the
source binder, inspected code, project/tests/harness/scientific callbacks or any
network/process/clock/timer/GPU/proof port. Ordinary text/API/JSON/AST/filesystem
and hash work is real; public source retrieval is real and distinct from target
work. No model/tensor/gradient, packet/checkpoint/scientific-array decode or
scientific recalculation occurs. No old passing or closed cohort is repeated.

Raw stage: `runs/pretrained/diagnostic/pgd-buffer-consumer-source-v1-20261124`.
Historical FIFO remains LSTAT-only. Device16777229 references the frozen prior
observation; no new probe/remap/waiver. Old device/target gates, four exact
directory lifecycle exceptions and all prior actualv3 proof/31-of40 partial
failure/consumed claims remain. Source/test/auditor versions, fixtures/logs/
failures/results/statuses and mutable-index histories are preserved.

Next is a separate local classification of already-saved target provenance
metadata: exact declarations versus build/source/backend/provider origins and
missing joins. No target query, new observation, source execution, scientific
payload decoding or rerun follows. Metadata equality cannot authenticate actual
execution or promote old failed/partial results.

Original runtime graph, scientific/native/hash/ownership/durability/shutdown/
Guard sources, continuing clocks and caps remain fixed. All actual grants stay
consumed; no scope/package/grant/approval request/launch command is created.
Budget remains REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE; the v5 gate refuses before
protocol clock, ATTEMPT or connection. Actual units/fresh replay/timing/overhead/
memory/recovery/economics/calibration/data/protocol/training remain gated.
Mechanism closed; research/training false. Financial/training caps and holds
stay fixed; no training percentage or time estimate is supported.
