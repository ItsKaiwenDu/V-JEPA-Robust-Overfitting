# Separate profiler overhead and GPU-range metadata policy

This is an offline metadata correction in
`models/predictor_sdpa_launch_evidence_v2.py`, with a saved-data CLI in
`scripts/check_predictor_sdpa_launch_evidence_v2.py`. It starts no model, autograd,
SDPA, profiler, CUDA, process, reference builder or university connection. The
old validators, component workers, handoffs, consumed claims and historical
refusals remain unchanged. No new component worker or target handoff is included.

## Raw preservation and roles

The input is the unchanged `fixed-SDPA-raw-Kineto-correlation-v1` envelope. Every
raw row, original index, correlation value, integer timestamp and device value
stays intact. The separate projection has the same number of rows and adds a
record role; no filtered or renumbered raw trace is constructed. Existing bounds
stay at 20,000 records and 8 MiB, with strict schema/types/clocks/duplicate checks.
Only the narrow classifications below differ from the frozen projection.

| Role | Participation in ownership and backend evidence |
| --- | --- |
| Frontend CPU | Unique positive correlation ID in one process; eligible to own runtime/device links. |
| Linked runtime | Requires its explicit recorded frontend owner and the existing stable-thread/containment rules. |
| CUDA activity | Requires device 0, no annotation flag and a recorded frontend link; only positive genuine activity can supply kernel evidence. |
| Profiler overhead signature | Retained with its negative device index; cannot own links, parent CPU operations, define phases or provide a kernel. |
| Profiler GPU range | Retained and paired to its actual CPU phase annotation by recorded correlation ID; cannot provide kernel evidence. |

The overhead exception is restricted to the exact runtime `2.8.0+cu128`, CPU
`Activity Buffer Request`, device index -1, resource 0, start/end thread 1,
forward thread 0, sequence -1, scope 0, zero linked correlation, positive correlation,
false async/annotation flags and legacy elapsed -1. All other basic field checks
still apply. This deliberately recognizes one observed, source-supported signature;
it does not claim every profiler overhead variant is supported. Unknown negative
IDs, other names, malformed signatures and unsupported runtime strings refuse.
A positive-process row with the same name remains an ordinary CPU row.

The negative-device overhead may carry the same correlation number as a real CPU
operation. In the saved trace, rows 9 and 10 both carry 10. The ownership map uses
only role-qualified frontend CPU rows, so actual kernel/runtime links still resolve
to row 9. Overhead is never an alternate owner. An overhead-only number cannot
satisfy a link. There is no global uniqueness assumption across different roles.

The only supported GPU user ranges are our forward/backward phase labels under
the same exact runtime, with the recorded annotation flag, device 0, zero linked
correlation, positive correlation, scope 0, sequence -1, forward thread 0, thread 1,
false async and legacy elapsed -1. The range's correlation must identify a unique
real CPU annotation of the same phase on a stable thread. Duplicate phase/stream
ranges, foreign devices, unknown phase labels, invalid owners or unsupported
runtime refuse. Ranges stay outside kernel-link lists. Removing a real required
kernel or relabeling it as an annotation still fails the backend checks.

## Versioned upstream basis and its limits

Official PyTorch v2.8.0 pins Kineto
`5e7501833f1021ce6f618572d3baf657b6319658`. Its overhead wrapper reports device -1
and maps the observed buffer-request name for CUDA 12.7 and later. PyTorch can
inherit a parent's correlation ID for a Kineto record that has none, explaining
why that number need not be a separate frontend-owner identity.
[Overhead representation](https://github.com/pytorch/kineto/blob/5e7501833f1021ce6f618572d3baf657b6319658/libkineto/src/CuptiActivity.h#L157),
[name mapping](https://github.com/pytorch/kineto/blob/5e7501833f1021ce6f618572d3baf657b6319658/libkineto/src/cupti_strings.cpp#L67),
[correlation inheritance](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/profiler/collection.cpp#L591).

Kineto also creates GPU annotation ranges with the CPU annotation's name and
correlation ID, plus the device and stream of the spanned GPU work. The raw Python
annotation flag recognizes CPU and GPU user annotation activity types. The range
is distinct from the kernels it spans.
[GPU range creation](https://github.com/pytorch/kineto/blob/5e7501833f1021ce6f618572d3baf657b6319658/libkineto/src/CuptiActivityProfiler.cpp#L495),
[raw-event Python bindings](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/init.cpp#L227).

These are source-supported metadata interpretations, not authentication of
arbitrary fabricated metadata, a direct native activity-kind witness for overhead,
or proof of the installed wheel's source commit. Saved byte hashes/provenance and
future live runtime guards remain necessary. No new native APIs, timing-based
ownership inference, fallback, generated IDs or name-only exclusion is introduced.

## Strict backend decision and saved-only result

The complete backend/phase/autograd validation function is an exact AST copy of
the frozen verifier, apart from its private function name. The original
thread/sequence/producer rule is reused unchanged. Efficient/math/CPU-flash families,
forward/backward requirements, actual kernel families, positive duration, device,
no-fallback and mixed-backend refusals stay. The wrapper reports only metadata
validity and explicitly leaves CUDA execution, worker completion, replay,
performance, research and full-model adoption unqualified.

All 106 saved university rows remain present. Under this separate policy, row 10
is retained as overhead and row 20 as a GPU range. Metadata assessment finds the
efficient backend and 12 distinct positive raw CUDA witnesses tied to real CPU
owners, with the original cross-thread autograd anchor. This is a retrospective
metadata-only assessment. The historical worker's error, incomplete status, false
later flags, missing complete/resource receipt and unrun components 1/2/3 stay.
The old validator still rejects the exact same raw trace.

78 focused checks pass with zero errors/failures/skips. They cover all prior
backend families on synthetic metadata, overhead/annotation spoofing, malformed
fields/runtime, ownership/correlation/ancestry/autograd failures, missing kernels,
full raw retention, unchanged validator AST and cold saved-only CLI assessment.
No test starts a model, derivative, SDPA, profiler, CUDA or reference builder.
The initial local assessment's overstrict global-ID refusal is preserved with its
source snapshot. The first test run's 61 passes/3 failures are also preserved:
one synthetic fixture accidentally shadowed an autograd producer, and two saved
trace checks exposed the GPU phase annotation. The fixture was corrected and
narrow GPU-range handling/regressions added. Counts are not accumulated; no
scientific bound or frozen source changed. Old suites were not rerun.

Next integrate this policy into a separately versioned component worker, retaining
publication-before-validation and original arithmetic, canonical FP64 reference,
startup, layout/input/flags, resource and fresh replay gates. Complete local
integration tests before preparing any new university handoff. This stage creates
no target scope, authorization, attempt, cleanup, installation or paid work.
