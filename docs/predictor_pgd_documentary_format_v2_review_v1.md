# Documentary format v2: bounded source differential review

V2's direct lists, embedded positioning, explicit role/context rules and seven
joint node constraints are traceable declarative changes. They do not qualify
the proposal for adoption or implementation. The next useful work is a data-only
contract for four open context-supply and error-boundary requirements.

[Decision config](../configs/predictor_pgd_documentary_format_v2_review_v1.json),
[saved review](../report/checks/2026-11-27-pgd-documentary-format-v2-review-v1-local-review.json)
and [raw records](../runs/pretrained/diagnostic/pgd-documentary-format-v2-review-v1-20261127/DELTA-FINDINGS-v1.json)
retain the exact review relations. The preserved
[v2 proposal](predictor_pgd_documentary_format_proposal_v2.md) and
[v1 source findings](predictor_pgd_documentary_format_source_review_v1.md)
remain unchanged.

## What the v2 changes resolve as declarations

All four families are embedded values at their original REQUIREMENTS-ONLY
profile fields. F02 uses a direct exact list, which agrees with the current
descriptor's container category. Its six v1 wrapper fields are absent from the
v2 value declaration; no unwrapping, adapter, new outer keys or standalone
acceptance is supplied. All seven list roles remain unsupported by the current
validator. F01 still has the exact-dictionary category check; neither category
check validates proposed nested values or completeness.

The eight binding rules explicitly require the nested role to equal its
enclosing field's role and the context identity to equal a unique known nonnull
role-policy/completeness declaration, C(role). Identity compares four declared
identity fields only. Unknown/null identity, missing context, ambiguity and
conflict cannot establish equality, choose a winner or supply permission.
C(role) remains an uninstantiated schema obligation; no actual context catalog,
namespace, member set, resource limit or ordering policy is supplied. All 39
role-specific slots remain unresolved.

The two shared declarations and F01 wrapper contain 17 declared field positions.
The identity's four field declarations and marker remain unchanged from v1;
the seven reference fields and six map fields carry the v2 changes. All proposed
nested checks remain unimplemented; inherited raw-byte, JSON, generic tree,
canonical, profile and strict outer-field checks still apply. F03 null remains
unknown/HOLD. F04 required-null refusal is prospective and unimplemented.

## Node constraints and their limits

Using the preserved source counting convention, identity I=9 nodes or null=1;
reference R=13+A+C, with A,C each 1 or 9; map M=12+A+2K; direct list
L=1+sum(R_i). These are symbolic declarations, not constructed inputs or results.

| Profile | Full nonnull identity expression | Necessary node bound |
| --- | --- | --- |
| P02 | 31+2K+31N | <=2048 |
| P03 | 63+2K | <=2048 |
| P04 | 13+31N | <=2048 |
| P05 | 43+31N | <=2048 |
| P06 | 13+31(N1+N2) | <=2048 |
| P07 | 73 | <=2048 |
| P08 | 19+31(N1+N2) | <=2048 |

The expressions agree with the seven v2 constraints. Every inherited limit
remains conjunctive: raw bytes <=65536, nesting <=8, tree nodes <=2048, object
keys <=128, array items <=64, strings <=4096 characters, integer text <=20,
float text <=64, signed64 integer domain and finite floats. Individual generic
ceilings are arithmetic only and need not be jointly admissible. No concrete
role counts, payloads, test cases, expected receipts or resource bounds are
selected. Node arithmetic proves neither encoded byte fit nor decoder peak
allocation, heap/RSS, time or whole-unit fit. Outside-context work is unaccounted
by these expressions and cannot be treated as free.

## Four open context-boundary requirements

1. **Context supply domain (CBR-01).** The current public signature is
   `inspect_documentary(payload)` and the helper accepts `name, domain, value`.
   V2 deliberately supplies no runtime context input/catalog/resolver API. A
   future requirements contract must describe necessary availability and
   uncertainty information without inventing an extra argument, outer key,
   transport, constructor, global/cache, hidden lookup or I/O.
2. **Cardinality-independent uncertainty (CBR-02).** A condition on every list
   element has no element to inspect when a list is empty. Preserve the separate
   role-level missing-context HOLD obligation regardless of cardinality. Empty
   lists/maps and nullable null never establish complete history, absence of
   work, fresh authority or known equality. This is a future obligation, not an
   observed v2 admission or a new fixture.
3. **HOLD and refusal ownership (CBR-03).** The current field loop accumulates
   unsupported reasons, continues in source field order and may later raise a
   strict type/permission/admission refusal. A new context HOLD cannot silently
   short circuit later guards or replace primary errors/already occurred effects.
   Logical conjunction does not select runtime error order, receipts or a
   preflight sequence. Source/process/causal order remains distinct from global
   cross-process epochs.
4. **Complete resource/lifetime accounting (CBR-04).** Any later context
   acquisition, validation/hash/copy/read, error handling and lifetime must have
   separate source, first-effect and ownership review under all original caps.
   No overhead subtraction, hidden I/O, extra clock window or free external work
   is authorized. The current finally block clears only two local references;
   it proves no caller/traceback/parser/allocator lifetime or secure erasure.

These four requirements are OPEN and not yet drafted. Their next contract is
DATA-ONLY, bounded to these findings, the original 13 roles and 39 open slots.
It may not choose a runtime API/format, actual context instances, resolver,
implementation, new error identifiers, transition logic, case membership or
oracles. No operational approval request is created.

## Scope and preserved evidence

The new review covers 17 field relations, 13 role relations, eight binding
relations, seven symbolic bound joins, three revision choices and two current
source-signature relations. The earlier 12 source anchors and 16 conditional
lexical milestones are inherited through unchanged hashes. Earlier reviewers,
inventories and closed cohorts are not rerun. The eight new findings are
bounded source/declaration analysis, not instrumented field traces or proof of
the full original source, inverse/domain/refusal/ownership requirements.

Nine review definitions are frozen before the first independent saved audit.
That audit checks saved source/AST/JSON/filesystem/hash relations; it never calls
the author, candidate, runner, original project/test/prototype/scientific code,
or scientific value decode/recalculation, process/clock/timer, GPU/proof, network
or target ports. Launching local metadata scripts and computing their hashes is
real local work. The saved audit does not rerun the old 48 cases or supply runtime
results for v2.

All 20 DRR and 24 DAS requirements remain NOT_IMPLEMENTED/NOT_PROVEN. Eight
independent real-evidence groups remain open; formats adopted=0, reference/context
instances=0, new fixtures/case plans/calls=0. The previous 48-case/48-call pass and
657504B evidence, consumed 48/32-case scopes, older failures/corrections, 43-boundary
review, genuine proof/31 gradients, cost rejections and closed cohorts are
preserved without repeat or promotion. Whole units=0; scientific candidate/replay
not started. All failed/partial/pending/consumed flags remain fixed.

Starting checkout was clean at user commit e1014a595de30268b786fa8476805371531101ae
with the same local origin/main tracking ref. No new remote query or agent Git
mutation occurs in this review. Every prior version and exact old index history
is retained. Device 16777233 is conditional LOCAL metadata only under the prior
v4 LSTAT review; old device29 refusal, target66307 and four exact lifecycle
exceptions remain fixed. FIFO is LSTAT ONLY, never opened/read/hashed. No mount
probe, remap, waiver, physical-volume identity or numeric-change cause is claimed.

Real grants and used local approvals remain consumed; attempts=0, binding=null,
target=HOLD_NOT_READY_NOT_REQUESTED, research/training=false. The supported v5
entry still refuses before the delegate clock/ATTEMPT/connection by unchanged
source; no private/older bypass or universal physical unreachability claim.
Current remote quiescence, deployed binary/build/ABI/backend/provider and actual
current-phase execution remain unknown. No lost-primary-cause claim is made.
No university query/SSH/upload/download/install/retry/resume/retrieval/cleanup,
GPU/proof/paid work, manual paste, active operational material or target request.

The graph remains 56 uploads/936711B, runtime48 files/481518B, cache47, nine
supervisors/270 paths; historical v3's47/450548B is not current execution evidence.
Original science/native/guards/ownership/hash/immutable copies/every fresh hash/
ordered state_digest/durability/shutdown/continuing clocks remain fixed. Objective
120/unit720/setup180/proof60/soft2280-hard2400/final120/collection600/connection3000,
RSS4GiB/allocator7GiB and all other caps remain fixed. Training PGD5/EOT1, routine
EOT1, optimizer backward, registry/study YAML and financial200/160/40/single10 caps
and holds remain fixed; old snapshots are not current quotes or balances. All
later source/inverse/domain/refusal/ownership/complete-unit timing/memory/fresh
replay/recovery/economics/calibration/data/protocol/training gates remain separate.
No training percentage or time estimate is supplied.
