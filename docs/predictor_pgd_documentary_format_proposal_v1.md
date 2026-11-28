# Local documentary reference-format proposal

**Proposal only: two shared declarations, four family formats,13 role
mappings and39 unresolved context slots. Nothing is adopted, instantiated,
implemented or dynamically validated. The validator, runtime, real admission
and training remain unchanged and closed.**

Preparation starts clean at user-pushed98ebd034. The user's “Committed;
next step!” directs the [separate declarative proposal scope](../runs/pretrained/diagnostic/pgd-documentary-reference-requirements-v1-20261126/NEXT-LOCAL-DOCUMENTARY-FORMAT-PROPOSAL-v1.json),
recorded in the [task basis](../runs/pretrained/diagnostic/pgd-documentary-format-proposal-v1-20261127/TASK-BASIS-v1.json).
It supplies no adoption, implementation, reference instance, case plan/run,
resolver, active record, operational request or target permission. No agent
Git mutation. All preceding review/requirements files are now committed by
the user; their earlier dirty-status records remain exact historical bytes.

See the [complete declarative proposal](../configs/predictor_pgd_documentary_format_proposal_v1.json),
[saved review](../report/checks/2026-11-27-pgd-documentary-format-proposal-v1-local-review.md),
[shared declarations](../runs/pretrained/diagnostic/pgd-documentary-format-proposal-v1-20261127/SHARED-SHAPE-PROPOSALS-v1.json),
[four families](../runs/pretrained/diagnostic/pgd-documentary-format-proposal-v1-20261127/FAMILY-FORMAT-PROPOSALS-v1.json)
and [all13 role contexts](../runs/pretrained/diagnostic/pgd-documentary-format-proposal-v1-20261127/ROLE-CONTEXTS-v1.json).
These files describe fields and textual constraints. They are not executable
JSON Schema, a parser, populated reference examples, test inputs or oracles.
Administrative source hashes/freeze records are not instances of the proposed
reference formats.

## Shared declarations

| Declaration | Proposed closed required fields | Meaning and limits |
| --- | --- | --- |
| DPS-IDENTITY | identity_kind, sha256, byte_length, byte_domain | A local marker,64 lowercase hex characters, nonnegative exact integer and exact-supplied-documentary-byte domain. No byte read, hash comparison, origin or permission is supplied. |
| DPS-REFERENCE | reference_kind, revision, role_id, relation_label, artifact_identity, context_identity, evidence_status | Local marker/revision1, exact frozen role name, unresolved context-declared relation label, nullable documentary identities and existing documentary evidence label. No locator or resolver. |

Each declaration proposes exact key/type/literal checks and refusal of missing,
extra, duplicate or coerced values. No boolean-as-integer conversion. Names
and relation labels are exact strings without case folding/path normalization;
their complete allowed vocabularies and extents remain unresolved. An identity
describes original supplied bytes, not a latest path or silently re-encoded
replacement. Its byte_length uses the inherited signed64 metadata number
domain only; that domain is not a role artifact-size allowance or resource cap.

The encoding proposal follows the inherited canonical UTF-8 JSON spelling:
sorted keys, compact separators, ASCII escaping, finite numbers and one final
newline. No instance has been encoded or parsed. Fit within the current raw/
numeric/nesting/tree/object/array bounds is NOT established, and those bounds
remain unchanged. The graph of proposed references is nonrecursive: references
contain documentary identities, never nested reference objects or locations.

The five existing UNKNOWN/UNAVAILABLE/SYNTHETIC/SOURCE_ASSERTION/
REVIEWED_SAVED_RECORD labels are retained as labels only. They do not authenticate
supplied bytes, independent review, actual source origins, complete history,
fresh authority, a physical clock/owner or workload fit. Null artifact/context
identity is unknown/unavailable, not empty bytes, zero work, complete evidence
or permission. A later documentary match would establish only the declared
identity after separate evidence review.

## Four proposed family forms

| Family / roles | Proposed form | Required unresolved context and null meaning |
| --- | --- | --- |
| F01: two named source/control maps | Closed wrapper with local format_kind, revision, role_id, namespace_identity, member_hashes, evidence_status. member_hashes maps exact names to64-character lowercase hashes. | Complete immutable allowed member namespace/spelling/byte domains required. Null or missing namespace identity stays HOLD. Null wrapper would need refusal. |
| F02: seven evidence sequences | Closed wrapper with local format_kind, revision, role_id, context_identity, ordered references and evidence_status. Nested references must use the wrapper role. | Role relation vocabulary, completeness, count/order/duplicate/alias rules required. Empty sequence never proves complete history/ownership/roles. Null wrapper would need refusal. |
| F03: user-basis and clock-owner roles | Outer null or DPS-REFERENCE. | Null is unknown/HOLD. A non-null reference still cannot supply actual authority or a continuing physical clock. |
| F04: full workload and fixed-cap roles | DPS-REFERENCE only. | Outer null would need explicit future refusal. A non-null reference can still have unknown artifact/context identity, which stays unresolved/HOLD rather than fabricated. |

The two wrappers have six declared fields each. Together with the4-field
identity and7-field reference there are23 declared field positions, reused
by the four families and13 roles. These are local proposed forms, not actual
operational grant/control parser forms or runtime adapters. Their marker
strings do not themselves prove all parser domains physically disjoint.
Current P02–P08 descriptor branches remain unsupported. These proposals are
not drop-in replacements and do not prove an inverse or original-domain fit.

## All39 context slots remain unresolved

Each of the13 original field roles maps to exactly one family and three slots:

- CONTEXT: complete immutable namespace or role-specific relation vocabulary
  and completeness declaration, using the exact prior cardinality obligation.
- BOUNDS: justified concrete name/string/count/byte/extent limits against the
  complete role context and inherited parser limits. No implicit zero,
  unbounded value, truncated evidence or physical quota claim.
- ORDER: role-specific ordering/multiplicity/alias/ambiguity policy, retaining
  the exact prior obligation rather than sorting, merging or choosing a winner.

Every slot is UNRESOLVED; no actual member set, bound definition or policy
instance is supplied. Source/upload/runtime/cache/supervisor/control namespaces
remain distinct, and historical55-upload/47-runtime cannot replace current
56-upload/48-runtime contexts. Known counts are not actual complete definitions.
Logical single-reference arity does not select encoded resource limits.

The role map preserves the13 exact prior field IDs, source field ordinals,
requirement text and family/cross-family obligations. It introduces no new
operational role or claim. Missing, mismatched, ambiguous and contradictory
references retain all supplied records; no first/last/newest winner, implicit
lookup, status promotion, consumed-state reset or default complete history.

## Coverage is separate from source and evidence review

[Twenty requirement relations](../runs/pretrained/diagnostic/pgd-documentary-format-proposal-v1-20261127/REQUIREMENT-COVERAGE-v1.json)
map every DRR requirement to the proposed declarations. All20 new requirements
and24 original DAS requirements remain NOT_IMPLEMENTED/NOT_PROVEN. All eight
independent real-evidence groups are copied exactly and stay open. Adopted
formats0, proposed instances0, local/real proof gaps closed0.

The [eight source/error review obligations](../runs/pretrained/diagnostic/pgd-documentary-format-proposal-v1-20261127/SOURCE-ERROR-REVIEW-OBLIGATIONS-v1.json)
retain the exact parent refusal conditions and obligations, with new review
tasks for these declarations. Every obligation is SEPARATE_SOURCE_REVIEW_REQUIRED_
NOT_PERFORMED. Coverage and metadata integrity do not perform that full review.
It must address aggregate parser bounds and compatibility, original field/
process/effect/refusal order, all identity/context/duplicate/alias errors,
first-allocation/read ownership, primary diagnostics/partial records and any
new fallible work. Missing final receipts never erase earlier effects.

No location/path/URI/executable/action/PID/ATTEMPT field or automatic file/network
resolution is proposed. A future resolver or additional validation still adds
fallible nonzero work requiring separate source/inverse/domain/refusal/lifetime/
ownership review and explicit scope. No hidden read/import/clock/target action
is authorized by these declarations. No failure/denial port was exercised,
case membership/oracle constructed, reference instantiated or validator called.

## Saved audit and unchanged gates

Independent saved stdlib audit v1 passed first execution:72,647 prior hashes,
78,129 prior fixtures/0new,123 pins/seven frozen definitions/two shared shapes/
four families/23 field positions/13 role mappings/39 unresolved slots/20
requirement relations/eight review obligations. It joins saved JSON/source/AST/
stat/hash metadata without calling author/candidate/runner/project/test/
prototype/scientific code. No scientific-value decode/recalculation, process/
clock/timer/GPU/proof/network/target work IN AUDIT. Metadata work is real.

All old source/test/auditor/fixture/log/failure/result/status/claim versions
remain. Prior48-case first pass/657504B evidence/consumed local scope, old32case/
43boundary/scientific proof/cost/closed cohorts inherited without repetition
or promotion. Old genuine proof31gradients/wholeunits0/scientific candidate-
replay notstarted/all failed-partial-pending-consumed states fixed. Old47/450548B
is not current48/481518B execution. Current remote quiescence/deployed binary/
build/ABI/backend/provider/current-phase execution and old lost primary cause
remain unknown. All real and used local scopes consumed, attempts0/bindingnull,
target HOLD/research-trainingfalse. No training percentage/time estimate.

Budget REJECT_CURRENT_ROUTE_FOR_NEW_SCOPE; supportedv5 unconditionally refuses
before delegateclock/ATTEMPT/connection, no private/older bypass. Graph56uploads
936711B/runtime48files481518B/cache47/nine supervisors/270paths/control/manifest
hashes fixed. Original science/native/guards/hash/immutable copies/EVERY fresh
hash/ordered state_digest/owner/durability/shutdown/continuing clocks fixed.
Objective120/unit720/setup180/proof60/soft2280-hard2400/final120/collection600/
connection3000/RSS4GiB/allocator7GiB/all other caps fixed. TrainingPGD5/EOT1/
routineEOT1/optimizer backward/registry/study YAML and financial200/160/40/single10
caps/holds fixed; historical snapshots not current quotes/balance. No benefit/
physical memory/time/security/ownership/resource-fit claim.

No university query/SSH/upload/download/install/retry/resume/retrieval/cleanup/
GPU/proof/paid work/manual paste, active grant/control/handoff/package/launch,
operational request, original model/Snapshot/Torch/NumPy/tensor/gradient/worker/
FileSink/ledger/scientific callbacks. Device16777233 prior v4 LSTAT review is
conditional local metadata only; old29 refusal/target66307/four exact lifecycle
exceptions and FIFO LSTAT-only never-open/read/hash retained. No mount probe/
remap/waiver/volume claim; new device change needs new record/review. Four live
indexes excluded from seal, exact before/after snapshots/history included.
No agent Git mutation.

## Next separate local step

[The next requirement](../runs/pretrained/diagnostic/pgd-documentary-format-proposal-v1-20261127/NEXT-LOCAL-PROPOSAL-SOURCE-REVIEW-v1.json)
is a bounded source-only review of these NEW declarations against the pinned
documentary parser and relevant preserved refusal boundaries. Assess the23
field positions, two shared/four family forms,13 mappings/39 context slots and
eight source/error obligations. Identify incompatibilities and missing choices
without adopting, implementing, resolving or instantiating forms. No source
change/compile for execution/import/case plan/run/prototype/benchmark/original
callback/binding/gate edit/operational request/target action is authorized.
