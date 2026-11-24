# Early-device whole-PGD preparation: local review

The separate cold preparation/operator is implemented and reviewed locally.
Destination tensors move to CUDA before the unchanged CPU mmap checkpoint audit
and strict copy. Original model construction, scientific sources, ownership and
resource caps remain. There is no new target authorization or attended launch.

Seventeen distinct adapter cases pass, plus one fresh small CPU non-gradient
control once. The independent saved audit v2 passes. This qualifies local source,
ordering and fake integration behavior; actual CUDA copy equivalence, host-RSS
reduction and target memory fit remain unqualified. See the
[implementation and limits](../../docs/predictor_pgd_workload_preparation_v1.md)
and [machine-readable review](2026-11-23-pgd-workload-preparation-v1-local-review.json).

| Evidence | Result |
| --- | --- |
| Runtime derivation | 47 files/450548 B; 46 byte-exact files, one provider with two inverse edits |
| Cold operator | Five inverse edits to pinned metadata builder; 57 input pins |
| Primary/convergence | 55 upload sources; 896039/1005007 B; 46-module cache; 264/624 result paths |
| Adapter checks | 17 PASS /34 executions: first17 fixture setup errors, then only those17 PASS |
| Production/test edits after first checks | None; only runner evidence-base setup corrected |
| Positive fake integration | Primary432 and convergence792 files, all four roles, false proof/compute provenance retained |
| Failure fake integration | Four stop-before-later-role cases, each171 files and verified transport |
| Fresh real CPU control | 60 parameters, new seeds, two forward evaluations, zero gradients; exact state/output bytes |
| Saved audit v2 | PASS:46720 prior hashes/51412 prior fixtures/4326 new fixtures/all pins/inverses/six wires |
| Prior grants/claims | All consumed; v2 target RSS failure and v1 unknown remote progress preserved |

The CPU control uses the byte-exact frozen canonicalization/load functions with a
new 4480-byte three-component toy checkpoint. It preserves parameter identities,
requires-grad flags, teacher independence/frozen status, modes and CPU RNG; two
malformed-state controls reject before any component copying. It is actual local
Torch computation, not fake numerical metadata, and does not exercise GPU transfer,
pretrained objectives or any gradient. All adapter lifecycle compute/OS/resource
ports are explicitly fake. Ordinary reserve/archive/source/JSON copies are real
filesystem work. No old successful tiny gradient or closed scientific cohort reran.

The first independent auditor rejected a blanket no-AUTHORIZATION-filename check;
the fake coordinator intentionally writes such records. Audit v2 narrowly verifies
their fake basis, controls and exact upper/lowercase bytes. V1 source/log/failure
are retained; no project/test/scientific callback or control was rerun for that
correction. Both audits keep old FIFO LSTAT-only and the reviewed device observation.

Six raw priority streams were independently parsed through EOF, with every payload/
reference, source/saved hash, mode/mtime, journal row/offset/hash chain, prefix,
terminal and trailing-byte boundary checked. Work/session/partial/journal FALSE
completion fields stay FALSE; separate transport TRUE does not promote science.
Fake proof remains passed/complete FALSE and zero bytes; its real reader refuses.

| Fake lifecycle | Files | Wire bytes | SHA256 |
| --- | ---: | ---: | --- |
| Primary | 432 | 44816851 | `8c3e2a5f2eca529044e76830e397f03c97758afd313cf625ac4f2267382da6bd` |
| Convergence | 792 | 45450736 | `fe09353cc12fe199293144ffa14b3092ebee302f9e476da8f8e696249452dff4` |
| Transfer refusal | 171 | 44481851 | `5ca3fd252af4b61001911353ab258864d433f77af22e5d22d2b87e657d8f3188` |
| Checkpoint refusal | 171 | 44481855 | `4650d939d1223c88961fa9bd65844e56f40c6a2163196edc931cdeaa3525d91f` |
| Transfer deadline | 171 | 44482479 | `5d5c41b9ac6759a84ac63679f6e156c0032a16e997dca6d37b5a9ad768205aa1` |
| Original RSS guard | 171 | 44481769 | `29a26227b0168c4ee27737e8279b6b89b714aaacec0f009966fc6269b2cb6849` |

All previous sources/test/auditor versions, fixture identities, logs, failures,
records and claims are preserved. New stage:
`runs/pretrained/diagnostic/pgd-workload-preparation-v1-20261123`;
seal `sealed-workload-preparation-v1.json`. Live mutable indexes have retained
before/after snapshots and are excluded from the seal. User stages/commits/pushes.

Next local: derive and review a separate corrected client/bootstrap binding to
these new bytes before any exact scope/handoff/approval. Keep the consumed v2
package unchanged. No target work, scope or approval request occurred here.
Target full units/replay/performance, memory/recovery, economics/calibration and
data/protocol gates remain. Mechanism stays closed, research/training stays FALSE,
and training policies/study/financial caps and holds stay unchanged.
