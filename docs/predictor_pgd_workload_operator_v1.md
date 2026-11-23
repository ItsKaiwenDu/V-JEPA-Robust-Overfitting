# Whole-PGD OS/bootstrap/client adapter — local review

The concrete adapter around the reviewed whole-PGD runtime is implemented and
reviewed locally. It retains all 47 runtime files / 450,166 bytes exactly and
adds the executable cache, child dispatch, nested clocks, resource/completion
readers and attended client wiring. Live execution remains unbound. There is no
new attended scope, launch package, real authorization, SSH connection or GPU work
in this stage.

The [review](../report/checks/2026-11-23-pgd-workload-operator-v1-local-review.md)
records 22 distinct checks passing over 52 executions, with only failed nodes
repeated. The [design contract](../configs/predictor_pgd_workload_operator_v1.json)
and [active input pins](../configs/predictor_pgd_workload_operator_inputs_v2.json)
describe the concrete sources. Input pins v1 and every earlier generated version
are retained as history; v2 is the active contract.

## Source and namespace integration

The upload contains 55 Python sources: 46 runtime modules plus nine supervisor
slots. The original 47-file runtime includes the whole-unit monitor, which remains
byte-exact in its supervisor slot. The other slots include the original proof
monitor and allocator, bootstrap, scratch guard, coordinator, collector, child
support and frozen plan helper. The executable cache closes literal imports over
models, attacks and datasets, refusing ambient project modules, reloads and
changed loader/origin/parent/path/source identities.

The plan helper was absent from the runtime closure and was added as separate
unchanged supervisor support. No scientific module was modified to supply it.
The original scanner, assessment, scratch Guard, GPU/host/device/mount predicates,
owned-handle launch/cleanup and proof recipe retain their reviewed sources.
Every derived supervisor/client has a checked whole-byte inverse. The original
Alarms class, original-packet receive/extract functions, causal client clock and
collection-root binding functions remain AST-exact.

Primary and convergence have separate cold source identities. The primary source
upload is 895,657 bytes; convergence is 1,004,625 bytes, both below the original
2 MiB envelope. Their result catalogues have 264 / 624 paths. Proof logging is
root supervisor metadata at proof.log; the unused capture/logs/proof.log path is
not admitted. All shared byte, partial-publication, entry-count, RSS/allocator,
process/thread and source ceilings remain unchanged.

The adapter reuses the original 17,192,095-byte archive with SHA
2278bce2b519943c0f83a49c556bb73dd928fe880fba0f6c56de2a4c3c2ec0da:
99 fixed members / 26,446,958 expanded bytes. It sends one archive and consumes
final EOF. Extraction verifies every source/payload hash and private ancestor.
No new pixel/checkpoint archive was generated; fixture copies are opaque bytes.
The component capsule/native-reference second-archive protocol is disabled here.

## Owned child, timer and completion path

Role order is proof, original-unit, candidate-unit, candidate-replay. Concrete
launch creates the release pipe, isolated -I -B child and owned PID handle before
release. Child bootstrap validates exact control, authorization, parent session,
plan/source and still-owned process/session identities before cache installation.
Dispatch then verifies original host/GPU/storage/source gates and selects only
the new whole-unit entry. Original packet/checkpoint/upstream preparation and
full-size CUDA runtime requirements remain in the unchanged worker graph.

The child support takes over only the authenticated active release timer with
SIGALRM at its default handler. The unit timer has an absolute deadline. Each
gradient temporarily arms a hard deadline no later than 120 seconds or the
remaining unit time. Successful exit restores the original absolute unit
deadline. Failure poisons the timer; an expired or caught failure cannot start
another objective or convert the unit into success. The original hard alarm and
soft wake/generation logic is reused, with no timer reset to a fresh allowance.

Each unit is capped at 720 seconds. The existing outer work soft deadline remains
2280, hard work deadline2400, finalization reserve120, collection600 and connection
deadline3000 including login. Per-role deadlines use remaining work time, so a
later unit may receive less than720 seconds or be refused. These are maxima,
not a promise that all three units fit. Setup180 and proof60 remain bounded.
The parent monitor keeps 4 GiB observed RSS, 7 GiB worker allocator, the original
ownership/call limits and explicitly non-exhaustive sampling interpretation.

Only the authenticated child binds the new entry and its objective timer. The
argument bridge fixes the selected method/diagnostic step/audit banks, exact
plan/source hash, original packet and existing checkpoint/upstream paths.
After the unit returns, anchored readers join the immutable start, every gradient
start/complete pair, unit summary, worker result, resource and completion records.
The parent joins the same new PID, plan/control, wall/RSS and resource identity.
Fresh candidate replay must match logical return hashes, metrics, scoring and
selection. Metadata does not assert original/candidate numerical agreement or
authenticate scientific execution by itself.

## Local evidence and retained failures

All adapter checks use private fake OS/import/process/pipe/PIDFD/GPU/proof/clock/
timer/compute namespaces. Real sys.modules project identities and sys.meta_path
remain unchanged; the observed extra modules were only grp, gzip, pwd and tarfile.
Worker model entry is replaced by an explicit fake numerical port in those
namespaces. No Torch/NumPy/model/gradient, actual worker process/PIDFD/signal/timer,
2.5 GiB proof, GPU, network or paid work ran. Python test infrastructure is not a
physical worker-enforcement test.

Opaque archive/source/JSON copying, anchored file I/O, byte hashing and ordinary
16 MiB allocated reserve fixtures are real filesystem work. The fake proof stays
passed=false / complete=false / zero bytes; its simulated verifier is marked as
such, and the unchanged real proof reader rejects it. True approved fields in
test authorization fixtures use the explicit fake basis and are not user grants.

The final primary/convergence simulations verify priority/dedup transport for
432 / 792 files. Standalone parsing independently reconstructs every frame,
payload/reference, mode/mtime/hash, prefix, terminal and trailing-byte boundary.
Session complete=false, work collection=false and partial journal false flags
remain intact; only the separate transport result is terminal-verified.
An added case uses saved wire bytes to preserve an interrupted prefix and confirm
that the client refuses a non-university scratch-root binding.

The execution history is retained in full:

| Check batch | Executions | Pass / fail |
| --- | ---: | ---: |
| First checks | 21 | 6 / 15 |
| Failed cases after fixture ancestor fix | 15 | 1 / 14 |
| Failed cases after canonical catalogue fix | 14 | 13 / 1 |
| Remaining namespace assertion case | 1 | 1 / 0 |
| New saved-stream case | 1 | 1 / 0 |

Initial fixture ancestors were accidentally 0755; only new fixture construction
was corrected to0700. The isolated archive check now precreates its declared
ancestors, as the production setup plan does. The only post-check production
correction changes eight new plan/parent catalogue values from tuples to canonical
JSON lists. All functions and scientific sources remain unchanged by that fix.
The final failed assertion was narrowed to the intended project-cache/import
machinery boundary while recording ordinary stdlib additions. No passing node,
previous successful tiny CPU gradient or completed scientific cohort was rerun.

The standalone stdlib audit v2 passes on 36,488 prior hashes, 40,097 prior fixture
entries, 9,274 new fixture entries, 56 input pins, source inverses and both complete
wire reconstructions. Audit v1 refused an incorrect expected pin count55; only the
audit expectation changed to56. No production/test/scientific work was repeated
for that correction. Audits execute no project/test callbacks or array
deserialization/value recalculation. Old FIFO lstat-only. Device16777233 matches
the prior conditional local observation; no old device record or target mapping
was rewritten.

Raw stage: runs/pretrained/diagnostic/pgd-workload-operator-v1-20261123.
Seal: sealed-workload-operator-v1.json. All source generations, failures,
snapshots, fixtures, logs, mutated negative records and old claims remain.
Live mutable indexes have sealed before/after snapshots and remain excluded
from the root seal. The user stages/commits/pushes.

## Next separate step

Prepare and review one exact attended scope/package from a selected method's
concrete bytes before requesting any new authorization. The primary helper is a
diagnostic local default, not a calibrated research choice. No live binding or
new university work follows from this local review. All prior grants are consumed.

Actual target proof/ownership/clock/resource enforcement, fresh whole-unit
primary/convergence results, perturbed target replay and matched performance
still need their own evidence. Hardware memory/exact recovery, complete economics,
training-only calibration and data/protocol/adoption gates remain. The mechanism
diagnostic stays closed; no new FP64 oracle/full-model trace or repeat of closed
cohorts was added. Training PGD5/EOT1, routine EOT1, optimizer backward, registry
and study YAML remain unchanged. Project200/compute160/noncompute40/single10
caps and financial holds are unchanged; October29 financial snapshots are
historical. Research/training admission remains false.

