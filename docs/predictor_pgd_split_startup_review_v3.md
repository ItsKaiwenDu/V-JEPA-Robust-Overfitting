# Local startup dependency and parent-exit review

This is a local-only investigation plan following the independently reviewed v2
startup refusal. It is not a new target helper, authorization, experiment or grant
extension. The consumed v1/v2 source, controls, proofs, claims and failures remain
frozen. Do not rerun a previous command.

## Confirmed child failure

The original packet already contains hash-pinned datasets/__init__.py and
prepared.py plus four other dataset source files. The installed loader caches
63 runtime and six supervisor modules, all under models/attacks. It neither maps
nor exclusively owns the datasets namespace. Isolated startup therefore failed
at native_diagnostic's import of datasets.prepared. Avoid a broad sys.path or
ambient-site-package fallback: any correction must load the exact declared
project source bytes and protect the same namespace against unpinned imports,
reload, source replacement and package-path changes.

Review the whole path to startup from the actual frozen numerical and component
entries, not just the single reported import. The conservative saved inventory has
195 project edges and fourteen dynamic import sites. Its other three unmatched
model references may occur in deferred/dead branches. Classify those paths from
their source and record the reason each is required, deferred or unreachable for
this scope. Do not blindly add every flagged file or treat static closure as an
executed import test. Include package parents, relative imports, from-package
submodule resolution, dynamic imports, source precedence and the upstream src
namespace used by the original gate.

The setup must resolve its original scratch/coordinator sources before packet
extraction. Packet-backed dependencies become available after extraction and
verification. Preserve this ordering and the exact original packet/member hashes;
never let older packet model files override newer declared overlays. Reconcile
any loader/control/source-count changes in a new version and contract rather than
editing the consumed v2 or weakening a frozen count/hash assertion.

Before considering a remote attempt, add focused meaningful local checks using
the actual source catalogue and loader resolution, with inert source definitions
or explicit external-library stubs where necessary. Keep all scientific functions,
Torch/NumPy execution, arrays/autograd, model construction, CPU references, storage
proofs, timers, processes, proc/PID operations, GPU and network calls out of this
local work. Document precisely what the stubs do not verify. No old scientific
cohort or already-passing unrelated test function needs repeating.

## Separately observed parent error

The parent saved ProcessLookupError while the child saved ModuleNotFoundError.
No parent traceback identifies the exact raising instruction. Inspect the frozen
launch/watch/exit/cleanup code and reproduce plausible early-exit interleavings
with inert fake children, readers, clocks and held PID handles. The child failure
must stay a failure, later work must stop, and the owned child must be confirmed
ended with its handle closed before finalization/collection can proceed. A race
must not create authority to signal a guessed numeric PID or an unrelated birth.
Keep deadlines and cleanup grace within the original clocks. Do not claim that a
source-only reproduction proves which interleaving happened on the target.

## Completion boundary

Review and seal the complete corrected package, original scientific bytes,
namespace/ownership gates and limits before asking for one new exact university
approval. No fresh manual hostname/nvidia-smi check is needed; future approved
helpers must perform live guards. The existing grant permits no extra connection,
proof, rerun, retrieval, cleanup, installation or paid compute. Financial holds and
scientific/training admission stay unchanged.
