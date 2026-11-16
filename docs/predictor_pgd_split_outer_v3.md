# Packet-aware outer handoff v3

V3 addresses the confirmed datasets import gap without adding files to the original
packet or changing scientific source. The bootstrap declares only two additional
module names: datasets and datasets.prepared. Both use the existing code/datasets
members, whose exact sizes and hashes are pinned in the frozen packet manifest.
All 63 scientific/runtime sources and the original packet stay byte-identical.

## Private import resolution

The initial stdin bootstrap must import the original supervisor sources before
packet extraction. It still verifies and caches exactly 69 original supervisor/
overlay names then. Packet-backed imports are declared at construction but read
lazily, after the packet has been received, extracted and verified. The original
held/no-follow/private-mode/root-identity reader verifies their bytes at resolution
and again at execution. Unknown datasets names, foreign package paths, preload,
reload, descriptor/source drift, symlinks and nonprivate or linked inputs refuse.
No cwd, sys.path or site-package project fallback is introduced. Overlay model
sources retain precedence over older packet model copies.

Namespace validation binds the exact packet policy, module origin, loader and
package path. It permits the real lifecycle in which identical bootstrap definitions
execute once in the initial checked stdin namespace and again as a pinned module.
Those copies have different Python class objects. A local regression exposed and
corrected the first candidate's inappropriate isinstance check; it now validates
the exact reviewed policy descriptors rather than class identity across copies.
This is source/control integrity checking, not protection against arbitrary code
already executing inside the same process.

The real catalogue's 71 module names resolve in the new local loader checks.
Compilation uses actual pinned source bytes; their execution boundary is replaced
by AST import traversal. Deferred/function imports are conservatively included,
and only literal if-False bodies are excluded. Seven inactive import statements
include the four occurrences of the three old unmatched model names. Fourteen
dynamic import sites have literal upstream src targets or Python sys. The
upstream gate and its source remain unchanged. No external library or scientific
module body executes. This proves source resolution under the declared catalogue,
not actual external-library behavior, exported-symbol evaluation, upstream module
execution or numerical correctness.

## Parent early-exit diagnostics

Inert child/handle tests reproduce three early-exit interleavings through the
original component and numerical monitors. A disappearing process can yield the
observed ProcessLookupError while preserving failure, bounded wait and owned
handle closure. This does not identify the exact instruction from the old live
run, whose parent traceback was not saved. No ownership or cleanup bypass is added.

The generated coordinator retains the complete previous source prefix. Two small
wrappers add the already-owned child's observed exit status as an exception note
and save bounded parent error frames/notes in the existing session record. At most
eight frames and four notes are kept, no locals are captured, and the diagnostic
has a 16 KiB hard byte cap within the existing 128 KiB session-file bound. Failed
saving never promotes the run or replaces the primary error. The old process
launch, release, watch, stop, finally, handle, signal and clock logic is unchanged.
There is no new signal/wait, retry, successful-result fallback or guessed PID.

## Whole-package validation and preservation

33 distinct new checks have passing latest results over 57 executions. Two failing
executions are retained: a test inventory assertion counted only four project
imports while seven total inactive imports exist, and the real duplicate-bootstrap
class-identity regression. The count assertion was corrected in the test only;
only that failed case repeated. The source correction changed generated bootstrap
bytes, so 22 affected passing loader/package checks repeated. The ten unaffected
monitor/diagnostic checks did not repeat. No historical test function or scientific
cohort repeated. All five logs, first/changed source snapshots and failed fixtures
remain saved.

The integration calls actual v3 setup with the original 172 assets, 419 permitted
result files and 39-directory catalogue. It caches 69 source modules before packet
contents are available, leaves both packet dependencies lazy and creates the
reviewed 26 setup directories while preserving all twelve worker claims. Real
private source writes and Guard scans run. Host/GPU, importer activation, packet
receive/extract/content checks, emergency allocation and proof admission are
explicit substitutes; the reserve fixture is zero bytes. Timers, processes,
PID/proc calls, actual signals, arrays, model/gradient/backend, CPU reference,
profiler, storage proof and SSH never run locally.

Only two of 69 uploaded sources change: bootstrap.py and the generated coordinator.
Each retains its complete earlier prefix and appends reviewed definitions. The
other 67 files are identical to the last target copy, including every scientific
source and original monitor. The outer driver has new source/scope identities and
a new source preparer; all timer, claim, transport, setup, directory, GPU/storage,
collection and numerical bounds retain the previous definitions. The client and
CLI change only their bound module import. No consumed source or claim is edited.

## Proposed next attempt

The complete candidate is prepared with only CONTROL-CANDIDATE.json,
SOURCE-PAYLOAD.json, PACKAGE.json and known-hosts. It has no authorization or attempt
claim and no local or remote work has started. Independent saved-file/AST review
must finish before one additional exact scoped approval can arm it.

All earlier grants are consumed. No repeated manual GPU check is needed: the future
approved helper checks live. Scope remains 40 minutes of work including full
2.5 GiB proof, eight matched gradients and two component VJPs/one inner VJP, then
up to ten minutes collection on the same SSH, within 50 minutes including login.
Original numerical/backend/ownership/resource guards remain. No extra objectives,
CPU reference, warmup, replay/cohort repeat, retries, separate retrieval, old-file
cleanup, installation, private-key upload, held-out pixels or paid compute are
included. Scientific/training adoption, financial caps, holds and reconciliation
remain unchanged.
