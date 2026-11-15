# Attack integration transition and private collection: local review

The completed one-clip clean EOT8 numerical/replay gate remains closed. The next
implementation is a new, opt-in **per-iterate input-gradient context for EOT8 PGD**.
The passing route is single-use and native-owner-token bound; the old PGD loop
calls autograd directly and includes no-grad clean/final/scoring evaluations.
Those paths cannot be combined by globally enabling the diagnostic route.

The source audit records **40 differentiations / 44 attack-bank forwards** per
PGD20/two-restart primary clip and 100/104 for the stronger PGD50 control. These
are logical counts, not timing estimates. The new local prototype is tiny CPU
PGD2/two-restart EOT8: four candidate and four same-input original checks, with at
most four conditional fresh-candidate replay objectives after passing initial
gates. Existing per-objective guards and bounds remain. No old clean cohort runs.

Parameter-update, training PGD5/EOT1, routine/EOT1, source policies, study YAML,
mask/selection rules, split closure and calibration remain unchanged. A clean
input-gradient pass does not establish PGD sign/trajectory equivalence. Kernel,
end-to-end performance, chosen-hardware/recovery, calibration and full-study cost
still need scoped evidence before training. Diagnostic worker walls are not
benchmark units; no speedup or new cost estimate is produced. No target handoff,
helper, new execution grant, claim or GPU request is prepared here.

A separate **local-only collection receiver** reuses the frozen decoder code
object and adds a pinned source descriptor plus private directory reconstruction.
All **26 distinct focused checks** have passing final outcomes, including the
actual **242-file / 599,605,882-logical-byte** collection and direct strict capsule
admission. The original response/mirror remain untouched, and consumed clients
are unchanged. The process umask restores on every success/refusal. Altered hashes,
framing, links, paths, modes, source replacement and reused destinations refuse.

The first suite's input-mode failure is preserved: the original response is 0644
inside its private session, so the receiver accepts that read-only source without
chmod while keeping all new output private. Only the failed and new checks ran
after correction. Saved audit assumptions about uppercase grant filenames were
also corrected: archived lowercase control/authorization may resolve through case
aliases locally; their canonical handoff and original consumed claim are unchanged.
No copied historical authorization is treated as a new grant.

Independent standard-library review verifies **3,237 prior sealed hashes**, all
**242/280/135/245 old target file descriptors**, new sources, reconstruction and
inert plan. First source/test/audit failures remain saved. No model, autograd,
SDPA, new oracle, optimizer, scientific cohort, GPU proof, SSH, upload, installation,
remote cleanup or paid work occurred. Financial caps, holds and pending
reconciliation remain unchanged.

Next: implement only the declared bounded local tiny PGD gradient-context
prototype and its new checks. Target/backend/performance work requires later
concrete preparation and a new exact scoped approval; no manual GPU check is
needed for the local prototype.
