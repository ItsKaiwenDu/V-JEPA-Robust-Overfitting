# Whole-PGD client refusal and unbound local repair

The attended whole-PGD attempt **failed** at the client's durable-root claim
check. Its exact grant is **consumed**. The independent review passes on the
preserved failure; it does not turn the attempt into success. All previous
preparation/arming records stay historical and unchanged.

The first reads were the bound local ATTEMPT.json and attended-session errors.
ATTEMPT joins outer49c76bde... and authorizationbad05798... with consumed=true.
CLIENT-REFUSAL.json records ValueError, “Durable remote claim or continuing
connection clock changed.” The only other session file is an empty ssh-stderr.log.
There is no raw claim, ROOT-RECEIPT, collection-ready/receipt, saved transport,
proof or worker/unit evidence. The SSH exit status is not recorded.

The user ran a real attended SSH command. **Remote setup/proof/GPU/worker progress
is unknown**: the server sends its claim before prepare_tree, and the client did
not preserve the rejected frame. Absence of collected evidence is not evidence
that no remote computation ran. No new SSH/query/upload/retrieval or target work
was performed during this review; the consumed grant is not retried or reset.

## Confirmed source defect

The exact approved launcher constructs:
`/var/tmp/vjepa-pgd-workload-v1-<uid>-<outer-sha>`.
The exact client still strips and requires `vjepa-pgd-candidate-v1-`. Therefore
a standard claim emitted by that launcher necessarily fails the client's path
predicate. The old scope JSON also retained the candidate prefix. Those records
are preserved; the defect is not repaired by rewriting the rejected package.
Actual rejected frame values are unavailable, so this is a source-confirmed
incompatibility reproducing the error, not a fabricated target-frame analysis.

The earlier checks missed the positive claim path between these two sources.
They verified bytes, preflight/refusals and fake worker/transport boundaries but
could pass while the source namespaces disagreed. The new controls exercise the
source-derived server claim through the actual client run body with fake ports.

## Local repair and evidence

The separate derivation helper produces an **unbound** client artifact. Its only
three edits are the two workload-prefix corrections and an exclusive bounded
CLAIM-OBSERVED.json write immediately after reading a claim, before its guards.
All other client bytes have a full inverse to the sealed source. All original
server, scientific, resource/clock gates, uploads, policy and CLI sources remain
unchanged. The observation is local client metadata, bounded by the original
64KiB claim frame; it does not expand remote result/storage caps or accept a
rejected claim. ROOT-RECEIPT remains reserved for accepted claims.

Sixteen new checks pass on their first run, with no corrections or repeats:
one byte-inverse check, one old-client counterexample, one repaired-client fake
lifecycle and13 negative claim mutations. The latter preserve strict prefix,
parent, UID, path/control hash, kind, hello/handshake/origin,2400/3000 deadlines
and false training-flag checks; each rejected claim is retained without a root
receipt or collection dispatch. Original source-derived root and claim-emission
statements use a fake UID and fake clocks. Process/threads/fds/timers/compute/
proof/GPU and collection ports are explicit fakes. Collection is a stub port,
not a new full server/codec/worker lifecycle or target replay qualification.
Real work is private fixture bytes/JSON only. No original packet extraction,
Torch/NumPy/array value decode/model/gradient/profiler/GPU work, prior passing
case, successful tiny gradient or closed scientific cohort was run.

Standalone stdlib saved audit v1 passes first-run on45248 prior hashes,49778
prior fixture entries and207 new entries. It verifies the consumed attempt and
error joins, all five old file hashes, four original file identity/stat fields,
the previously recorded authorization mode/uid/nlink/mtime/size/hash, source
prefix/scope incompatibility, three full-byte inverses, first-run XML and fake
fixture records. The authorization's historical inode was not recorded, so no
historical inode equality is claimed for it. The package directory stays
inode44893436/device16777233/uid/gid/mode; tracked nlink6->7->9/mtime changes are
limited to the approved AUTH/ATTEMPT/session lifecycle. All other fixture fields
stay exact. Old component/candidate exceptions remain separate; old FIFO lstat
only, never opened. Device observation remains conditional, never remapped.
The saved auditor calls no project/test/process/timer/signal/network/proof/GPU
or scientific code and decodes/recalculates no array values.

[Machine-readable review](2026-11-23-pgd-workload-handoff-v1-target-review.json).
Raw stage: pgd-workload-handoff-v1-20261123/target-review-20261123.
Seal: sealed-workload-target-review-v1.json. Preserve every version, failure,
fixture and old grant/claim. The user stages/commits/pushes.

Next is a separate LOCAL binding/package adapter using the repaired source,
correct imports/local pins and workload-prefix scope. It must pass positive
client/server/collection joins with explicit fake ports before a new exact scope
or approval request. No new package/grant/approval request or launch command is
created here. The old attempt stays failed, remote progress unknown, grants
consumed. Mechanism stays closed; target whole-unit/replay/performance, hardware/
recovery/economics/calibration/data/protocol/adoption and training stay gated.
Policies/study/financial caps/holds remain unchanged; old wallet snapshots remain
historical. Mac Terminal need not remain open for the completed failed command.
