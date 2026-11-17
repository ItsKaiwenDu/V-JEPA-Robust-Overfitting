# Repaired PGD launcher and client handoff

This handoff packages the locally reviewed full-catalogue Store and owned-exit
monitor repairs. The Store closes its owned write fd before recording final
metadata, as verified by the live XFS byte probe. The monitor uses one at-most
0.1-second confirmation wait on an already-owned child inside its original phase
and work clocks. It still rejects unconfirmed exits; ownership/RSS/final guards
and cleanup remain strict. Neither repair has completed a new full-model run.

## Concrete final package

Only use:
`runs/pretrained/diagnostic/pgd-close-handoff-v1-20261117/handoff-parent/linux10609-final-unarmed`.
It currently contains CONTROL-CANDIDATE.json, SOURCE-PAYLOAD.json, PACKAGE.json
and known-hosts. No authorization, attempt, connection or target work exists.

- Outer control SHA256:
  `9c6670f7495fe9cec5faf219d0ef7664b2b49bc89a197f23ad4315f74a7a112b`.
- Supervised control SHA256:
  `3b9bf9649f491447b99f185ecd1da93ba1684ee9117088b12ff5c2c7cd6d6a9a`.
- Outer source: **33,359 bytes**, SHA256
  `ab1b588cca7e3d76f34c54a802bd6681db80416c32570c63532198c0d760f684`.
- 69 uploaded source files: **878,631 bytes**, encoded envelope **1,174,629 bytes**.
- Original packet: **17,192,095 bytes**, one original training clip, prior clean
  reference derivatives and frozen source/metadata. No held-out pixels.
- 34 pinned local sources cover the new outer/client/CLI/preparer, reviewed
  integration/repair recipes/readers and all retained cold/reference dependencies.

Destination is `kdu@linux10609.dc.engr.scu.edu`, one fresh private `/var/tmp` root
bound to the new control hash. The public host fingerprint remains
`SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ`. The control carries the
unchanged historical budget snapshot/reference bytes; this is not a fresh
account-balance assertion. No private SSH key or model checkpoint is uploaded.
Existing university environment/checkpoint bindings remain required.

## Exact changes and retained behavior

The 69-source payload and supervised control are byte-identical to the preceding
reviewed repaired candidate. Exactly five uploaded files differ from the consumed
10609 attempt: Store, monitor and dependent hashes in coordinator/scratch/bootstrap.
The other 64 and all scientific code, values/seeds, numerical/backend bounds,
allocation-proof and collection implementations remain unchanged. The catalogue
remains 172 assets and 419 result files; setup creates the original 26 directories
and leaves worker result namespaces for their owners.

The new outer source changes only reviewed policy/bootstrap literals, new
control/authorization kind strings, local source coverage, the local pure-source
verifier paths and the package-preparer import. Its full remote_session, alarms,
claim, setup, collection, directory plan and all other runtime functions remain
AST-exact. The original preclaim-refusal correction remains exact. The client
changes only its three outer/bootstrap/collection imports; the CLI changes only
its client import. Both have exact byte inverses to frozen source. Old source,
controls, grants, claims, results and fixtures are preserved.

## Unchanged execution scope

One attended connection, **40 minutes work** including original setup and full
**2.5 GiB allocation/write/fsync/read-back proof**, eight matched original/candidate
input gradients and two component VJPs/one inner VJP. Original role caps remain:
proof 60s, numerical 1,680s, components 120s each, and all setup/finalization clocks.
Collection uses the **same SSH**, up to **10 minutes**, within **50 minutes total
including login**. No numerical objective, warmup or completed-cohort repeat is
added by the repairs.

Caps stay: setup **3 GiB**, capture **2.25 GiB**, response **3 GiB**; raw **1,792 MiB**,
metadata **64 MiB including 8 MiB trace**, GPU allocator **7 GiB**, component RSS
**4 GiB**. Original host/UUID/driver/foreign-compute/private-XFS/storage/inode/
process-identity/backend/numerical/clock guards stay. App/RSS/free-space samples
are not instantaneous kernel quotas or continuous exclusivity reservations.
A foreign compute process refuses; it must be left alone.

No retries, separate retrieval, old cleanup, installations, private-key or held-out
pixel upload, old claim resets, extra objectives or paid compute. The original
proof may remove its own proof extent through the unchanged protocol. Failure
stops later work and retains partial evidence. The repaired monitor is only
locally fake-process qualified; live full-model progress remains unobserved.

## Checks and next action

Twenty distinct new focused checks pass in one run, without failure, skips,
passing repeats or historical/scientific cohort reruns. Actual source/metadata
reads, hashes, codecs, catalogue, source cache, directory planning, journal/claim
and full collection-header validation run locally. Packet contents/hash, reserve,
GPU outputs, process/PID APIs, worker, collection data and remote phase/alarm clocks
are explicit fakes. The emergency fixture has zero bytes; no proof executes.
One unarmed client call reads a monotonic timestamp before refusing, but starts
no timer or claim. No scientific library, array, model, gradient, real child,
process signal, timer, network, SSH, GPU query or paid compute runs.

The client preflight test uses an explicit packet-byte-hash stub. After user
approval, real local preflight must hash the actual 17,192,095-byte packet without
unpickling/decoding pixels or loading a checkpoint. The final package is unarmed.

All previous university grants are consumed. One new exact scoped approval is
required before arming or target work. Only then create AUTHORIZATION.json at
the exact final path, verify the saved seal and source pins and perform real local
client.check_local. Record/seal arming and give one attended Mac command using
scripts/run_predictor_pgd_close_v1.py plus that exact final absolute path. No
agent SSH and no repeated manual GPU/key paste. Wait until VPN is fully connected,
enter the university password promptly and keep the Mac/Terminal awake. A refusal
consumes this one attempt. Never rearm, reset, rerun or retry collection. After
Done/error inspect only saved local output first.

Original scientific/training/performance/cost admission stays false until
independent result review. Financial caps, holds and reconciliation stay unchanged.
The user stages, commits and pushes.
