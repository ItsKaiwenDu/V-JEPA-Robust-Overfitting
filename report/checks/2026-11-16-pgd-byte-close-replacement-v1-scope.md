# Proposed single replacement byte probe on linux10609

The previous SSH failed hostname resolution before connecting; the second CLI
invocation was blocked locally. Its one-use claim stays consumed. The user
reports a possible race with VPN activation. Wait for VPN access before launch.

The [replacement plan](../../docs/pgd_byte_close_replacement_v1.md) is prepared and
unarmed. It uses the unchanged **32,245-byte stdlib program**, seven source pins,
public host key and limits from the reviewed original probe. Only the local
handoff path/control hash changes. No new implementation or test/cohort reruns.

One pinned SSH to `kdu@linux10609.dc.engr.scu.edu` sends the program and public
nonce/control/program hashes. A new private exclusive `/var/tmp` root holds two
inert `0xA5` files of **12,583,040 bytes** each. Frozen then candidate; actual
before/after owned-close metadata, full-row checks and read-back byte hashes.
Unexpected failure stops later work; no target metadata substitutions.

Unchanged limits: **60s work + 5s finalization**, independent **65s timeout + 2s
kill grace**, **180s client including login and same-SSH response**. App caps:
**32 MiB logical / 64 MiB allocated**, five names/nine dirs, **64 KiB metadata**;
response **64 KiB**, SSH stderr **16 KiB**. Preserve new data and partial output.
Distinct data total 25,166,080 bytes; atomic publication may temporarily give one
inode two names. App checks are not kernel quotas or atomic reservations.

No retries, polling, separate retrieval, old cleanup, installation, model, arrays,
storage proof, GPU query, existing-job signals, paid compute, training/reference
packet, financial ledger or private-key upload. No manual GPU check needed.
One new exact scoped approval is required before arming or SSH; the previous
grant is consumed and cannot be reset. A refusal consumes the new grant too.
No scientific/training/performance/cost admission or financial-state change.

New control `8322d6e072bcf4d4929dd8f30bd87025305533e32efbcbcde252ac3bd5507540`.
Program `42678bfc978df91469522c9c2750480df73882548c0e150748be8d82a9c8cf93`.
Host public fingerprint `SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ`.
