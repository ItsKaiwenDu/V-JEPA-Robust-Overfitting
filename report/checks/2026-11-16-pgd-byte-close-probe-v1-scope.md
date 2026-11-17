# Proposed single byte-publication probe on linux10609

Prepared, locally tested and unarmed. See the complete
[operator plan](../../docs/pgd_byte_close_probe_v1.md).

One SSH to `kdu@linux10609.dc.engr.scu.edu` carries the verified 32,245-byte
standard-library program and public nonce/control/program hashes in its command.
It creates a fresh private `/var/tmp/vjepa-byte-close-v1-<uid>-<control hash>` root
and runs the frozen writer then the candidate on two opaque `0xA5` files of
12,583,040 bytes each. It records before/after owned-close metadata, full-row
checks and real read-back byte hashes. The one-file byte inventory is explicit;
this is not a scientific catalogue or model run. Unexpected failures stop later
work; no synthetic allocation changes are made on the host.

Bounds: 60 seconds work plus five finalization; independent 65-second GNU timeout
plus two-second kill grace; 180-second client window including login and same-SSH
response. Root app checks: 32 MiB logical, 64 MiB allocated, five names/nine dirs,
64 KiB metadata. Distinct data total 25,166,080 bytes; original atomic publication
can briefly have two names for one inode. These checks are not kernel quotas.
Response 64 KiB, SSH stderr 16 KiB. Preserve new data, status, and local partials.
No retries, polling, separate retrieval, old-file cleanup, installations, arrays,
model, gradients, storage proof, GPU query, existing-job signals, paid compute,
training/reference packet, financial ledger or private-key upload.

Host/runtime/XFS/private scratch guards run directly. No manual GPU status paste.
Previous grants are consumed and do not cover this new connection/write scope.
One new explicit exact scoped approval is required before the final handoff is
armed or SSH starts; a refusal consumes that one grant. No original model claim,
numerical bound or financial reservation is reset.

Program SHA256 `42678bfc978df91469522c9c2750480df73882548c0e150748be8d82a9c8cf93`.
Control SHA256 `6dc695e980485d3ba7f57e4e07e4f28f947d1d7a113227604cacae9c54d235e7`.
Pinned public host fingerprint
`SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ`.
No training admission follows from this probe's outcome. The user stages/commits/pushes.
