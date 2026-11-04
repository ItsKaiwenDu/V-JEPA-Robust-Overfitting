# Target CPU scratch exception approved — November 4, 2026

The user explicitly approved one additional attended CPU-only target attempt on
linux10603, including the unchanged 19 MB packet/new supervisor upload to a fresh
private `/var/tmp` directory and the 2.5 GiB write/fsync/readback check. The original
600-second bound includes setup and the four fresh workers; each worker remains
capped at 90 seconds. Same-connection collection has a separate 600-second limit.
No automatic retry, control-host grant, old-claim reset or old-evidence deletion
is authorized.

The new `AUTHORIZATION.json` binds target control
`8a42039c95a2592becfbef19f59afdaa5692b8e2125f08f6ede864f19bfe4a78`.
Its own SHA256 is
`1c1c09b6e70adb096229191b9a78aee67bee16076fa93dffe6eed39414df1bdd`.
The unchanged client accepts its scope, source/packet/prior-claim hashes and Mac
capacity during local validation. No `ATTEMPT.json`, SSH connection, allocation
proof or reference calculation has started. Linux10609 remains unarmed and also
requires independent review of this new target run.

Next run the new target `run-attended.sh` once from a local Mac Terminal, enter
the university password there, and keep the foreground session/Mac awake through
collection. Independently review the copied evidence before proceeding. The 60
operational checks are unchanged; no numerical or full-size proof is inferred.
Original scientific settings, failures, financial caps and holds stay unchanged.
See [the tested execution protocol](../../docs/cpu_oracle_scratch_operator.md).
