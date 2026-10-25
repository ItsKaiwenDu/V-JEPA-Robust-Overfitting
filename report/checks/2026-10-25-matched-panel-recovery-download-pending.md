# Single-case recovery finished remotely; evidence download pending

The saved attended transcript contains the unchanged operator's successful
case read-back and single-case completion messages, followed by launcher exit
zero. The recorded attended SSH exit is also zero. This reports completion
of clip-01-bank-0 attempt-02 under the existing grant; it must not be rerun.
The remaining queue was not started.

The subsequent SCP copy reported a closed connection. The local download
directory is empty. This is a transfer failure after the completed run, not
an observed GPU failure. Its network/server/SFTP cause remains unknown. There
are now three remotely reported completed cases, of which two remain locally
independently verified until the new archive is copied and audited.

A new download-only helper uses one ordinary SSH stream rather than SFTP.
It reads the existing case_verified status, case export/member manifest,
original grant records, current journal and exactly the reported launcher
logs. It writes no remote files and starts no tests, model or grant. A bounded
regular-file transport includes every member checksum and a whole-stream
checksum. Local verification refuses unexpected names, paths, symlinks,
changed or truncated bytes, and invalid case member inventories before
recording successful byte checks. Scientific review remains separately pending.
Eight byte-fixture checks and one test of the actual streaming writer pass;
these use ordinary text bytes, no neural tensors or generated model receipts.
Shell/Python syntax checks pass. The helper has not yet been executed.

[Machine-readable observation](2026-10-25-matched-panel-recovery-download-pending.json).
