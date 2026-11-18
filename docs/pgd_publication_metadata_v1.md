This unarmed operator retrieves the missing parent failure record after the
linux10614 scanner-repair run failed during work and its download timed out.
Read report/checks/2026-11-17-pgd-publication-collection-timeout-v1.md and the
metadata-v1-scope.md for the saved facts, exact target/files and bounds.

The server program is scripts/pgd_publication_metadata_remote_v1.py. The reusable
local client is models/pgd_publication_metadata_v1.py; the thin CLI is
scripts/run_pgd_publication_metadata_v1.py. The exact plan and three source/five
incident pins are in configs/pgd_publication_metadata_v1.json and the unarmed
PLAN.json. The request includes public bindings only. The control file is hashed
and not returned; the session is returned only after matching the saved receipt's
SHA-256. Runtime audit hooks and owned descriptor checks enforce read-only access.

The one-use local claim and original three-minute clock cover the new connection.
Remote timeout is twenty seconds plus one-second kill grace. Source delivery uses
SSH stdin; no source file is written on the workstation. A new explicit scoped
approval must precede arming, and only the user runs the attended SSH helper.
This operator is separate from the consumed experiment; it never resumes work or
the large interrupted transfer. Old files, claims and limits remain unchanged.

The 31 new checks use inert metadata and fake target/process/selector/clock APIs.
Their logs, source revisions, JUnit and fixtures remain in the raw metadata stage.
They do not establish native execution, numerical success or current GPU status.
No remote work is authorized or executed by this preparation. No training or
performance qualification and no financial hold release is implied.
