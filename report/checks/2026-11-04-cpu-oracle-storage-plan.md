# CPU reference storage-recovery plan — November 4, 2026

**Prepared offline; capacity survey and a runnable operator are pending.**
The [unarmed proposal](../../configs/cpu_oracle_storage_recovery_v1.json) keeps the
original four-worker CPU reference protocol while placing new capture and export
files on verified local disk scratch. Target linux10603 comes first, followed by
independent Mac review before any control-host launch. This design is not a new
CPU resource grant and implements no runtime guard yet.

The offline audit freshly rehashes the original 128 packet files, 61 scientific
sources, 33 upstream bindings and 13 consumed claim/plan records (including the
cleanup claim). Original packet/archive hashes match the previous review; no
scientific implementation, fixture, failed capture or consumed claim is changed.
Both previous full probes remain incomplete and the numerical discrepancy remains
unresolved. No new numerical or guard test is reported as passing.

The storage calculation accounts for 668,467,200 array-payload bytes for four
workers. The historical local raw-evidence/archive sum is 1,331,522,866 bytes,
excluding other files and filesystem overhead. The proposed capture categories
sum to 2,407,530,496 bytes beneath a 2.25 GiB ceiling; a separate 3 GiB setup
ceiling accommodates a full 2.5 GiB write/fsync/readback check. These are planned
limits, not an implemented bound or a reservation. A fresh 4 GiB free-space gate,
inode checks, Mac capacity and continuing I/O guards remain prerequisites.

The requested read-only survey checks filesystem type, free blocks/inodes, device
identity and resolved `/tmp`/`/var/tmp` paths on both workstations. It starts no
experiment. No SSH connection, upload, allocation test, CPU reference call, GPU
work, deletion, paid session or financial hold change occurs in this preparation.

Next review those observations, select suitable scratch paths or stop, implement
and test the guarded operator, then request the separate concrete resource
exception. No old launch or cleanup helper may be repeated. Full details and
limits are in [the recovery design](../../docs/cpu_oracle_storage_recovery.md).

Verification: `git diff --check` and the scaffold/YAML check pass. No new
numerical calculation, live storage allocation or operator guard test ran.
