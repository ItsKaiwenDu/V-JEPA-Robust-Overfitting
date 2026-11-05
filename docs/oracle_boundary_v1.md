# Original CPU-return boundary observer v1

The [preparation review](../report/checks/2026-11-04-oracle-boundary-v1-preparation.md)
verifies a separate observer for the missing historical computed CPU values.
The original worker, oracle, inputs and bookkeeping expression remain unchanged.
The handoff is **unarmed**: no new university authorization or attempt claim
exists, and the prepared helper refuses before SSH.

## What one proposed attempt does

On linux10603, call the original `encoder-1 / math-label / replay-0` worker once.
Its original setup configures execution, checks the GPU, initializes CUDA and
sets the original 4 GiB allocator fraction cap before loading the fixed fixture
and calling the CPU FP64 oracle. The math label is metadata here; attention
backend selection is downstream of this capture's stop boundary.

A standard Python profile hook observes the unchanged oracle's return. It saves
the actual CPU FP64 `output`, `dq`, `dk` and `dv` to a bounded NPZ before the
original gate can reject them. No new oracle calculation, conversion fallback,
input change, tolerance change or replacement oracle is allowed. Loaded worker,
oracle, transfer and iteration function code is checked against the packet.

The original Torch gate runs at `rtol=1e-12, atol=1e-14`. If it refuses, its exact
error and decisions remain. If it passes, the hook intercepts the first original
`clone_layout` call before its body and raises an intentional `BoundaryReached`
stop. Thus the original GPU worker remains `complete=false, passed=false` in
both outcomes. A separate capture receipt distinguishes a captured gate pass,
a captured original refusal, and a technical failure. A successful transfer
helper means files were verified; it does not qualify the GPU worker.

Detailed library/thread introspection occurs after the original gate or
technical stop. The hook restores prior observation and alarm state without
resetting elapsed allowance. Existing trace/profile hooks are refused. Source
reads, the profile hook, new capture I/O and imports still differ from the old
program: this is an observed capture, not a proven identical historical replay
or a single-variable causal test. Post-gate context is not a measurement of
every library's state before its first calculation.

The new outer operator never invokes the original parent/task loop. Its inner
`original-schema-plan.json` retains the old validator's 1200/120-second fields
only for compatibility and explicitly binds the new 600-second outer control.
It is created only inside a separately authorized attempt; it grants no old
parent run. The old efficient-arm comparison receipt is not used: this scope
ends before that downstream comparison. This is not a complete historical
math-worker replay.

## Exact payload and bounds

The verifier needs the **full unchanged fixed-input packet**, containing all
three synthetic fixture descriptors. This is the 85,622,679-byte archive
`51c4e4b71ec292afc65ee3bd53c5556c6b538056d607244aef4945ed5fcded9a`:
139 files / 90,901,652 logical unpacked bytes. It is not the 19 MB CPU-only
packet. Nine overlay files total 125,527 bytes. The new plan carries source and
prior-evidence hashes, not prior raw captures or the financial ledger. No
weights, pixels, dependency installs, private keys or passwords are uploaded.

| Bound | Proposed scope |
| --- | --- |
| Host / concurrent scientific workers | linux10603 only / one |
| New attempts / original oracle calls | one / one; no automatic retry |
| Work | 600 seconds, including setup, proof and capture |
| Original worker | at most 120 seconds, within the original outer clock |
| Proof | full 2.5 GiB write, fsync and readback; at most 60 seconds |
| Local scratch | fresh private mode-0700 directory under `/var/tmp` |
| Initial free space / runtime floor | 8 GiB / 1 GiB, with inode checks |
| Observed setup / capture ceilings | 3 GiB / 2.25 GiB |
| Input / unpacked code category | 128 MiB each, to fit the unchanged packet |
| Raw captures / individual tensor file | 256 MiB / 32 MiB |
| Collection | same SSH connection; at most 600 seconds / 3 GiB |
| CUDA domain | original setup only; stop before input transfer and SDPA |
| Paid compute | none |

Storage accounting observes logical bytes, allocated blocks and capacity;
individual process file-size limits also apply. These checks are not a kernel
aggregate quota, exclusive disk reservation or guarantee of future capacity.
The GPU identity/driver and foreign-compute guard run before work and are
sampled during capture at five-second intervals. Sampling does not prove
continuous exclusive use. Foreign processes are left alone; only this
operator's supervised children can be stopped when its bounds fail.

The public key is pinned to linux10603 and the independently supplied
`SHA256:ARgVRWOzRLA1LmAANxD6Ya17EIIo2sa4ZbHvkeqzgKQ` fingerprint. The bounded
369,487-byte bootstrap travels over stdin, keeping the SSH command at 1,024
bytes. Upload and collection use one attended connection. The Mac helper keeps
the machine awake; keep its Terminal and lid open and answer the local SSH
password prompt promptly. Any failure consumes the launch: preserve and review
partial files before using only the separate collection helper. Collection
cannot restart a worker or regenerate evidence. Remote scratch is retained.

## Validation and remaining gate

All 79 local checks pass, with zero failures, errors or skips. Cold extracted
tiny fixtures run the actual original worker/oracle on the Mac. A positive
fixture captures the passing gate and stops before transfer. A separately
identified negative fixture changes only its own canonical reference, captures
the same computed return and retains the original refusal. Tests also cover
write failure, existing-hook refusal, function replacement, clock restoration,
full-package cold import with dispatch blocked, strict GPU guard, failed proof,
atomic writer pairs, corrupt transport, one-stream upload, collection-only
behavior, consumed grants and unarmed refusal. No full allocation proof or
CUDA initialization is performed by these local checks.

An independent saved-evidence review, with process/network creation denied,
reverifies all 139 original archive members, 946 preserved source/evidence
bindings and 310 rehearsal files with exact modification times. Eight saved
gate checks reproduce the original decisions. The positive and negative
fixtures have identical actual inputs and computed returns; only the declared
negative canonical reference differs. The local reader is arm64/PyTorch 2.14.1;
this does not qualify the university's PyTorch 2.8/CUDA 12.8 path.

The prepared control is
`eb5d8be9edfce5a6a9cb0b7e6fadd28f6ce8d34b961cd58a535b552d8f1db63e`.
The raw handoff is under
`runs/pretrained/diagnostic/oracle-boundary-v1-20261104/linux10603/`.
Both `AUTHORIZATION.json` and `ATTEMPT.json` are absent there. Do not run an old
consumed helper or create a claim by hand.

Before this new attempt, [AGENTS.md](../AGENTS.md) requires “separate
reviewedbound/authorization,noreset.” The previous CPU-only grants are consumed
and do not cover CUDA setup. Seek one explicit exception against this concrete
host/payload/domain/bound, then arm only the new helper. Its live preflight must
recheck host, GPU, foreign processes and storage. Linux10609 stays idle.

The old discrepancy, GPU qualification, optimization adoption and research
admission remain unresolved. The $200 project / $160 compute / single $10
diagnostic caps, existing holds and pending reconciliation remain unchanged.
