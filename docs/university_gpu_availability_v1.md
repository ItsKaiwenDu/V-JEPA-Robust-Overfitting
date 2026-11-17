# Bounded read-only university GPU observation

This helper checks availability without spending another model-attempt grant. It
has its own read-only authorization and one local READ-CONNECTION claim. It cannot
launch a model/proof, create or reset a model ATTEMPT, upload a data packet or budget
ledger, change server files/settings or signal an existing job. All old model
sources, controls, consumed grants, evidence and financial bounds stay unchanged.

The single destination is kdu@linux10603.dc.engr.scu.edu, with its already verified
ED25519 public host key. The new connection still needs its separate exact approval.
The final source-only candidate contains CONTROL.json, QUERY.py, known-hosts and
PACKAGE.json; no READ-AUTHORIZATION or READ-CONNECTION exists. Earlier source-only
candidate/config snapshots remain unarmed and must not be substituted.

The 10,339-byte standard-library program is passed in the SSH command, plus three
public binding values: request nonce, control SHA and program SHA. No server file
is installed. Python -I -S -B skips user/site startup and bytecode writes. Only host/
system/Python identity and two fixed nvidia-smi queries are read: GPU name/UUID/
total and used memory/driver, then compute PID/process name/used memory. A different
host/runtime refuses before a query. No proc contents, user datasets, tensor,
archive, cache, proof, source repository, private key or financial ledger is read.

There is one SSH invocation with strict pinned known-hosts, no forwarded agent,
no forwarded services or local SSH command, no user SSH config, password or keyboard-
interactive authentication only, one password prompt and no connection retry.
No model authorization is accepted as a read-only grant. Copied helpers, changed
source/host/program/limits, expanded grants and repeated reads refuse locally.
The user enters the university password in their Mac Terminal, never in chat.

The client has a 180-second window covering local preflight, login, SSH capture and
response validation. Nonblocking bounded pipes and wait use that original deadline.
Saving small local evidence follows without another connection. Separately, GNU
timeout gives the remote command group 20 seconds plus one second of kill grace.
Each nvidia-smi query has an eight-second deadline. The remote bound is independent
of the client clock, including if SSH disconnects; no clock epochs are equated.
On timeout/overflow the helper may stop only its own unreaped query or SSH child;
GNU timeout may stop its own read-only command group. It never chooses or signals
any existing GPU job. These are application/command limits, not an instantaneous
kernel-stop guarantee for uninterruptible OS activity.

Per-query stdout/stderr bounds are 8 KiB/2 KiB. Remote response is at most 64 KiB;
local SSH stderr is at most 16 KiB. Partial stdout/stderr survive timeout, malformed
or oversized output. All local records are exclusive private files; no failed file
is replaced and no automatic retry occurs. The read grant is consumed before the
single SSH starts, even if login or a query fails. This does not consume or restore
a model attempt.

A complete valid response is classified from its saved CSV, not a remote readiness
assertion. The exact expected GPU/UUID/driver/8 GiB total must match. An empty compute
list and at most 512 MiB used yields idle_observed; any compute process, including
NX even with low memory use, or greater memory use yields busy_observed. Malformed,
failed or incomplete data yields unknown; identity differences refuse. Queries are
serial snapshots, not atomic or a GPU reservation. Neither busy nor idle starts,
queues or authorizes any later work. Future separately approved model work must
check live guards again.

Twenty-nine distinct new checks have passing latest results over 32 executions
(28/2/2), with no failures/errors/skips. Three affected passing checks repeated:
the command check after adding -S and two scope/response checks after clarifying
the client-only time-window label. A new in-flight pipe-timeout case was added.
The remaining passing checks and all historical tests/scientific cohorts did not
repeat. All query/SSH/process/selector/clock bindings in checks are inert; only
ordinary local metadata IO and standard-library definitions execute. Parsing uses
v4's saved GPU query text and synthetic variations. No actual nvidia-smi, SSH,
process, timer, GPU, proof, model, tensor, gradient or paid work runs locally.

After complete source review, request one scoped read-only connection. On approval,
arm only the exact final handoff and run its single Mac CLI. Reply Done for local
review of the result. A busy read should lead to waiting or another separately
approved availability observation, not an immediate model-run grant. An idle read
may support preparing a fresh model handoff, which still needs its own exact
approval; old model grants remain consumed. Do not ask for repeated manual
hostname/nvidia-smi pastes or silently switch to another workstation.
