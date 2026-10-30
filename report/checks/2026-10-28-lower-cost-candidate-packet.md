# Lower-cost EOT8 benchmark packet prepared — October 28

The corrected **v4 packet** is restored and verified locally. All **10 packaging
tests** and **77 original extracted CPU checks** pass, with no failures, errors
or skips. An independent audit reads and hashes all 76 source archive members,
verifies unchanged input archives and confirms the launch template is unarmed.
Scaffold checks pass. No cloud connection, upload, GPU job or pretrained model
load occurs in this preparation; CPU model fixtures use synthetic inputs.

The original 75 source files and all 61 scientific input pins are byte-identical
to the completed SE benchmark. The sole source addition is the unchanged original
`scripts/check_workflow.py`, previously omitted by the packet allowlist. All 24
workflow provenance dependencies are now present from the start. The new builder
uses the frozen explicit list, avoiding accidental inclusion of newly added
budget/packaging tests whose dependencies are outside the scientific packet.
The original 100 training clips, Linux wheels, upstream bundle and restorer are
copied byte-for-byte. No private SSH keys or confirmation/test pixel files are added.

The original 13 benchmark phases, source binding, attack settings and numerical
tolerances remain unchanged. Future CPU checks must explicitly place temporary
files on the actual `/workspace` session volume. This prevents repeating the
30 GB container `/tmp` failure while retaining the original 55 GiB free-space
guard. Existing manifests, failed setup evidence and earlier packets remain.

The actual local restore pins upstream commit
`204698b45b3712590f06245fbfba32d3be539812`. The Mac emits its known case-collision
warning for unrelated upstream V-JEPA2.1 vitG/vitg configuration names; the
required scientific source tree is clean and all restored CPU checks pass.
This is local preparation, not qualification of another CUDA device.

The packet remains under
`runs/pretrained/diagnostic/candidate-eot8-handoff-v4-20261028/`.
Its manifest SHA256 is
`bf2b124b9fff819cc9f1b3acbc095c208c8a4df1293cda24267b802edd5d0b8a`.
A separate setup-requirements record specifies the corrected fixture location
and unchanged resource/runtime gates. It is not an executable launcher.
The launch template has `authorized:false`; GPU, quote, allowance, ledger,
deployment time and deadline are null. No previous session grant is reused.

## Tomorrow's pending checks

The user reports billing is still updating and defers billing/quotes until
October 29. Retain the current $1.75 session reservation, historical $2.00 hold
and unresolved invoice gap. Conservative encumbrance stays $9.22, leaving
$0.78 unreserved within the single $10 pilot envelope. No hold is created or
released; the $200 project / $160 compute caps and 20-class design remain.

Check the final `vjepa-eot8-benchmark-07` charge, current balance and total debits.
Then obtain fresh availability, **Total/hour** and system RAM for one A5000 or
RTX 3090 with 30 GB container and 80 GB `/workspace` storage. At least **40 GiB
usable system RAM** (about 43 GB) and 22 GiB visible VRAM are required. Old cheap
quotes are not assumed available, and the SE throughput is not transferred to
another device. No deployment or queue subscription is requested now.

A one-time reminder is saved in this chat for **October 29 at 10:00 AM Pacific**.
It only asks for those checks; it cannot authorize paid work or release holds.
After the checks arrive, review the actual remaining allowance and prepare/test
a concrete bounded operator before proposing deployment. Only measured hardware
and economics can support subsequent full calibration.

[Structured preparation and fingerprints](2026-10-28-lower-cost-candidate-packet.json).
Raw logs, JUnit results, source snapshots and independent audit are saved under
`runs/pretrained/diagnostic/candidate-lower-cost-packet-validation-20261028/`.
