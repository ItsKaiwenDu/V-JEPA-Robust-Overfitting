# DNS failure; duplicate invocation blocked locally

The user invoked the same Mac command twice. The first saved a consumed attempt
and an SSH exit status of 255 after 0.06685 seconds. Its stdout is empty. The
106-byte stderr is: `ssh: Could not resolve hostname linux10609.dc.engr.scu.edu:
nodename nor servname provided, or not known`.

The first invocation never connected to the host, so no remote byte probe, files,
GPU query or model work started. “Evidence saved” was a collection message, not
a successful-comparison result. The second invocation raised the existing
“Attempt consumed” guard locally. Exact sealed source order places this check
before nonce generation, SSH command construction, claim creation and capture;
it could not start a second connection. The original claim and saved result stay.

Independent stdlib saved-file/hash/AST review passes all 14,429 prior hashes,
12,539 fixture entries and four joined local result/attempt/stdout/stderr records.
The old FIFO was lstat-only. Control, authorization, nonce, original 180-second
window, flags and saved wire hashes match. No tests, historical/scientific cohort,
project callback, network/DNS query, process, timer, SSH or paid work runs in review.

The user reports that VPN activation may have trailed launch by 1–2 seconds.
That is consistent with the error, but the VPN cause has not been independently
verified. Wait for VPN access before any new launch. The one-use grant remains
consumed even though the hostname lookup failed. Never rerun or reset this old
handoff, claim or collection. Prepare a fresh identical-code/bounds handoff
locally; obtain one new exact scoped approval before arming it. No manual GPU
status paste is needed for this byte-only check.

Outcome: **inconclusive**, not a numerical rejection, allocation observation or
candidate pass. Prior scientific/resource/ownership/time gates, financial caps,
holds and reconciliation remain unchanged. The user stages/commits/pushes.
