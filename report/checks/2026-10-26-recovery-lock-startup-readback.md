# Recovery startup readback verified — October 26, 2026

The read-only capture verifies 31 regular files and a 256,000-byte transport
with SHA256 `ceb8d98eb7e0fe09c1aece046c794702fe77628c7d8990496bd987d5882cf746`.
All 139 campus tests passed with zero failures/errors/skips on linux10603;
source and launcher bytes match the original deployed packet. All 56 retained
coordinator file hashes match. The grant and attempt02 were both absent.

The failed preflight ran 35 seconds (6:35:18–6:35:53 PM Pacific). Charge
36 seconds with timestamp rounding margin. The same approved grant has
4,764 seconds (79m24s) left. No numerical worker or new case was created.
The original STOP and failed case remain intact. The 6:54:13 PM Pacific
capture snapshot showed no CUDA compute process; runtime checks remain required.

The [structured audit](2026-10-26-recovery-lock-startup-readback.json) records
all pins, eligibility and time accounting. The [separate corrected handoff](2026-10-26-same-grant-lockfix-preparation.md)
uses the unchanged authorization and the remaining limit.
