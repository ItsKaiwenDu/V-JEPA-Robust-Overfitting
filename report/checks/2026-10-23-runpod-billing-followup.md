# Runpod billing follow-up — October 23, 6:20–6:23 PM Pacific

The Summary screenshots total **$5.338** in daily UTC buckets: $5.195 GPU cloud
and $0.143 storage. The twelve displayed hourly rows sum to $5.339: $5.196 GPU
and $0.143 storage. The $0.001 difference is consistent with separately rounded
display values. The October 23 UTC total combines $1.644 from October 22 local
evening and $1.408 from October 23 local hours before 5 PM; these are not charges
to add again to the daily aggregate.

The current wallet screenshot shows **$9.53**, implying **$5.47 cumulative
credit consumed** from $15 initial credit, assuming no later additions/refunds.
This replaces the earlier provisional $5.44 working figure; it is not another
charge to add. The difference from the displayed daily Summary is $0.132,
which remains unassigned rather than attributed to a particular Pod.

The latest visible hourly bucket starts at 4 PM Pacific. The Explorer still
discloses a one-hour delay. The user demonstrates that October 24 is disabled
in the calendar; the earlier suggestion to select it is withdrawn. The exact
cause of the missing post-5 PM bucket is unconfirmed, so do not claim that a
specific frontend timezone bug has been proved.

[Runpod's billing documentation](https://docs.runpod.io/accounts-billing/billing)
describes ongoing credit deductions with billing processing every five minutes.
The wallet is the current practical aggregate reference, without guaranteeing
zero processing delay or final per-Pod attribution. The configured
[`runpodctl user` command](https://docs.runpod.io/runpodctl/reference/runpodctl-user)
also exposes balance and current hourly spend. The
[billing history API](https://docs.runpod.io/api-reference-v2/billing/get-pod-billing-history)
accepts explicit RFC3339 ranges but does not document an instantaneous-billing
guarantee. No account API request or CLI installation was performed.

Both Pods remain user-confirmed terminated. Keep detailed charges and existing
$3/$3/$2.80 reservations pending; no hold is released or rental authorized.
No more screenshots are needed for offline planning now. Later full billing
reconciliation can follow once the remaining buckets become accessible.

[Structured observations](2026-10-23-runpod-billing-followup.json).
