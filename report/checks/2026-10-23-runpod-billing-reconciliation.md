# Runpod per-Pod billing reconciliation — October 23, 6:25 PM Pacific

The resource Details view supplies six Pod costs. Their displayed four-decimal
amounts sum to **$5.4652**, which rounds to the displayed **$5.47 total debits**.
Together with the latest **$9.53 wallet balance**, the cent-precision figures
reconcile to the user-confirmed **$15.00 initial account credit**.

| Pod | Shown cost |
|---|---:|
| vjepa-pilot-01 | $0.5978 |
| vjepa-workflow-pilot-02 | $1.6882 |
| vjepa-recovery-pilot-03 | $1.6444 |
| vjepa-cost-benchmark-04 | $0.3282 |
| vjepa-cost-benchmark-05 | $0.7499 |
| vjepa-cost-benchmark-05-migration | $0.4567 |

The latest A40 session's original and migrated Pods combine to **$1.2066**,
approximately $1.21, within its $2.80 inclusive reservation. The earlier A40
setup costs $0.3282; the recovery pilot costs $1.6444, each within its $3
reservation. All three internal planning reservations are now settled against
the observed charges. This bookkeeping neither refunds funds nor increases
the provider wallet. Earlier partial totals are superseded rather than added.
In particular, the workflow-02 Pod's shown full charge is $1.6882, not the
earlier provisional $0.603 report or a difference of overlapping daily totals.

Count the full $5.47 of consumed service credit conservatively against the
shared $10 diagnostic allowance, leaving **$4.53**. The provider wallet has
$9.53; wallet credit and remaining diagnostic allowance are separate quantities.
The $200 project/$160 planned compute caps remain unchanged.

Both original/migrated Pods remain explicitly user-confirmed terminated. The
missing deletion badge on the migration billing row does not establish an
active Pod. This reconciliation applies as of the supplied Details/balance
snapshots; exact stop timestamps and higher-precision invoice data are not
asserted. The hourly/daily Explorer can retain reporting and grouping delays.

Preserve historical reports, immutable source pins and frozen launch constants.
A future handoff must bind freshly reviewed budget metadata to the reconciled
ledger; no rental or research run is authorized by this accounting update.
The A40 screen still has six completed phases and an unfinished control phase.

[Structured ledger](2026-10-23-runpod-billing-reconciliation.json).
