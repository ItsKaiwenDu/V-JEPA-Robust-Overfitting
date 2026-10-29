# Candidate hardware quote review — October 28

Recommend **RTX PRO 4500 SE for the next bounded benchmark**, conditional on
availability and host checks. The user confirms **$0.74/hour total** with a 30 GB
container and 80 GB volume. Their screenshot shows 32 GB VRAM and 94 GB system RAM.
[NVIDIA specifies a Blackwell GPU with 32 GB GDDR7](https://www.nvidia.com/en-us/data-center/rtx-pro-4500-blackwell-server-edition/).
This is a capacity-appropriate unmeasured candidate; no speedup or recovery pass
is inferred from the specification.

A40 at the user's **$0.61/hour total** can run the diagnostic, subject to actual
usable RAM (43 advertised GB is close to the 40 GiB gate). Its previous four-draw
study screen was already $221.56–224.01 against the unchanged $200 cap. New EOT8
work is unmeasured; neither GPU is admitted for the full study. At these quotes,
the SE needs 74/61 times A40 throughput to break even on cost for equal work.

Two billed hours cost **$1.48 on SE versus $1.22 on A40**. Propose an inclusive
**$1.75 / two-hour session envelope**, starting at deployment and covering setup,
idle and cleanup. The runner cannot terminate cloud billing. The $8.90 balance
is sufficient for this proposal; recharging is unnecessary for it and does not
raise the $200 project/$160 planned-compute caps.

Keep the historical unresolved invoice gap and old inclusive hold. Conservative
prior encumbrance stays $5.47 + $2.00 = $7.47, which already contains the latest
wallet-implied $6.10 usage; do not add wallet movement again. A proposed $1.75
session would bring encumbrance to $9.22 and leave $0.78 of the single $10 pilot
reserve. The budget envelope is reconciled conservatively; final account invoices
are not declared reconciled. No old hold is released and no new hold is created.

The current review draft remains **authorized:false**, with null deployment,
deadline and GPU identity. Its financial gate was checked with synthetic clocks
only in memory. No GPU, SSH connection, study calibration or paid launch started.
Actual deployment approval and host identity still need to be established.
