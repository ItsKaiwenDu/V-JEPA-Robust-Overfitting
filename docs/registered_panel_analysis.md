# Analyze the originally registered diagnostic panel

The complete20-case analysis is [verified](../report/checks/2026-10-28-complete-panel-analysis.md).
It uses the frozen four contrasts, averages the two technical banks within
each clip, and keeps the three scoring banks separate. Ten clips are the
descriptive analysis units. All original identities remain unchanged; the
mixed execution wrappers are handled by the read-only loader without changing
the old journal or reclassifying failed attempts.

Increasing steps and attack-mask draws both raises mean independent-scoring
loss. Four→eight draws improves it3.54–3.95%at20steps and3.84–4.15%at50steps;
20→50steps improves it1.53–2.03%at4draws and1.94–2.49%at8draws. These ranges
describe three separate bank means. No configuration or convergence pass is
selected automatically; step-size calibration and research admission remain.

The input manifest pins the actual registered20-case receipt, all original
case origins, local saved numerical directories, archive sidecars and review
markers. The loader rechecks evidence on CPU before calling the original pure
aggregation function. Source layout is explicit: seven per-case runner files
and the shared frozen scientific dependencies. A modified input, incomplete
collection, source/receipt drift or mismatched technical pair refuses analysis.

To reproduce into a **new output directory** from the repository root:

```bash
.venv/bin/python -B scripts/analyze_registered_panel.py \
  --inputs report/checks/2026-10-28-complete-panel-inputs.json \
  --inputs-sha256 4a32d77c46bed1b28cc2ba00e10e5400fadf2a6f4c531a6f96da1f6a5afdb578 \
  --output runs/pretrained/diagnostic/complete-panel-analysis-reproduction-UNIQUE
```

The command needs the existing saved evidence and PyTorch, starts no model or
GPU worker, opens no held-out data and refuses to overwrite an existing output.
Outputs are JSON with raw paired scores, all frozen contrasts,739 original
fingerprints and source provenance, plus Markdown tables. The independent
exact-rational verification script and results are saved with the original
analysis under its validation directory.

Forty-one focused tests pass with zero skips. This panel describes attacks at
the released checkpoint on ten training clips; robust-overfitting conclusions
require the later training trajectories and locked evaluation. Next prepare
the explicit research attack/mask/calibration decision, review numerical and
GPU recovery qualification, and reconcile the measured study budget.
