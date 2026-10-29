# Research primary-attack candidate

The [offline decision packet](../report/checks/2026-10-28-research-candidate-review.md)
recommends qualifying PGD20/R2/EOT8/score4 primary evaluation, retaining matched
PGD50/R2/EOT8 with two scoring audits as the stronger control. This is a
**proposal, not an adopted research configuration**. A separate
[bounded diagnostic implementation](candidate_eot8_implementation.md) is now
tested; the original proposal and its identity are unchanged.

The complete fixed panel found positive4→8draw gains in all ten clips under
all three scoring banks. More steps still helps; no convergence pass or largest-
loss-arm choice is automatic. Existing study/proposal YAML and evaluator stay
unchanged. Training remains PGD5/step1/255 with the retained20-class,100-clip,
eight-run staged design and$200total/$160planned-compute caps.

The explicit candidate records the original ten training-monitor rows and200
new globally distinct mask draws. Calibration needs all30PGD20/EOT8 evaluations
for0.5/1/2per255, highest independent main mean then smaller exact tie. No step
is selected yet; audit banks do not select it. Calibration requires the chosen
research runtime; no four-draw or other-GPU artifact may be relabeled.

The old primary evaluator deliberately rejects eight draws. The new versioned
opt-in diagnostic path and bounded hardware benchmark packet are prepared.
Next review current hardware quotes and the reconciled launch allowance. The new primary and calibration timings and current quote are null.
At historical$0.61/hour, even the optimistic primary timing ceiling is149–152s
if calibration were free and all other A40 units were unchanged. This is a
necessary algebraic screen, not a zero-calibration estimate or study admission.
The actual three-step bundle, other affected units and unknown work need bounds.

Exact native two-update recovery passed on RTX5090. A40 exact resume was not
tested, and8GB university full AdamW training remains unqualified. Preserve
these separate scopes. Run the full research calibration only after the chosen
hardware/economics path qualifies; there is no new GPU command to run now.

Reproduce this metadata-only proposal from the repository root using fresh paths:

```bash
.venv/bin/python -B scripts/prepare_research_attack_candidate.py \
  --candidate /tmp/research-attack-candidate-UNIQUE.json \
  --report /tmp/research-candidate-review-UNIQUE.json
```

Twenty-three focused tests, the scaffold check and the independent fraction
budget audit pass. The builder reads metadata/reports only and preserves old
artifacts. Neither this file nor the candidate authorizes paid work, extra
attempt time, held-out evaluation, configuration adoption or training.
