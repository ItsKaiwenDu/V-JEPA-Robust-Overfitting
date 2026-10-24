# A40 setup portability controls — October 23, 2026

The first setup on the replacement A40 is rejected: **507 passed, two failed,
zero errors/skips**. Its 43,563-byte export and all ten members verify locally.
No pretrained checkpoint download or benchmark phase started in that attempt.
The rejected run and source snapshot are preserved.

The generated CPU profiler fixture exceeded its ten-second allowance in all
four fresh workers. Saved progress ends during component construction or
warmup backward; the precise low-level startup delay is not proved. The positive
test now allows 30 seconds per worker, 150 seconds overall and 180 seconds for
its subprocess. Existing short timeout/descendant-kill tests are unchanged.
No profiler production limits or scientific thresholds changed.

The budget fixture remains exactly reproducible within Python 3.12. Its
comparison with the Python 3.11 report differs in 59 float leaves, at most
3.411e-13; all non-float leaves match. Explicit sequential summation reproduces
the historical variants exactly. [Python documents the 3.12 float-summation
change](https://docs.python.org/3.12/library/functions.html#sum).
The test now checks native-runtime repeatability exactly and replays historical
summation explicitly for exact historical equality. Historical reports, source
pins, production budget arithmetic and admission decisions remain unchanged.

All 39 relevant local checks pass, both before packaging and after extraction.
Both repaired positive controls pass on the same A40 in 54.459 seconds, with
zero skips. The separately pinned retry changes only the two test files; all
395 restored members verify. The complete replacement suite now passes all 509 cases with zero skips,
errors or failures in 1161.827s. JUnit/runtime copies verify locally. This is
a full current-suite result; the original rejected attempt remains unchanged.
The setup timer expired after download; exit124 is preserved. Independent
verification then accepts the complete checkpoint hash and source/JUnit pins.
User requested a break; see the [resume checkpoint](2026-10-23-a40-pause-checkpoint.md).

The retry uses the same existing Pod, $0.61/hour quote, $2.80 hold and absolute
**2:58 PM Pacific** cutoff. It adds no rental or hold. Billing remains active
until the user stops/terminates the Pod. New pretrained timings, economic
screen, A40 recovery and research admission remain pending.

See [machine-readable evidence](2026-10-23-a40-runtime-portability.json).
