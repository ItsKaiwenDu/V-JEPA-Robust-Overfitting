# Positive test startup and historical arithmetic

The [A40 portability report](../report/checks/2026-10-23-a40-runtime-portability.md)
records the two rejected setup checks, their retained evidence, the test-only
repairs and the independent controls. The complete replacement CUDA suite
must pass before pretrained measurements are accepted.

Fresh interpreter/import/autograd startup counts against the positive CPU
profiler fixture's finite 30-second worker allowance. A timeout remains a
failure. Separate deliberately short-budget tests retain descendant-kill and
deadline coverage; production diagnostic caps and numerical gates are unchanged.

Historical budget comparisons explicitly replay sequential float summation
from the Python 3.11 report. Native-runtime repeatability, historical equality,
source checks, workload counts and budget decisions remain exact. Do not
rewrite historical evidence or loosen tolerances to accommodate runtime changes.
