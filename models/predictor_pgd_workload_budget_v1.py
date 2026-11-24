"""Closed execution gate for the presently unqualified whole-unit budget.

No dictionary, authorization file or CLI option can qualify missing performance
evidence. A future budget solution requires a separately reviewed source version.
"""
REVIEWED_COMPLETE_UNIT_BUDGET = None


def require_execution():
    raise ValueError('Whole-PGD budget is unqualified: no new scope or launch; '
                     'complete preparation/scoring/observation and replay timing first.')
