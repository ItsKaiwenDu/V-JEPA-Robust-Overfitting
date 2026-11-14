"""Strict tensor acceptance with separately validated diagnostic bookkeeping.

Callers must recompute checks from verified saved arrays. This helper cannot
establish tensor, worker, backend, runtime or full-cohort identity on its own.
Only relative_l2_error uses the existing bookkeeping bound. No field is dropped
or rewritten, and a failed elementwise check always rejects.
"""
import math

POLICY = 'fixed-SDPA-numerical-receipt-bookkeeping-v1'
KIND = 'fixed-SDPA-numerical-receipt-comparison-v1'
RETURNS = ('output', 'dq', 'dk', 'dv')
BOOK = {'rtol': 1e-12, 'atol': 1e-14}
BOUNDS = {'output': {'rtol': 1e-4, 'atol': 2e-5},
          **{name: {'rtol': .002, 'atol': 2e-7} for name in ('dq', 'dk', 'dv')}}
FIELDS = frozenset(('passed', 'failed_elements', 'shape', 'max_absolute_error',
                    'max_tolerance_ratio', 'relative_l2_error', 'rtol', 'atol'))


def _validate(value, name):
    if type(value) is not dict or set(value) != FIELDS:
        raise ValueError('Exact numerical receipt fields required.')
    if (type(value['passed']) is not bool or value['passed'] is not True
            or type(value['failed_elements']) is not int or value['failed_elements'] != 0):
        raise ValueError('Actual elementwise acceptance must pass with zero failed elements.')
    shape = value['shape']
    if type(shape) is not list or len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Exact positive four-dimensional shape required.')
    for field in ('max_absolute_error', 'max_tolerance_ratio', 'relative_l2_error', 'rtol', 'atol'):
        number = value[field]
        if type(number) is not float or not math.isfinite(number) or number < 0:
            raise ValueError('Finite nonnegative floating-point receipt metrics required.')
    if {field: value[field] for field in ('rtol', 'atol')} != BOUNDS[name]:
        raise ValueError('Original tensor error bounds cannot change.')
    if value['max_tolerance_ratio'] > 1.0:
        raise ValueError('Actual elementwise error exceeds the original bound.')
    if value['max_absolute_error'] == 0.0 and (value['max_tolerance_ratio'] != 0.0 or value['relative_l2_error'] != 0.0):
        raise ValueError('An exact zero-error receipt must retain exact zero summaries.')


def compare_checks(recomputed, recorded, *, context):
    """Compare every field; never turn approximate bookkeeping into acceptance."""
    if context not in ('canonical-FP64', 'passing-original'):
        raise ValueError('Explicit canonical or passing-original comparison required.')
    if any(type(value) is not dict or set(value) != set(RETURNS) for value in (recomputed, recorded)):
        raise ValueError('All four recomputed and recorded return checks required.')
    details = {}
    for name in RETURNS:
        actual, prior = recomputed[name], recorded[name]
        _validate(actual, name)
        _validate(prior, name)
        exact_fields = FIELDS - {'relative_l2_error'}
        if any(actual[field] != prior[field] for field in exact_fields):
            raise ValueError('Numerical acceptance fields must match exactly.')
        delta = abs(prior['relative_l2_error'] - actual['relative_l2_error'])
        allowed = BOOK['atol'] + BOOK['rtol'] * abs(actual['relative_l2_error'])
        if not math.isfinite(delta) or delta > allowed:
            raise ValueError('Relative L2 diagnostic disagrees beyond the original bookkeeping bound.')
        details[name] = {'exact_summary_match': prior['relative_l2_error'] == actual['relative_l2_error'],
                         'recorded': prior['relative_l2_error'], 'recomputed': actual['relative_l2_error'],
                         'absolute_difference': delta, 'allowed_bookkeeping_difference': allowed}
    return {'kind': KIND, 'policy': POLICY, 'context': context, 'comparison_passed': True,
            'bookkeeping': dict(BOOK), 'tensor_bounds': {name: dict(BOUNDS[name]) for name in RETURNS},
            'relative_l2': details, 'all_acceptance_fields_exact': True,
            'original_elementwise_bounds_passed': True, 'producer_records_rewritten': False,
            'worker_completion_inferred': False, 'research_admitted': False, 'performance_qualified': False}
