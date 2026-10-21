"""Timing and exact recovery comparisons for the bounded CUDA workflow pilot."""

from contextlib import contextmanager
import functools
import json
import math
import os
from pathlib import Path
import time

import torch

from models.checkpointing import CheckpointStore
from models.evaluation_store import validate_records
from models.runtime import synchronize
from models.study_artifacts import read_record


class Timings:
    """Append completed/failed spans; nested spans overlap and must not be added."""

    def __init__(self, path, device):
        self.path, self.device = Path(path), device
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.depth = 0

    @contextmanager
    def span(self, name, **metadata):
        synchronize(self.device)
        started, depth = time.perf_counter(), self.depth
        self.depth += 1
        result = {'name': name, 'depth': depth, **metadata}
        try:
            yield result
            synchronize(self.device)
            result['status'] = 'ok'
        except BaseException as exc:
            result.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            self.depth -= 1
            result['seconds'] = time.perf_counter() - started
            with self.path.open('a') as handle:
                handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())

    def instrument(self, instance, names, prefix):
        for name in names:
            original = getattr(instance, name)

            def wrap(function, label):
                @functools.wraps(function)
                def timed(*args, **kwargs):
                    with self.span(label):
                        return function(*args, **kwargs)
                return timed

            setattr(instance, name, wrap(original, prefix + '.' + name))


def compare_trees(left, right):
    """Exact finite values, types, shapes and dtypes; never loosen GPU tolerance."""
    differences, leaves = [], 0

    def record(path, reason, **extra):
        if len(differences) < 20:
            differences.append({'path': path, 'reason': reason, **extra})

    def visit(a, b, path):
        nonlocal leaves
        if type(a) is not type(b):
            record(path, 'type')
        elif isinstance(a, torch.Tensor):
            leaves += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                record(path, 'tensor_shape_or_dtype')
            elif not torch.isfinite(a).all() or not torch.isfinite(b).all():
                record(path, 'nonfinite_tensor')
            elif not torch.equal(a, b):
                record(path, 'tensor_values', max_absolute_difference=float((a.double() - b.double()).abs().max()))
        elif isinstance(a, dict):
            if set(a) != set(b):
                record(path, 'dictionary_keys')
            for key in a.keys() & b.keys():
                visit(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                record(path, 'sequence_length')
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + '/' + str(i))
        else:
            leaves += 1
            if isinstance(a, float) and (not math.isfinite(a) or not math.isfinite(b)):
                record(path, 'nonfinite_number')
            elif a != b:
                record(path, 'value')

    visit(left, right, '')
    return {'exact_match': not differences, 'compared_leaves': leaves,
            'first_differences': differences, 'difference_list_limit': 20}


def comparison_payload(directory, expected_step):
    """Validate checkpoint plus every receipt before comparing logical contents."""
    directory = Path(directory)
    state, _ = CheckpointStore(directory / 'checkpoints', create=False).load()
    workflow = read_record(directory / 'workflow.json')
    if (state['step'] != expected_step or not workflow['contract']['diagnostic_only']
            or workflow['contract']['training_contract_sha256'] != state['contract_sha256']):
        raise ValueError('Recovery comparison has wrong step, mode, or contract.')
    from models.schedule import EpochSchedule
    schedule = workflow['contract']['schedule']
    schedule = EpochSchedule(schedule['epochs'], tuple(schedule['primary_epochs']), tuple(schedule['panels']))
    count = len(state['contract']['ids'])
    batch = state['contract']['settings']['effective_batch']
    updates_per_epoch = math.ceil(count / batch)
    expected_tasks = [t for t in schedule.tasks() if t['epoch'] * updates_per_epoch <= expected_step]
    rows = workflow['completed_evaluations']
    if [row['task'] for row in rows] != expected_tasks:
        raise ValueError('Recovery comparison requires every evaluation through the saved boundary.')
    evaluations = []
    for row in rows:
        payload = read_record(row['receipt']['path'], expected_sha256=row['receipt']['sha256'])
        validate_records(payload)
        if not payload['complete'] or payload['contract']['checkpoint'] != row['checkpoint']:
            raise ValueError('Incomplete or mismatched evaluation receipt.')
        # Serialization file hashes can differ despite identical logical state.
        # All weight digests, settings, seeds, metrics and other provenance remain.
        contract = dict(payload['contract'])
        contract['checkpoint'] = {k: v for k, v in contract['checkpoint'].items() if k != 'file_sha256'}
        evaluations.append({'task': row['task'], 'contract': contract,
                            'records': [{k: v for k, v in r.items() if k != 'seconds'}
                                        for r in payload['records']]})
    return {'state': state, 'workflow_contract': workflow['contract'], 'evaluations': evaluations}


def compare_recovery(reference, resumed, *, expected_step=2):
    result = compare_trees(comparison_payload(reference, expected_step),
                           comparison_payload(resumed, expected_step))
    return {'kind': 'exact_workflow_recovery_diagnostic', 'expected_step': expected_step,
            **result, 'diagnostic_only': True, 'research_result': False, 'budget_approved': False}
