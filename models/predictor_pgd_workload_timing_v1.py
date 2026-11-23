"""Matched complete-unit CPU control timing, including caller restoration.

This harness only constructs the independent four-element analytic model. It
cannot measure the native scientific route or qualify target performance.
"""
import time
from pathlib import Path

import torch

from models.candidate_evaluation import EvaluationSettings, evaluate_clip
from models.predictor_pgd_workload_v1 import LocalUnit, Snapshot, Journal, tensor_hash, local_boundary
from models.predictor_pgd_workload_controls_v1 import AnalyticFactory, analytic_fixture


def matched_analytic_units(folder, *, method, step_size_255, sensitivity_banks):
    folder = Path(folder)
    folder.mkdir(mode=0o700)
    settings = EvaluationSettings.for_method(method, step_size_255=step_size_255)
    video = 'local-workload-control/timing-boundary-v1'
    original_model, clean, masks = analytic_fixture()
    candidate_model, candidate_clean, candidate_masks = analytic_fixture()
    local_boundary(clean, settings, video, sensitivity_banks)
    original_rng = torch.get_rng_state().clone()
    journal = Journal(folder/'matched')
    journal.put('pair-started', {'method': method, 'step_size_255': step_size_255,
                               'analytic_gradient_budget': settings.steps*settings.restarts*2,
                               'research_admitted': False})

    def measure(role, function):
        started = time.perf_counter()
        journal.put(role+'-started', {'role': role, 'status': 'consumed'})
        metrics, adversarial, unit = function()
        # The caller sees both restoration and the route's terminal publication.
        if not torch.equal(torch.get_rng_state(), original_rng):
            raise ValueError('Complete unit did not restore caller RNG.')
        elapsed = time.perf_counter()-started
        row = {'role': role, 'metrics': metrics, 'adversarial': tensor_hash(adversarial),
               'caller_complete_seconds': elapsed, 'unit': unit}
        journal.put(role+'-complete', row)
        return row, adversarial

    def reference():
        baseline = Snapshot(original_model, (clean,))
        metrics, result = evaluate_clip(original_model, clean, masks, settings,
                                       video_id=video, sensitivity_banks=sensitivity_banks)
        baseline.check()
        return metrics, result, None

    def candidate():
        unit, result = LocalUnit(factory=AnalyticFactory()).run(candidate_model, candidate_clean,
            candidate_masks, settings, video_id=video, sensitivity_banks=sensitivity_banks,
            journal=folder/'candidate')
        return unit['metrics'], result, unit

    a, original_result = measure('original', reference)
    b, candidate_result = measure('candidate', candidate)
    if (not torch.equal(original_result, candidate_result)
            or {k: v for k, v in a['metrics'].items() if k != 'seconds'}
            != {k: v for k, v in b['metrics'].items() if k != 'seconds'}):
        raise ValueError('Matched complete units changed attack/scoring results.')
    summary = {'kind': 'matched-analytic-complete-unit-timing-v1',
               'method': method, 'step_size_255': step_size_255, 'outputs_exact': True,
               'original_caller_seconds': a['caller_complete_seconds'],
               'candidate_caller_seconds': b['caller_complete_seconds'],
               'measurement_boundary': 'caller start record through returned unit, restoration and RNG observation; before matched terminal JSON publication',
               'original_evaluator_seconds_excludes_restoration': a['metrics']['seconds'],
               'candidate_evaluator_seconds_excludes_restoration': b['metrics']['seconds'],
               'candidate_guards_per_step_hashes_and_fsynced_journals_included': True,
               'observation_overhead': 'included; not independently estimated or subtracted',
               'model_setup_outside_timing': True, 'fixed_order': ['original', 'candidate'],
               'statistical_performance_estimate': False, 'GPU_or_native_performance_qualified': False,
               'research_admitted': False}
    journal.put('pair-complete', summary)
    return summary
