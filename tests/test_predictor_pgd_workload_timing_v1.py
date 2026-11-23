"""New controls for caller-level timing; no old passed node or native objective repeat."""
import json
import os
from pathlib import Path

import pytest
import torch

from models.predictor_pgd_workload_timing_v1 import matched_analytic_units
from models.predictor_pgd_workload_records_v1 import validate_receipt


@pytest.mark.parametrize('method,step,audits', [('primary', .5, 0), ('convergence', 2., 2)])
def test_caller_timing_includes_complete_restoration_and_observations(tmp_path, request, method, step, audits):
    base = os.environ.get('VJEPA_PGD_WORKLOAD_EVIDENCE')
    folder = Path(base)/request.node.name if base else tmp_path/'case'
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        summary = matched_analytic_units(folder, method=method, step_size_255=step, sensitivity_banks=audits)
    finally:
        torch.set_num_threads(old)
    a = json.loads((folder/'matched/original-complete.json').read_text())
    b = json.loads((folder/'matched/candidate-complete.json').read_text())
    assert summary['outputs_exact'] and summary['observation_overhead'].startswith('included')
    assert a['caller_complete_seconds'] >= a['metrics']['seconds']
    assert b['caller_complete_seconds'] >= b['unit']['full_unit_seconds_including_guards_journal_and_observations']
    assert validate_receipt(b['unit'])['complete_gradients'] == (40 if method == 'primary' else 100)
    assert summary['GPU_or_native_performance_qualified'] is False
