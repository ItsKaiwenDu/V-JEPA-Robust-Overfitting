"""Independent analytic derivatives, execution layouts, and evidence rejection."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from models.cuda_convolution import (MODES, TOLERANCES, VARIANTS, agreement, analytical_reference,
                                     evaluate, execution_context, make_operands, prepare_execution, summarize)
from models.cuda_profile import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('shape,kernel', [((2, 2, 4, 6, 9), (2, 3, 3)),
                                        ((1, 2, 6, 8, 12), (3, 2, 3))])
def test_analytic_unpatch_and_all_derivatives_match_conv_double(shape, kernel):
    torch.manual_seed(72)
    conv = torch.nn.Conv3d(2, 5, kernel_size=kernel, stride=kernel).double()
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    y = conv(x).flatten(2).transpose(1, 2)
    cot = torch.randn_like(y)
    gradients = torch.autograd.grad(y, (x, conv.weight, conv.bias), grad_outputs=cot)
    oracle = analytical_reference(x, conv.weight, conv.bias, cot)
    for key, actual in zip(('output', 'input', 'weight', 'bias'), (y, *gradients)):
        torch.testing.assert_close(oracle[key], actual, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('mode', MODES)
def test_original_conv_layouts_and_gradients_preserve_values(upstream, variant, mode):
    module, x, cot = make_operands(upstream, True)
    before = copy.deepcopy(module.state_dict())
    op, leaves, gout, layout = prepare_execution(module, x, cot, variant, mode, 'cpu')
    values, times = evaluate(op, leaves, gout, variant, 'cpu')
    expected = analytical_reference(x, module.proj.weight, module.proj.bias, cot)
    for key, value in values.items():
        assert agreement(value, expected[key], TOLERANCES[key])['passed']
    assert set(values) == ({'output', 'input'} if mode == 'input' else {'output', 'input', 'weight', 'bias'})
    assert all(torch.equal(before[k], v) for k, v in module.state_dict().items())
    assert layout['state_keys'] == ['proj.weight', 'proj.bias']
    assert layout['input_conversion_included_in_forward'] == (variant == 'channels-last-3d')
    assert all(v > 0 for v in times.values())
    if variant == 'channels-last-3d':
        assert layout['convolution_input_stride'] != layout['input_leaf_stride']


def test_elementwise_error_gate_rejects_small_local_error_and_nonfinite_values():
    ref = torch.zeros(10000)
    actual = ref.clone(); actual[170] = 1e-3
    result = agreement(actual, ref, TOLERANCES['input'])
    assert not result['passed'] and result['failed_elements'] == 1
    assert result['max_tolerance_ratio'] > 1
    with pytest.raises(ValueError, match='nonfinite'):
        agreement(torch.tensor([float('nan')]), torch.zeros(1), TOLERANCES['input'])


@pytest.mark.parametrize('enabled', (True, False))
def test_native_context_retains_explicit_precision_and_restores_backend(monkeypatch, enabled):
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', enabled)
    monkeypatch.setattr(torch.backends.cudnn, 'enabled', True)
    with execution_context('native'):
        assert torch.backends.cudnn.allow_tf32 == enabled
        assert not torch.backends.cudnn.enabled and torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.enabled and torch.backends.cudnn.allow_tf32 == enabled


@pytest.fixture(scope='module', params=('on', 'off'))
def roundtrip(request, tmp_path_factory):
    root = tmp_path_factory.mktemp('conv-roundtrip') / 'result'
    command = [sys.executable, str(ROOT / 'scripts/compare_cuda_convolution.py'),
               '--device', 'cpu', '--tiny', '--upstream', request.config.getoption('--upstream'),
               '--output', str(root), '--variants', 'contiguous', '--cudnn-tf32', request.param,
               '--max-seconds', '90', '--worker-seconds', '20']
    result = subprocess.run(command, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stderr + result.stdout
    session = json.loads((root / 'session.json').read_text())
    return root, session


def test_fresh_process_correctness_replay_and_optional_profiler(roundtrip):
    root, session = roundtrip
    assert session['complete'] and session['all_checks_passed'] and session['selected_variant'] is None
    assert len(session['tasks']) == 4 and not session['model_changed']
    for task in session['tasks']:
        w = json.loads((root / task['result']).read_text())
        assert w['validation_passed'] and w['within_process_exact'] and w['finished']
        assert w['contract']['cudnn_tf32_policy'] == session['cudnn_tf32_policy']
        assert w['contract']['runtime']['cudnn_tf32'] == (session['cudnn_tf32_policy'] == 'on')
        assert w['validation_signatures']['candidate'] == w['signature']
        assert w['profiler']['requested']
        if w['profiler'].get('available'):
            assert any('conv' in item['name'].lower() for item in w['profiler']['top_cpu'])
    for mode in MODES:
        comparison = session['comparisons']['contiguous'][mode]
        assert comparison['qualified'] and comparison['candidate_to_reference_total_ratio'] == 1


@pytest.mark.parametrize('defect', ('timeout', 'checksum', 'source', 'tolerance', 'runtime',
                                   'precision_policy', 'gradient', 'numerical_failure', 'missing_weight', 'timing'))
def test_invalid_or_nonreplaying_evidence_cannot_qualify(roundtrip, tmp_path, defect):
    source, session = roundtrip
    dest = tmp_path / 'result'; shutil.copytree(source, dest)
    tasks = copy.deepcopy(session['tasks'])
    task = tasks[-1]
    if defect == 'timeout':
        task['status'] = 'timeout'
    else:
        path = dest / task['result']; value = json.loads(path.read_text())
        if defect == 'checksum': value['signature']['input']['sha256'] = 'tampered'
        elif defect == 'source': value['provenance'] = {'source': 'changed'}
        elif defect == 'tolerance': value['contract']['tolerances']['weight']['atol'] = 1
        elif defect == 'runtime': value['contract']['runtime']['deterministic_algorithms'] = False
        elif defect == 'precision_policy':
            value['contract']['cudnn_tf32_policy'] = 'off' if session['cudnn_tf32_policy'] == 'on' else 'on'
        elif defect == 'gradient': value['signature']['input']['sha256'] = 'different'
        elif defect == 'numerical_failure': value['validation']['candidate_vs_oracle']['input'].update(passed=False, failed_elements=1)
        elif defect == 'missing_weight': value['validation']['candidate_vs_oracle'].pop('weight')
        elif defect == 'timing': value['median_seconds']['total_seconds'] = -1
        write_json(path, value)
        if defect != 'checksum': task['result_sha256'] = file_hash(path)
    comparison = summarize(dest, tasks, ['contiguous'], session['provenance'],
                           cudnn_tf32=session['cudnn_tf32_policy'])['contiguous']['parameters']
    assert not comparison['qualified']
    assert 'candidate_to_reference_total_ratio' not in comparison


def test_deadline_preserves_incomplete_evidence(upstream, tmp_path):
    dest = tmp_path / 'deadline'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/compare_cuda_convolution.py'),
        '--device', 'cpu', '--tiny', '--upstream', str(upstream), '--output', str(dest),
        '--variants', 'contiguous', '--max-seconds', '1', '--worker-seconds', '.05'],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 1, result.stderr
    session = json.loads((dest / 'session.json').read_text())
    assert session['finished'] and not session['complete'] and not session['all_checks_passed']
    assert any(t['status'] in ('timeout', 'not_started_deadline') for t in session['tasks'])


def test_cli_refuses_fallback_reuse_or_invalid_variant(tmp_path):
    dest = tmp_path / 'existing'; dest.mkdir()
    for args in (['--device', 'cpu'], ['--device', 'cuda', '--tiny'],
                 ['--device', 'cpu', '--tiny', '--output', str(dest)],
                 ['--device', 'cpu', '--tiny', '--output', str(tmp_path / 'new'), '--variants', 'native']):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/compare_cuda_convolution.py'),
                                 '--upstream', 'missing', *args], capture_output=True, text=True)
        assert result.returncode == 2
    assert list(dest.iterdir()) == []
