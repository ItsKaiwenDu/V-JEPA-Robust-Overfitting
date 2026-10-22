"""Profiler parser regressions using synthetic events; actual CUDA stays explicit."""

import copy
from types import SimpleNamespace

import pytest
import torch

from models.execution import configure_execution
from models.native_probe import kernel_boundary_checks, native_backward_probe, profile_records, validate_probe
from test_native_execution import restore_flags

# Stem taken from the verified university native Conv3d profiles. Rows below
# are synthetic parser fixtures, not a newly executed GPU result.
KERNEL = 'void at::native::vol2im_kernel<float, float>(long, float const*, ...)'


def records():
    return [{'name': 'aten::convolution_backward', 'device_type': 'cpu', 'duration_us': 30.},
            {'name': KERNEL, 'device_type': 'cuda', 'duration_us': 12.}]


def test_recorded_kernel_stem_is_volume_reconstruction_not_the_helper_name():
    assert 'col2vol' not in KERNEL
    checks = kernel_boundary_checks(records())
    assert checks['passed'] and checks['native_reconstruction_kernels'] == [KERNEL]


@pytest.mark.parametrize('defect', ['cpu_only', 'zero_duration', 'missing_backward', 'unrelated_kernel',
                                  'cudnn_device', 'cudnn_operator'])
def test_missing_or_wrong_backend_evidence_cannot_pass(defect):
    rows = records()
    if defect == 'cpu_only':
        rows[1]['device_type'] = 'cpu'
    elif defect == 'zero_duration':
        rows[1]['duration_us'] = 0.
    elif defect == 'missing_backward':
        rows[0]['name'] = 'aten::mul'
    elif defect == 'unrelated_kernel':
        rows[1]['name'] = 'ampere_sgemm_128x128_nn'
    elif defect == 'cudnn_device':
        rows.append(dict(name='cudnn::cnn::dgrad2d_grouped_direct_kernel', device_type='cuda', duration_us=13.))
    else:
        rows.append(dict(name='aten::cudnn_convolution_backward', device_type='cpu', duration_us=13.))
    assert not kernel_boundary_checks(rows)['passed']


@pytest.mark.parametrize('duration', [float('nan'), float('inf'), -1.])
def test_invalid_profiler_durations_are_rejected(duration):
    rows = records()
    rows[1]['duration_us'] = duration
    with pytest.raises(ValueError, match='Invalid native backward profiler'):
        kernel_boundary_checks(rows)


def test_raw_profiler_event_device_type_is_not_inferred_from_its_name():
    # A CPU event with a native kernel-looking name cannot prove a CUDA launch.
    event = SimpleNamespace(name=KERNEL, device_type=torch.autograd.DeviceType.CPU,
                            time_range=SimpleNamespace(start=10., end=22.))
    parsed = profile_records(SimpleNamespace(events=lambda: [event]))
    assert parsed[0]['device_type'] == 'cpu' and not kernel_boundary_checks(parsed)['passed']


def test_actual_cpu_probe_records_flags_and_does_not_claim_cuda():
    configure_execution('native-fp32', device='cpu')
    evidence = native_backward_probe('cpu')
    assert validate_probe(evidence, device='cpu')
    assert not evidence['actual_cuda_kernel_evidence']
    assert not evidence['kernel_boundary_checks']['cuda_events_observed']
    assert evidence['flags_before'] == evidence['flags_after_backward']


def test_failed_probe_evidence_is_retained_before_rejection(monkeypatch):
    configure_execution('native-fp32', device='cpu')
    # Inject impossible CUDA evidence into a real CPU fixture. It must reject,
    # but the caller must receive the recorded evidence before that exception.
    monkeypatch.setattr('models.native_probe.profile_records', lambda profile: records())
    retained = []
    with pytest.raises(ValueError, match='CPU fixture unexpectedly'):
        native_backward_probe('cpu', on_evidence=lambda value: retained.append(copy.deepcopy(value)))
    assert len(retained) == 1 and retained[0]['profiler_events'] == records()


def test_export_validation_recomputes_instead_of_trusting_pass_boolean():
    configure_execution('native-fp32', device='cpu')
    evidence = native_backward_probe('cpu')
    evidence['kernel_boundary_checks']['passed'] = True
    with pytest.raises(ValueError, match='disagrees with retained'):
        validate_probe(evidence, device='cpu')


def test_worker_json_keeps_probe_evidence_on_a_rejected_probe(tmp_path, monkeypatch):
    import json
    from models.native_diagnostic import worker
    args = SimpleNamespace(result=tmp_path / 'worker.json', device='cpu', tiny=True,
                           execution_policy='native-fp32', upstream=tmp_path)
    evidence = {'profiler_events': records(), 'schema_version': 2}
    def rejected(device, *, on_evidence):
        on_evidence(evidence)
        raise RuntimeError('Injected probe rejection')
    monkeypatch.setattr('models.native_diagnostic.provenance', lambda root, upstream: {})
    monkeypatch.setattr('models.native_diagnostic.native_backward_probe', rejected)
    with pytest.raises(RuntimeError, match='Injected probe rejection'):
        worker(args, tmp_path)
    value = json.loads(args.result.read_text())
    assert not value['passed'] and value['finished']
    assert value['stage'] == 'native_backward_kernel_probe'
    assert value['native_backward_probe'] == evidence
