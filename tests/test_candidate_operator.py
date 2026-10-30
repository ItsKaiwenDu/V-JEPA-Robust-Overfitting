from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from models.candidate_operator import command, export, remaining, run_command, validate

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'runs/pretrained/diagnostic/candidate-eot8-handoff-v4-20261028'


@pytest.fixture
def inputs():
    m = json.loads((PACKET / 'manifest.json').read_text())
    p = json.loads((PACKET / 'launch.template.json').read_text())
    p.update(authorized=True, expected_gpu='NVIDIA RTX A6000', expected_gpu_uuid='GPU-test',
             deployed_unix=1000, deadline_unix=8200, total_hourly_usd='.61', inclusive_session_budget_usd='1.30')
    l = dict(reconciled=True, all_prior_pods_terminated=True, total_cap_usd=200,
        compute_cap_usd=160, pilot_cap_usd=10, noncompute_reserve_usd=40, pilot_spent_and_other_holds_usd='8.6788')
    return p, l, m


def test_valid_bound_includes_setup_and_idle(inputs):
    p, l, m = inputs
    assert validate(p, l, m, 2000) == 6200
    assert remaining(8200, 7200, 2000, 1000, 9000) == 6020
    assert remaining(8200, 7200, 1000, 2000, 9000) == 5020


@pytest.mark.parametrize('defect', ['unauthorized', 'rate', 'overspend', 'old-hold', 'caps', 'duration', 'expired', 'uuid', 'phases', 'binding', 'future'])
def test_rejects_stale_or_unapproved_launch(inputs, defect):
    p, l, m = inputs; now = 2000
    if defect == 'unauthorized': p['authorized'] = False
    elif defect == 'rate': p['total_hourly_usd'] = '.62'
    elif defect == 'overspend': p['inclusive_session_budget_usd'] = '1.40'
    elif defect == 'old-hold': l['pilot_spent_and_other_holds_usd'] = '7.00'
    elif defect == 'caps': l['compute_cap_usd'] = 200
    elif defect == 'duration': p['deadline_unix'] = 9000
    elif defect == 'expired': now = 8100
    elif defect == 'uuid': p['expected_gpu_uuid'] = None
    elif defect == 'phases': p['required_phases'] = ['preflight']
    elif defect == 'binding': p['candidate_binding']['profile'] = 'legacy'
    else: p['deployed_unix'] = 2100
    with pytest.raises(ValueError): validate(p, l, m, now)


def test_clock_cannot_be_reset_and_expiry_stops():
    with pytest.raises(TimeoutError): remaining(8200, 7000, 500, 6900, 900)
    assert remaining(8200, 7000, 500, 2000, 900) == 900


def test_commands_keep_workspace_temp_and_base_torch(tmp_path):
    v = command(tmp_path, 'cpu-fixtures')
    assert '--basetemp=' + str(tmp_path / 'cpu-fixtures-temp') in v
    assert v.count('tests/test_candidate_benchmark.py') == 1
    assert '--system-site-packages' in command(tmp_path, 'venv')
    deps = command(tmp_path, 'dependencies')
    assert '--no-index' in deps and '--no-deps' in deps


def test_stage_failure_and_timeout_are_fail_closed(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_command([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path / 'failed.log', 10, tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path / 'timeout.log', 1, tmp_path)


def test_export_retains_sources_and_preflight_but_not_optimizer_tensors(tmp_path):
    m = json.loads((PACKET / 'manifest.json').read_text())
    p = tmp_path / 'package'; p.mkdir()
    for name in ['manifest.json', 'restore.py']: shutil.copyfile(PACKET / name, p / name)
    for name in m['archives']['source']['files']:
        dest = tmp_path / 'code' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    (tmp_path / 'setup').mkdir(); (tmp_path / 'setup/state.json').write_text('{"failed":true}')
    a = tmp_path / 'run/preflight'; a.mkdir(parents=True); (a / 'saved.pt').write_bytes(b'fixture-not-real-tensors')
    b = tmp_path / 'run/optimizer'; b.mkdir(); (b / 'large.pt').write_bytes(b'excluded-fixture')
    r = export(tmp_path, m)
    assert 'run/preflight/saved.pt' in r['files'] and 'setup/state.json' in r['files']
    assert 'run/optimizer/large.pt' not in r['files']
    assert sum(n.startswith('source/') for n in r['files']) == 76


def test_unarmed_template_stays_unarmed(inputs):
    p, l, m = inputs; template = json.loads((PACKET / 'launch.template.json').read_text())
    with pytest.raises(ValueError): validate(template, l, m, 2000)
