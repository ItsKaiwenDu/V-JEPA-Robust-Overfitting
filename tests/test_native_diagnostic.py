"""Fresh-process fixture evidence and rejection of incomplete/mixed exports."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from models.cuda_profile import file_hash
from models.native_diagnostic import summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def tiny_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp('native-workers') / 'results'
    command = [sys.executable, str(ROOT / 'scripts/check_native_model.py'), '--diagnostic',
               '--tiny', '--device', 'cpu', '--output', str(directory), '--max-seconds', '120', '--worker-seconds', '25']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=135)
    assert result.returncode == 0, result.stderr + result.stdout
    return directory, json.loads((directory / 'session.json').read_text())


def test_fresh_process_pipeline_compares_actual_saved_gradients_and_attacks(tiny_run):
    directory, session = tiny_run
    assert session['complete'] and session['passed'] and len(session['tasks']) == 4
    assert session['tiny_fixture'] and not session['research_result'] and not session['budget_approved']
    assert not session['comparison']['full_model_optimizer_or_recovery_validated']
    checked = summarize(directory, session['tasks'], session['provenance'], tiny=True)
    assert checked == session['comparison']
    assert all(checked['within_backend_exact_replay'].values())


@pytest.mark.parametrize('defect', ['partial', 'timeout', 'duplicate_replay', 'source', 'scale', 'hash', 'policy'])
def test_incomplete_or_mixed_evidence_never_passes(tiny_run, defect):
    directory, session = tiny_run
    tasks, provenance = copy.deepcopy(session['tasks']), copy.deepcopy(session['provenance'])
    tiny = True
    if defect == 'partial':
        tasks.pop()
    elif defect == 'timeout':
        tasks[0]['status'] = 'timeout'
    elif defect == 'duplicate_replay':
        tasks[1]['replay'] = 0
    elif defect == 'source':
        provenance['source_sha256']['models/native_patch.py'] = '0' * 64
    elif defect == 'scale':
        tiny = False
    elif defect == 'hash':
        tasks[0]['result_sha256'] = '0' * 64
    else:
        tasks[0]['policy'] = 'original'
    with pytest.raises(ValueError):
        summarize(directory, tasks, provenance, tiny=tiny)


def test_cli_requires_explicit_diagnostic_and_cpu_fixture_bounds(tmp_path):
    for flags in ([], ['--diagnostic', '--tiny'], ['--diagnostic', '--tiny', '--device', 'cpu', '--worker-seconds', '181']):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_native_model.py'),
                                 '--output', str(tmp_path / 'unused'), *flags], capture_output=True, text=True)
        assert result.returncode == 2 and not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('defect', ['geometry', 'checkpointing', 'precision'])
def test_comparison_retains_shared_geometry_checkpointing_and_actual_precision(tiny_run, tmp_path, defect):
    import shutil
    directory, session = tiny_run
    copied = tmp_path / 'evidence'
    shutil.copytree(directory, copied)
    tasks = copy.deepcopy(session['tasks'])
    for task in tasks:
        if task['policy'] != 'native-fp32':
            continue
        path = copied / task['result']
        value = json.loads(path.read_text())
        runtime = value['contract']['runtime']
        if defect == 'geometry':
            runtime['model_execution']['context_patch']['stride'][0] += 1
        elif defect == 'checkpointing':
            runtime['model_execution']['activation_checkpointing']['encoder'] = False
        else:
            runtime['cudnn_tf32'] = True
        path.write_text(json.dumps(value))
        task['result_sha256'] = file_hash(path)
    with pytest.raises(ValueError):
        summarize(copied, tasks, session['provenance'], tiny=True)
