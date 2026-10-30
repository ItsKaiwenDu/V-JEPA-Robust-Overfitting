"""Immutable packet identity, source closure and no implicit resource permission."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from models.candidate_packet import (BASE_MANIFEST_SHA256, EXTRA, EXTRA_SHA256,
    CPU_TESTS, build, canonical, launch_template, setup_requirements, source_files)
from models.pilot_bundle import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/pretrained/diagnostic/candidate-eot8-handoff-v3-20261028'


@pytest.fixture
def manifest():
    assert digest(BASE / 'manifest.json') == BASE_MANIFEST_SHA256
    return json.loads((BASE / 'manifest.json').read_text())


def test_exact_allowlist_preserves_science_and_includes_provenance_dependency(manifest):
    files = source_files(ROOT, manifest)
    assert set(files) == set(manifest['archives']['source']['files']) | {EXTRA}
    assert len(files) == 76 and digest(files[EXTRA]) == EXTRA_SHA256
    assert set(CPU_TESTS) <= set(files)
    assert 'tests/test_candidate_budget.py' not in files
    assert 'tests/test_candidate_packet.py' not in files
    assert not any(n.startswith(('runs/', '.ssh/')) or n.endswith(('.pt', '.whl')) for n in files)


@pytest.mark.parametrize('defect', ['legacy', 'changed-binding', 'extra-preexisting'])
def test_changed_or_relabelled_source_metadata_is_refused(manifest, defect):
    if defect == 'legacy': manifest['candidate']['candidate_binding']['profile'] = 'old-four-draw'
    elif defect == 'changed-binding': manifest['candidate']['candidate_binding']['source_sha256']['models/training.py'] = '0' * 64
    else: manifest['archives']['source']['files'][EXTRA] = {'sha256': EXTRA_SHA256, 'bytes': 8187}
    with pytest.raises(ValueError): source_files(ROOT, manifest)


@pytest.fixture
def isolated_source(tmp_path, manifest):
    for name, path in source_files(ROOT, manifest).items():
        target = tmp_path / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    return tmp_path


@pytest.mark.parametrize('defect', ['missing-provenance', 'changed-test', 'symlink'])
def test_missing_or_changed_bytes_fail_before_packet_creation(isolated_source, manifest, defect):
    path = isolated_source / (EXTRA if defect != 'changed-test' else CPU_TESTS[0])
    if defect == 'changed-test': path.write_text(path.read_text() + '\n# drift\n')
    else:
        path.unlink()
        if defect == 'symlink': path.symlink_to(ROOT / EXTRA)
    with pytest.raises(ValueError, match='Pinned source is missing or changed'):
        source_files(isolated_source, manifest)


def test_launch_template_is_unarmed_and_preserves_all_phases(manifest):
    p = launch_template(manifest['candidate'])
    assert not p['authorized'] and p['diagnostic_only']
    assert len(p['required_phases']) == 13
    assert p['candidate_binding'] == manifest['candidate']['candidate_binding']
    assert p['container_gb'] == 30 and p['persistent_gb'] == 80 and p['minimum_system_ram_gib'] == 40
    for key in ['total_hourly_usd', 'inclusive_session_budget_usd', 'deployed_unix',
                'deadline_unix', 'expected_gpu', 'expected_gpu_uuid', 'ledger_sha256']:
        assert p[key] is None
    p['candidate_binding']['profile'] = 'mutated'
    assert manifest['candidate']['candidate_binding']['profile'] == 'research-primary-eot8-v1'


def test_workspace_CPU_fixture_recipe_does_not_reuse_an_old_grant():
    v = setup_requirements('a' * 64)
    assert v['CPU_basetemp_must_be_on_workspace'] and 'workspace' in v['CPU_basetemp']
    assert v['expected_CPU_test_count'] == 77 and v['minimum_free_workspace_gib_after_inputs'] == 55
    assert v['run_all_13_original_benchmark_phases']
    assert not any(v[k] for k in ['launch_authorized', 'old_session_deadline_or_grant_reused',
        'old_financial_hold_released', 'new_financial_hold_created', 'remote_upload_performed'])
    assert v['session_budget_usd'] is None and v['hourly_quote_usd'] is None and v['SSH_endpoint'] is None


def test_builder_refuses_existing_output_and_bad_base_without_overwriting(tmp_path):
    existing = tmp_path / 'existing'; existing.mkdir(); marker = existing / 'evidence'; marker.write_text('preserve')
    with pytest.raises(FileExistsError): build(ROOT, BASE, existing)
    assert marker.read_text() == 'preserve'
    wrong = tmp_path / 'wrong'; wrong.mkdir(); (wrong / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='immutable verified v3'): build(ROOT, wrong, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
