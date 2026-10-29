"""Complete-panel gating, original identities and independent descriptive arithmetic."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import math
from pathlib import Path

import pytest

from models.registered_panel_analysis import (AUTHORITY, AUTHORITY_SHA, BANKS, KIND, ORDER, PLAN_SHA,
    authority, bind_identity, canonical, collect, path_in, require_complete, source_location, validate_manifest, verify_pairs)
from models.matched_panel import aggregate_scores


def fixture():
    cases = [dict(case_id=cid, clip_index=i // 2, video_id=f'generated-{i // 2}',
        attack_bank=i % 2, attack_seeds=[i + 10]) for i, cid in enumerate(ORDER)]
    plan = dict(cases=cases, ordered_clips=[dict(video_id=f'generated-{i}') for i in range(10)])
    scores, receipts = {}, {}
    for i, case in enumerate(cases):
        # Deliberately unequal lower losses exercise mean-of-relative-gaps versus pooled ratios.
        base = Fraction(1 + i, 4)
        changes = dict(A=Fraction(0), B=Fraction(i % 3 - 1, 10), C=Fraction(1, 8), D=Fraction(i % 3, 10))
        scores[case['case_id']] = {arm: {bank: dict(clean=float(Fraction(1, 10) + k),
            adversarial=float(base + change + 2 * k)) for k, bank in enumerate(BANKS)} for arm, change in changes.items()}
        receipts[case['case_id']] = dict(contract=dict(clip=case['video_id'], runtime='generated-common-runtime',
            weights='generated-common-weights', settings='generated-settings', clean_sha256='clean',
            targets_sha256='target', starts_sha256='start',
            mask_tensors_sha256=dict(attack_0=str(i), main_0='same-main', audit_0_0='same-audit')),
            execution_sha256=f'generated-original-execution-{i}', scores=scores[case['case_id']])
    return plan, scores, receipts


def oracle(scores, contrast, bank):
    low, high = {'B-A': ('A', 'B'), 'D-C': ('C', 'D'), 'C-A': ('A', 'C'), 'D-B': ('B', 'D')}[contrast]
    clips, percentages = [], []
    for i in range(10):
        absolute, relative = [], []
        for b in (0, 1):
            row = scores[f'clip-{i:02d}-bank-{b}']
            a, z = (Fraction.from_float(row[x][bank]['adversarial']) for x in (low, high))
            absolute.append(z - a)
            relative.append(None if not a else 100 * (z / a - 1))
        clips.append(sum(absolute) / 2)
        percentages.append(None if None in relative else sum(relative) / 2)
    return clips, percentages


@pytest.mark.parametrize('contrast', ['B-A', 'D-C', 'C-A', 'D-B'])
@pytest.mark.parametrize('bank', BANKS)
def test_frozen_aggregator_matches_independent_fraction_arithmetic(contrast, bank):
    plan, scores, _ = fixture()
    summary = aggregate_scores(plan, scores)['summary'][contrast][bank]
    absolute, relative = oracle(scores, contrast, bank)
    assert summary['mean'] == pytest.approx(float(sum(absolute) / 10), abs=1e-14)
    assert summary['relative_percent_mean'] == pytest.approx(float(sum(relative) / 10), abs=1e-12)
    assert summary['minimum'] == float(min(absolute))
    assert summary['maximum'] == float(max(absolute))
    assert [summary[k + '_clips'] for k in ('positive', 'zero', 'negative')] == [sum(v > 0 for v in absolute), sum(v == 0 for v in absolute), sum(v < 0 for v in absolute)]


def test_audit_banks_stay_separate_and_n_is_ten_clips():
    plan, scores, _ = fixture()
    first = aggregate_scores(plan, scores)
    changed = deepcopy(scores)
    for row in changed.values():
        for arm in row.values(): arm['audit_0']['adversarial'] *= 100
    second = aggregate_scores(plan, changed)
    assert all(first['summary'][c]['main'] == second['summary'][c]['main'] for c in first['summary'])
    assert first['analysis_unit'] == 'clip' and first['clips'] == 10 and first['technical_banks_per_clip'] == 2
    assert first['score_banks_pooled'] is False
    assert first['automatic_primary_choice'] is False and first['automatic_convergence_pass'] is False


def test_zero_lower_loss_keeps_absolute_gap_and_undefined_relative():
    plan, scores, _ = fixture()
    scores[ORDER[0]]['A']['main']['adversarial'] = 0
    x = aggregate_scores(plan, scores)
    assert x['per_case'][ORDER[0]]['B-A']['main']['relative_percent'] is None
    assert x['summary']['B-A']['main']['relative_percent_mean'] is None
    assert math.isfinite(x['summary']['B-A']['main']['mean'])


@pytest.mark.parametrize('change', ['missing', 'extra'])
def test_partial_or_extra_collection_rejected(change):
    plan, scores, _ = fixture()
    if change == 'missing': scores.pop(ORDER[-1])
    else: scores['unknown-case'] = scores[ORDER[0]]
    with pytest.raises(ValueError): require_complete(scores)
    with pytest.raises(ValueError): aggregate_scores(plan, scores)


@pytest.mark.parametrize('field', ['runtime', 'weights', 'settings', 'targets_sha256', 'starts_sha256', 'clean_sha256', 'mask_tensors_sha256'])
def test_paired_shared_identity_drift_rejected(field):
    plan, _, receipts = fixture()
    receipts[ORDER[1]]['contract'][field] = {} if field == 'mask_tensors_sha256' else 'changed'
    with pytest.raises(ValueError): verify_pairs(plan, receipts)


def test_clean_baseline_drift_rejected_before_aggregation():
    plan, _, receipts = fixture()
    receipts[ORDER[1]]['scores']['B']['main']['clean'] += 1
    with pytest.raises(ValueError): verify_pairs(plan, receipts)


def test_original_mixed_execution_ids_preserved_while_shared_inputs_match():
    plan, _, receipts = fixture()
    checks = verify_pairs(plan, receipts)
    assert len(checks) == 10
    assert checks[0]['original_execution_ids'] == ['generated-original-execution-0', 'generated-original-execution-1']


def test_failure_in_final_case_cannot_return_any_complete_collection():
    plan, _, receipts = fixture()
    called = []
    def loader(cid, item):
        called.append(cid)
        if cid == ORDER[-1]: raise ValueError('Generated final-case evidence mismatch')
        return receipts[cid], {}
    with pytest.raises(ValueError): collect(plan, dict.fromkeys(ORDER), loader)
    assert called == list(ORDER)


@pytest.mark.parametrize('change', ['case', 'host', 'fixture', 'execution', 'receipt', 'plan', 'replay'])
def test_original_registered_case_binding_cannot_be_substituted(change):
    cid = ORDER[0]
    execution = dict(domain='university-cuda', expected_host='original-host', research_admitted=False)
    identity = dict(case_id=cid, host='original-host', execution_sha256=canonical(execution))
    receipt = dict(case_id=cid, execution_sha256=identity['execution_sha256'], plan_sha256=PLAN_SHA,
                   fresh_A_replay_exact=True, research_admitted=False)
    bind_identity(cid, identity, execution, receipt)
    if change == 'case': identity['case_id'] = ORDER[1]
    elif change == 'host': identity['host'] = 'another-host'
    elif change == 'fixture': execution['domain'] = 'generated-cpu-fixture'
    elif change == 'execution': execution['research_admitted'] = True
    elif change == 'receipt': receipt['execution_sha256'] = '0' * 64
    elif change == 'plan': receipt['plan_sha256'] = '0' * 64
    else: receipt['fresh_A_replay_exact'] = False
    with pytest.raises(ValueError): bind_identity(cid, identity, execution, receipt)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'a/../outside', 'a\\b'])
def test_unsafe_evidence_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError): path_in(tmp_path, name)


def test_symlink_cannot_redirect_evidence(tmp_path):
    (tmp_path / 'alias').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError): path_in(tmp_path, 'alias/file')


def test_fake_registration_is_rejected_before_opening_any_case(tmp_path):
    path = tmp_path / AUTHORITY
    path.parent.mkdir(parents=True)
    path.write_text('{"verified": true}')
    with pytest.raises(ValueError): authority(tmp_path)


def test_manifest_preserves_original_origins_and_all_twenty():
    index = {cid: dict(case_id=cid, origin=f'original-{i}') for i, cid in enumerate(ORDER)}
    manifest = dict(kind=KIND + '-inputs', authority_path=AUTHORITY, authority_sha256=AUTHORITY_SHA,
        plan_sha256=PLAN_SHA, research_admitted=False,
        cases={cid: dict(registered_identity=deepcopy(row)) for cid, row in index.items()})
    validate_manifest(manifest, dict(registered_case_index=index))
    manifest['cases'][ORDER[-1]]['registered_identity']['origin'] = 'rewritten'
    with pytest.raises(ValueError): validate_manifest(manifest, dict(registered_case_index=index))


def test_original_split_source_layout_is_declared_and_never_falls_back(tmp_path):
    directory = tmp_path / 'case'
    directory.mkdir()
    captured = directory / 'source/attacks/__init__.py'
    captured.parent.mkdir(parents=True)
    captured.write_text('Generated misplaced dependency; not the qualified source')
    expected = tmp_path / 'runs/pretrained/diagnostic/matched-panel-handoff-20261024/snapshot/attacks/__init__.py'
    assert source_location(tmp_path, directory, 'attacks/__init__.py') == expected
    # Missing per-case runner files must not silently use the snapshot's copy.
    assert source_location(tmp_path, directory, 'models/matched_panel.py') == directory / 'source/models/matched_panel.py'
