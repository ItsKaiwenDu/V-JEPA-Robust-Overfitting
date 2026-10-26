"""CPU metadata/command guards using genuine previously verified journal metadata.

No neural execution and no generated neural/CUDA pass receipts.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('attended_case', ROOT / 'scripts/attended_panel_case.py')
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
JOURNAL = ROOT / 'tests/fixtures/attended_three_case_journal.json'
REVIEW = ROOT / 'tests/fixtures/attended_recovery_review.json'


def request():
    return {'kind': 'attended-panel-fresh-case-request-v1', 'one_case_only': True,
            'fresh_attempt_only': True, 'case_seconds': 4800, 'kill_grace_seconds': 15,
            'additional_allowance_seconds': 0, 'remaining_queue': False,
            'paid_compute': False, 'research_admitted': False,
            'case_id': 'clip-01-bank-1', 'completed_case_ids': ['clip-00-bank-0', 'clip-00-bank-1', 'clip-01-bank-0'],
            'operator_source_sha256': ops.digest(ROOT / 'scripts/attended_panel_case.py'),
            'journal_file_sha256': ops.digest(JOURNAL), 'previous_review_sha256': ops.digest(REVIEW)}


def boundary(journal=None, review=None, asked=None, **changes):
    original = ops.read_record(JOURNAL)
    args = dict(journal=copy.deepcopy(original) if journal is None else journal,
                review=ops.read_record(REVIEW) if review is None else review,
                request=request() if asked is None else asked, case_order=original['case_order'],
                journal_sha256=ops.digest(JOURNAL), review_sha256=ops.digest(REVIEW))
    args.update(changes)
    return ops.validate_boundary(**args)


def test_real_reviewed_prefix_permits_only_the_next_fresh_case():
    asked = request()
    assert ops.validate_request(asked, ops.canonical(asked)) == 'clip-01-bank-1'
    boundary()


@pytest.mark.parametrize('key,value', [('one_case_only', False), ('one_case_only', 1),
    ('fresh_attempt_only', False), ('case_seconds', 4801), ('case_seconds', True),
    ('kill_grace_seconds', 16), ('additional_allowance_seconds', 80),
    ('remaining_queue', True), ('paid_compute', True), ('research_admitted', True),
    ('case_id', '../escape'), ('case_id', True), ('completed_case_ids', []),
    ('completed_case_ids', ['clip-00-bank-0'] * 2), ('previous_review_sha256', 'unknown')])
def test_changed_or_extended_request_is_refused(key, value):
    asked = request(); asked[key] = value
    with pytest.raises(ValueError):
        ops.validate_request(asked, ops.canonical(asked))


def test_self_consistent_changed_request_still_needs_external_pin():
    asked = request(); original_pin = ops.canonical(asked)
    asked['case_id'] = 'clip-02-bank-0'
    with pytest.raises(ValueError, match='reviewed launch'):
        ops.validate_request(asked, original_pin)


@pytest.mark.parametrize('change', ['journal-pin', 'review-pin', 'missing-case', 'extra-case',
    'missing-first-acceptance', 'old-case', 'skip-next', 'prefix-order', 'runtime-source', 'research'])
def test_stale_or_unreviewed_boundary_rejects(change):
    journal, asked = ops.read_record(JOURNAL), request()
    changes = {}
    if change == 'journal-pin': changes['journal_sha256'] = '0' * 64
    if change == 'review-pin': changes['review_sha256'] = '0' * 64
    if change == 'missing-case': journal['completed'].pop('clip-01-bank-0')
    if change == 'extra-case': journal['completed']['clip-01-bank-1'] = {}
    if change == 'missing-first-acceptance': journal['first_acceptance'] = None
    if change == 'old-case': asked['case_id'] = 'clip-01-bank-0'
    if change == 'skip-next': asked['case_id'] = 'clip-02-bank-0'
    if change == 'prefix-order': asked['completed_case_ids'].reverse()
    if change == 'runtime-source': journal['execution_sha256'] = '0' * 64
    if change == 'research': journal['research_admitted'] = True
    with pytest.raises(ValueError): boundary(journal=journal, asked=asked, **changes)


@pytest.mark.parametrize('key,value', [('reviewed_export', False), ('completed_case_count', 2),
    ('journal_file_sha256', '0' * 64), ('case_receipt_file_sha256', '0' * 64),
    ('attempt', 'attempt-01'), ('case_id', 'clip-00-bank-1'), ('research_admitted', True)])
def test_previous_export_review_cannot_be_replaced_or_repurposed(key, value):
    review = ops.read_record(REVIEW); review[key] = value
    with pytest.raises(ValueError): boundary(review=review)


@pytest.mark.parametrize('name', ['cases/clip-01-bank-1', 'operator-attended/clip-01-bank-1',
    'exports/clip-01-bank-1-attempt-01.tar.gz', 'exports/clip-01-bank-1-attempt-01.tar.gz.manifest.json'])
def test_any_previous_attempt_or_orphan_export_requires_review(tmp_path, name):
    ops.require_unattempted(tmp_path, 'clip-01-bank-1')
    path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
    if name.startswith('exports/'): path.write_text('ordinary metadata fixture')
    else: path.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        ops.require_unattempted(tmp_path, 'clip-01-bank-1')


def test_partial_initial_zero_never_enters_the_legacy_retry_path(tmp_path):
    old = tmp_path / 'cases/clip-01-bank-1/attempt-01'; old.mkdir(parents=True)
    (old / 'session.json').write_text('{"elapsed_seconds": 0, "complete": false}')
    with pytest.raises(ValueError): ops.require_unattempted(tmp_path, 'clip-01-bank-1')
    assert not (tmp_path / 'operator-attended').exists()


def test_operation_reservation_is_single_use_for_the_case_even_with_changed_request(tmp_path):
    asked = request(); directory = ops.reserve_operation(tmp_path, asked)
    assert directory == tmp_path / 'operator-attended/clip-01-bank-1'
    asked['date'] = 'different metadata'
    with pytest.raises(FileExistsError): ops.reserve_operation(tmp_path, asked)


@pytest.mark.parametrize('name', ['../escape', '/outside', 'directory/../escape', './dot', 'a\\b'])
def test_paths_cannot_escape_evidence_root(tmp_path, name):
    with pytest.raises(ValueError): ops.bound(tmp_path, name)


def test_symlink_and_broken_symlink_are_refused(tmp_path):
    link = tmp_path / 'operator-attended'; link.symlink_to(tmp_path / 'missing', target_is_directory=True)
    with pytest.raises(ValueError): ops.require_unattempted(tmp_path, 'clip-01-bank-1')


def test_only_original_supervisor_source_can_be_imported(tmp_path):
    bad = tmp_path / 'unapproved.py'; bad.write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError, match='source changed'): ops.load_supervisor(bad)
    original = ops.load_supervisor(ROOT / 'scripts/guarded_panel_recovery.py')
    assert callable(original.supervise)


@pytest.mark.parametrize('gpu,pids', [(ops.GPU, ''), ('wrong GPU', ''), (ops.GPU, '123\n')])
def test_hardware_identity_and_busy_compute_are_checked(monkeypatch, gpu, pids):
    def probe(command, **_):
        output = pids if '--query-compute-apps=pid' in command else gpu
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')
    monkeypatch.setattr(ops.subprocess, 'run', probe)
    if gpu == ops.GPU and not pids: ops.hardware_gate()
    else:
        with pytest.raises(ValueError): ops.hardware_gate()


def test_command_uses_one_qualified_case_and_keeps_first_case_review(tmp_path):
    command = ops.build_command('/original/python', tmp_path, tmp_path / 'panel', tmp_path / 'vitl.pt', 'clip-01-bank-1')
    assert command[command.index('--case-id') + 1] == 'clip-01-bank-1'
    assert command.count('retry') == 1 and 'remaining' not in command
    assert '--acceptance' in command and '--diagnostic' in command


def test_production_entry_has_no_fixture_or_unbounded_execution_mode(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/attended_panel_case.py'),
        '--package', str(tmp_path), '--panel', str(tmp_path), '--request', str(tmp_path / 'missing'),
        '--expected-request-sha256', '0' * 64, '--previous-review', str(tmp_path / 'missing-review')],
        capture_output=True, timeout=15)
    assert result.returncode != 0
    assert not (tmp_path / 'operator-attended').exists()
