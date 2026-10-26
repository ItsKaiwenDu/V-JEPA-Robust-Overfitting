#!/usr/bin/env python3
"""One fresh attended case around the unchanged qualified scientific runner.

This is an operational envelope. It cannot retry an attempted case or dispatch
the remaining queue, and never reuses a recovery allowance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys

MANIFEST_SHA = '73f361fc072a053439bf71f35835bb91227932edd2db739adde2db5f96c6b0f4'
PLAN_SHA = 'f6b81dec841aeee703413dba23ea16afe3dbf14736edecadc0f7830f3ded34ae'
EXECUTION_SHA = '1e59a735e2f0b6f5485cf0867851c638dcd027715d19270422f390f7745aa70e'
SUPERVISOR_SHA = '231b7db4b10361d97f788eba09c80b349b3b18e2e6b1e4ed41698c288c885182'
HOST = 'linux10609'
GPU = 'NVIDIA GeForce RTX 3070, GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52, 8192, 580.178.04'
PACKAGE = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024/unpacked/matched-panel-20261024'
LIMIT = 4800


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_record(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != {'payload', 'sha256'} or canonical(value['payload']) != value['sha256']:
        raise ValueError('Record checksum mismatch: ' + str(path))
    return value['payload']


def bound(root, name):
    root = Path(root).resolve()
    relative = PurePosixPath(name)
    path = root / name
    if (not name or relative.is_absolute() or '..' in relative.parts or '\\' in name
            or str(relative) != name or name == '.' or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in [path, *list(path.parents)[:len(relative.parts)-1]])):
        raise ValueError('Unsafe evidence path: ' + name)
    return path


def validate_request(request, expected_sha256):
    if not isinstance(request, dict):
        raise ValueError('Use a structured one-case request.')
    if not isinstance(expected_sha256, str) or re.fullmatch('[a-f0-9]{64}', expected_sha256) is None:
        raise ValueError('Use the externally pinned request checksum.')
    if canonical(request) != expected_sha256:
        raise ValueError('Case request differs from the reviewed launch.')
    policy = {'kind': 'attended-panel-fresh-case-request-v1', 'one_case_only': True,
              'fresh_attempt_only': True, 'case_seconds': LIMIT, 'kill_grace_seconds': 15,
              'additional_allowance_seconds': 0, 'remaining_queue': False,
              'paid_compute': False, 'research_admitted': False}
    if any(k not in request or canonical(request[k]) != canonical(v) for k, v in policy.items()):
        raise ValueError('Use one fresh case within its original allowance only.')
    if not isinstance(request.get('case_id'), str) or re.fullmatch(r'clip-0[0-9]-bank-[01]', request['case_id']) is None:
        raise ValueError('Invalid case identity.')
    for name in ('operator_source_sha256', 'journal_file_sha256', 'previous_review_sha256'):
        if not isinstance(request.get(name), str) or re.fullmatch('[a-f0-9]{64}', request[name]) is None:
            raise ValueError('Missing launch identity: ' + name)
    prefix = request.get('completed_case_ids')
    if not isinstance(prefix, list) or not 1 <= len(prefix) < 20 or any(not isinstance(n, str) for n in prefix) or len(set(prefix)) != len(prefix):
        raise ValueError('A reviewed completed-case prefix is required.')
    return request['case_id']


def validate_boundary(journal, review, request, case_order, journal_sha256, review_sha256):
    prefix = request['completed_case_ids']
    if (journal_sha256 != request['journal_file_sha256'] or review_sha256 != request['previous_review_sha256']
            or prefix != case_order[:len(prefix)] or request['case_id'] != case_order[len(prefix)]
            or journal.get('case_order') != case_order or set(journal.get('completed', {})) != set(prefix)
            or journal.get('plan_sha256') != PLAN_SHA or journal.get('execution_sha256') != EXECUTION_SHA
            or journal.get('research_admitted') is not False or journal.get('first_acceptance') is None):
        raise ValueError('Journal or next-case boundary differs from independent review.')
    last = prefix[-1]
    pointer = journal['completed'][last]
    if (review.get('kind') not in ('matched-panel-recovery-review-acceptance-v1', 'matched-panel-attended-review-acceptance-v1')
            or review.get('reviewed_export') is not True or review.get('research_admitted') is not False
            or review.get('case_id') != last or type(review.get('completed_case_count')) is not int
            or review['completed_case_count'] != len(prefix) or review.get('journal_file_sha256') != journal_sha256
            or review.get('plan_sha256') != PLAN_SHA or review.get('execution_sha256') != EXECUTION_SHA
            or review.get('archive_sha256') != pointer['archive_sha256']
            or review.get('case_receipt_file_sha256') != pointer['receipt_file_sha256']
            or pointer['directory'] != 'cases/' + last + '/' + str(review.get('attempt'))):
        raise ValueError('Previous export has no matching independent review.')


def require_unattempted(panel, case_id):
    paths = ['cases/' + case_id, 'operator-attended/' + case_id,
             'exports/' + case_id + '-attempt-01.tar.gz', 'exports/' + case_id + '-attempt-01.tar.gz.manifest.json']
    if any((p := bound(panel, name)).exists() or p.is_symlink() for name in paths):
        raise ValueError('Case or operation already exists; preserve evidence and inspect, never reset time.')


def reserve_operation(panel, request):
    directory = bound(panel, 'operator-attended/' + request['case_id'])
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def load_supervisor(path):
    path = Path(path)
    if path.is_symlink() or digest(path) != SUPERVISOR_SHA:
        raise ValueError('Tested supervisor source changed.')
    spec = importlib.util.spec_from_file_location('pinned_attended_supervisor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(package, panel, request, previous_review):
    package, panel = Path(package).resolve(), Path(panel).resolve()
    if socket.gethostname() != HOST or package != (Path.home() / PACKAGE).resolve() or panel != package / 'runs/panel.qIXeDn':
        raise ValueError('Use the original university package and panel on linux10609.')
    if digest(__file__) != request['operator_source_sha256'] or digest(package / 'PANEL-HANDOFF.json') != MANIFEST_SHA:
        raise ValueError('Operator or qualified handoff identity changed.')
    manifest = json.loads((package / 'PANEL-HANDOFF.json').read_text())
    for name, pin in manifest['files'].items():
        path = bound(package, name)
        if not path.is_file() or path.stat().st_size != pin['bytes'] or digest(path) != pin['sha256']:
            raise ValueError('Qualified handoff changed: ' + name)
    journal = read_record(panel / 'journal.json')
    review = read_record(previous_review)
    plan = json.loads((package / 'configs/matched_panel_diagnostic_v1.json').read_text())
    order = [row['case_id'] for row in plan['cases']]
    validate_boundary(journal, review, request, order, digest(panel / 'journal.json'), digest(previous_review))
    require_unattempted(panel, request['case_id'])
    for cid, pointer in journal['completed'].items():
        directory = bound(panel, pointer['directory'])
        receipt_file = bound(directory, 'receipt.json')
        if digest(receipt_file) != pointer['receipt_file_sha256']:
            raise ValueError('Completed receipt changed: ' + cid)
        receipt = read_record(receipt_file)
        if receipt['case_id'] != cid or receipt['execution_sha256'] != EXECUTION_SHA:
            raise ValueError('Completed identity changed: ' + cid)
        if digest(bound(directory, 'session.json')) != receipt['session_sha256']:
            raise ValueError('Completed session changed: ' + cid)
        for name, pin in receipt['worker_files_sha256'].items():
            if digest(bound(directory, name)) != pin:
                raise ValueError('Completed worker artifact changed: ' + cid + '/' + name)
        for key in ('archive', 'archive_manifest'):
            if digest(bound(panel, pointer[key + '_path'])) != pointer[key + '_sha256']:
                raise ValueError('Completed archive changed: ' + cid)
    return journal


def hardware_gate():
    identity = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if identity != GPU:
        raise ValueError('Host GPU UUID/driver differs from the frozen reference.')
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if active:
        raise ValueError('Another CUDA compute process is active; stop before this case.')


COLLECTION_CHECK = '''import json,sys
from pathlib import Path
root,panel=map(Path,sys.argv[1:3]);sys.path.insert(0,str(root))
from models.matched_panel import PanelJournal
from models.matched_panel_io import load_plan,execution_identity
from models.study_artifacts import read_record
plan=load_plan(root);j=PanelJournal(panel,plan=plan,execution=execution_identity(root,plan))
prefix=json.loads(sys.argv[3]);cid,phase=sys.argv[4:6]
expected=prefix if phase=='before' else prefix+[cid]
progress=j.progress()
assert progress['completed_case_ids']==expected
assert progress['complete']==(len(expected)==len(plan['cases']))
assert progress['scientific_aggregate'] is None
if phase=='after':
    before=read_record(Path(sys.argv[6])/'before-journal.json');after=j.read()
    assert after['completed'][cid]['directory']=='cases/'+cid+'/attempt-01'
    for key,value in before.items():
        if key=='completed':
            for old,pointer in value.items():assert after[key][old]==pointer
        else:assert after[key]==value
print('Verified completed-case prefix:',len(expected),'cases; remaining queue is not started.',flush=True)
'''


def check_collection(python, package, panel, request, phase, operation=None):
    subprocess.run([str(python), '-B', '-c', COLLECTION_CHECK, str(package), str(panel),
        json.dumps(request['completed_case_ids']), request['case_id'], phase, str(operation or '')],
        check=True, timeout=180)


def build_command(python, package, panel, checkpoint, case_id):
    # The frozen CLI calls its single-case entry "retry". Freshness is enforced
    # above; it must never be used here for an existing/partial case.
    return [str(python), '-B', '-u', str(Path(package) / 'scripts/check_matched_panel.py'),
            'retry', '--diagnostic', '--case-id', case_id, '--output', str(panel),
            '--checkpoint', str(checkpoint), '--acceptance', str(Path(panel) / 'first-case-acceptance.json')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--expected-request-sha256', required=True)
    parser.add_argument('--previous-review', type=Path, required=True)
    args = parser.parse_args()
    request = read_record(args.request)
    cid = validate_request(request, args.expected_request_sha256)
    supervisor = load_supervisor(Path(__file__).with_name('guarded_panel_recovery.py'))
    before = preflight(args.package, args.panel, request, args.previous_review)
    hardware_gate()
    python = Path.home() / '.venvs/vjepa/bin/python'
    checkpoint = Path.home() / 'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/checkpoints/vitl.pt'
    if not python.is_file() or not checkpoint.is_file():
        raise ValueError('Use the existing qualified Python and released checkpoint.')
    check_collection(python, args.package, args.panel, request, 'before')
    require_unattempted(args.panel, cid)
    directory = reserve_operation(args.panel, request)
    supervisor.write_record(directory / 'request.json', request)
    supervisor.write_record(directory / 'previous-review.json', read_record(args.previous_review))
    supervisor.write_record(directory / 'before-journal.json', before)
    print('Starting one fresh attended case:', cid, flush=True)
    print('Status:', directory / 'status.json', flush=True)
    print('Worker log:', directory / 'operator.log', flush=True)
    state = supervisor.supervise(build_command(python, args.package, args.panel, checkpoint, cid), directory, LIMIT)
    state.update(operation_kind='attended-panel-fresh-case-v1', case_id=cid,
                 request_sha256=args.expected_request_sha256)
    if state.get('exit_status') != 0 or state['state'] != 'process_finished':
        supervisor.write_record(directory / 'status.json', state)
        return 1
    try:
        check_collection(python, args.package, args.panel, request, 'after', directory)
    except BaseException:
        state.update(state='verification_failed')
        supervisor.write_record(directory / 'status.json', state)
        raise
    state.update(state='case_verified', scientific_case_verified=True)
    supervisor.write_record(directory / 'status.json', state)
    print('One attended case finished. Download its export for independent review.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
