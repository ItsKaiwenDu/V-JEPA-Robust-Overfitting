"""One explicitly authorized Archery recovery; preserve the consumed attempt and all prior clocks."""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

KIND = 'matched-panel-archery-recovery-v1'
PREFIX = 'archery-recovery-20261027'
TARGET = 'archery-recovery-execution-20261027'
BASE = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
CORE_PREFIX = 'archery-timecarry-20261027'
CORE_TARGET = 'archery-timecarry-execution-20261027'
CORE_OP = '17a122f0c6d3c35c00b7a6dcf2037583466bf9498c996941a4f449cd50e470d4'
CORE_SOURCE = 'c27906c5bd383516d3259b8edc34ca798206776b6db7e89719a248850728a61c'
PRIOR_RECEIPT = '1bb93c739f3664bb4337b8d4333d5be01a59d5b54550e00a94c7be054bdb0f5e'
PROPOSAL = '9e8f177667c3ecc9e8d8c5c38b8bdb25b4ef1dfdad0ee16fa266aad490493126'
AUTH_FILE = '41141726758e31b4babf76feadfe48618b4399a5147ace7dcea614ea14feabc3'
AUTH_PAYLOAD = '830464d0b2df7ca18e87ff108c5ecfde4575cfe70967a86143f12946207c6c48'
GRANT_ID = 'fd4b4cb8d42e442f8d9ee8ac7130945e'
FAILED_STOP = 'b74da2014b8842c580a6161f740e075a47efffdef966649a9963bd90ba83a762'
FAILED_EXECUTION = 'd3ba72570b30f358dfe597f1e1bea809aa258f10bd8f92febfe406622da3afd2'
INCIDENT_REVIEW = '2a21957f3d4803f48beb92fa88c8371f9a6164482470d6f897bbbb8387bff2f6'
ACCOUNTING_PIN = '7e5c24277588c4cb251f805d97238046cb885e43639fa409f9cc067dbbab63f8'
ATTEMPT = 'recovery-01'
RP = '5ca7596fe95ecb8f6b6c6752426e56bb354b1f019d3576c6618f7ede98ae4116'
MP = 'da5aeb93bd6d3e46040b158cc14f0d6feb1284a83b2b2778a3639b420db0acf9'
REFERENCE = 'two-host-execution-20261026/unpacked/two-host-execution-20261026'
PACKAGE = 'unpacked/matched-panel-20261024'
LEGACY = 'two-host-panel-20261026'
STOP = '776e80c041599c1c1a35ee31348312b4f696eec03f0e1e8787cc27dd231b11f5'
ORDER = [f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0, 1)]
ACCEPTED = ORDER[:13] + ['clip-07-bank-0']
TARGETS = {'linux10603': 'clip-06-bank-1'}
CORE_FILES = ('CONTINUATION.json', 'INPUT.json', 'models/archery_continuation.py', 'prior-INPUT.json', 'prior-OPERATION.json', 'prior-receipt.json', 'scripts/continue_archery.py', 'scripts/prepare_archery_continuation.py', 'support/core-packet/CONTINUATION.json', 'support/core-packet/INPUT.json', 'support/core-packet/models/reviewed_panel_batch.py', 'support/core-packet/prior-INPUT.json', 'support/core-packet/prior-OPERATION.json', 'support/core-packet/prior-receipt.json', 'support/core-packet/scripts/continue_reviewed_batch.py', 'support/core-packet/scripts/prepare_reviewed_batch.py', 'support/core-packet/support/core-packet/CONTINUATION.json', 'support/core-packet/support/core-packet/models/reviewed_panel_continuation.py', 'support/core-packet/support/core-packet/prior-INPUT.json', 'support/core-packet/support/core-packet/prior-OPERATION.json', 'support/core-packet/support/core-packet/prior-receipt.json', 'support/core-packet/support/core-packet/scripts/continue_reviewed_panel.py', 'support/core-packet/support/core-packet/scripts/prepare_reviewed_continuation.py', 'support/core-packet/support/core-packet/support/reviewed_panel.py', 'support/core-packet/support/core-packet/tests/fixtures/reviewed_continuation_metadata.json', 'support/core-packet/support/core-packet/tests/test_reviewed_continuation.py', 'support/core-packet/support/reviewed_panel_sync.py', 'support/core-packet/tests/fixtures/reviewed_batch_metadata.json', 'support/core-packet/tests/test_reviewed_batch.py', 'support/yoyo_review_sync.py', 'tests/fixtures/archery_continuation_metadata.json', 'tests/test_archery_continuation.py')
SOURCES = ('models/archery_recovery.py', 'scripts/recover_archery.py',
    'scripts/prepare_archery_recovery.py', 'tests/test_archery_recovery.py',
    'tests/fixtures/archery_recovery_metadata.json') + tuple('support/core-packet/' + n for n in CORE_FILES)
METADATA = ('PROPOSAL.json', 'authorization.json', 'incident-verification.json',
            'incident-accounting.json', 'failed-numeric.manifest.json')
RECORD_SUPPORT = 'support/core-packet/support/core-packet/support/core-packet/support/reviewed_panel.py'
HARDWARE_STAGES = tuple(role + '-' + stage for role in ('matrix', 'replay')
                        for stage in ('before', 'probe', 'after'))
ORIGINAL_LIMIT = 4800
CARRIED = 2348
OLD_REMAINING = 2452
LIMIT = 4800
PRESERVED = 470


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_operation(op, pin):
    fixed = {'kind': KIND, 'core_operation_sha256': CORE_OP, 'core_module_sha256': CORE_SOURCE,
        'prior_activation_receipt_file_sha256': PRIOR_RECEIPT, 'proposal_sha256': PROPOSAL,
        'authorization_file_sha256': AUTH_FILE, 'grant_id': GRANT_ID,
        'reference_revision_sha256': RP, 'reference_implementation_sha256': MP,
        'imported_case_ids': ACCEPTED, 'first_requests': TARGETS, 'coordinator_namespace': TARGET,
        'historical_STOP_sha256': STOP, 'failed_STOP_file_sha256': FAILED_STOP,
        'all_existing_STOP_records_preserved': True, 'numeric_settings_changed': False,
        'case_seconds': LIMIT, 'previous_charged_seconds': CARRIED, 'old_remaining_seconds': OLD_REMAINING,
        'old_remainder_added': False, 'prior_numeric_attempts': 1, 'additional_numeric_attempts': 1,
        'kill_grace_seconds': 15, 'matrix_seconds': 3600, 'replay_seconds': 900, 'parent_seconds': 4620,
        'max_attempts': 1, 'extra_case_seconds': 4800, 'automatic_GPU_retries': 0,
        'max_parallel_cases': 1, 'one_case_per_host': True, 'automatic_queue': False,
        'paid_compute': False, 'research_admitted': False, 'preserved_file_count': PRESERVED}
    if canonical(op) != pin or any(k not in op or canonical(op[k]) != canonical(v) for k, v in fixed.items()):
        raise ValueError('Use the pinned, explicitly authorized one-attempt recovery; no clock reset or extra queue.')
    if set(op['sources_sha256']) != set(SOURCES) or op['sources_sha256']['support/core-packet/models/archery_continuation.py'] != CORE_SOURCE:
        raise ValueError('Incomplete or changed recovery source inventory.')


def validate_authorization(value):
    if (canonical(value) != AUTH_PAYLOAD or value.get('approved') is not True
            or value['proposal_sha256'] != PROPOSAL or value['grant_id'] != GRANT_ID
            or value['case_id'] != 'clip-06-bank-1' or value['host'] != 'linux10603'
            or value['additional_attempts'] != 1 or value['extra_case_seconds'] != 4800
            or value['old_remainder_added'] is not False or value['strict_GPU_guard_unchanged'] is not True):
        raise ValueError('Only the actual user authorization for this proposal and one new grant is valid.')
    return value


def expected_input(old_data, proposal, authorization, failed_manifest):
    validate_authorization(authorization)
    if canonical(proposal) != PROPOSAL:
        raise ValueError('Original reviewed recovery proposal changed.')
    if (proposal['approved'] is not False or proposal['preserved_file_count'] != 465
            or proposal['past_time_accounting']['total_original_allowance_charged_seconds'] != CARRIED
            or proposal['past_time_accounting']['original_allowance_remaining_seconds'] != OLD_REMAINING
            or proposal['rejected_execution_sha256'] != FAILED_EXECUTION):
        raise ValueError('Preserve the original inert proposal, failed execution and consumed clock.')
    pins = dict(proposal['preserved_files_sha256'])
    for n, h in old_data['preserved_files_sha256'].items():
        if pins.get(n) != h:
            raise ValueError('Historical preservation pin changed.')
    case = CORE_TARGET + '/cases/clip-06-bank-1/attempt-01/'
    if (pins.get(CORE_TARGET + '/STOP.json') != FAILED_STOP
            or pins.get(CORE_TARGET + '/outcomes/clip-06-bank-1.json') != FAILED_STOP
            or failed_manifest['sha256'] != proposal['failed_numeric_archive_sha256']):
        raise ValueError('Failed STOP, outcome or retained archive changed.')
    for n in ('inputs.pt', 'A.pt', 'B.pt', 'C.pt', 'D.pt'):
        pins[case + 'matrix/' + n] = failed_manifest['files']['matrix/' + n]['sha256']
    if len(pins) != PRESERVED or set(old_data['case_index']) != set(ACCEPTED):
        raise ValueError('Preserve all fourteen original origins, 465 proposal pins and five raw tensor files.')
    return {'kind': KIND + '-input', 'proposal_sha256': PROPOSAL, 'authorization_file_sha256': AUTH_FILE,
        'case_index': old_data['case_index'], 'preserved_files_sha256': pins,
        'past_time_accounting': proposal['past_time_accounting'], 'models_started': False, 'research_admitted': False}


def boundary(packet, pin):
    packet = Path(packet)
    op = json.loads((packet / 'CONTINUATION.json').read_text())
    validate_operation(op, pin)
    for name, h in op['sources_sha256'].items():
        if digest(packet / name) != h:
            raise ValueError('Frozen recovery source changed: ' + name)
    core_packet = packet / 'support/core-packet'
    core = load_module('qualified_failed_archery', core_packet / 'models/archery_continuation.py')
    old = core.boundary(core_packet, CORE_OP)
    if ((packet / 'prior-OPERATION.json').read_bytes() != (core_packet / 'CONTINUATION.json').read_bytes()
            or (packet / 'prior-INPUT.json').read_bytes() != (core_packet / 'INPUT.json').read_bytes()
            or digest(packet / 'prior-receipt.json') != PRIOR_RECEIPT):
        raise ValueError('Actual failed-attempt activation boundary changed.')
    prior = load_module('original_recovery_record_bytes', packet / RECORD_SUPPORT)
    receipt = json.loads((packet / 'prior-receipt.json').read_text())
    core.verify_activation(receipt, old['op'], old['data'], prior)
    proposal = json.loads((packet / 'PROPOSAL.json').read_text())
    authorization = validate_authorization(json.loads((packet / 'authorization.json').read_text()))
    if (digest(packet / 'authorization.json') != AUTH_FILE or digest(packet / 'incident-verification.json') != INCIDENT_REVIEW
            or digest(packet / 'incident-accounting.json') != ACCOUNTING_PIN):
        raise ValueError('Actual approval or independent incident/accounting bytes changed.')
    accounting = json.loads((packet / 'incident-accounting.json').read_text())
    incident = json.loads((packet / 'incident-verification.json').read_text())
    if (accounting != proposal['past_time_accounting'] or incident['accounting'] != accounting
            or incident['verified'] is not True or incident['complete_worker_or_case_accepted'] is not False
            or incident['new_execution_sha256'] != FAILED_EXECUTION):
        raise ValueError('Original rejected case must stay rejected with its full charged time.')
    case = CORE_TARGET + '/cases/clip-06-bank-1/attempt-01/'
    if digest(packet / 'failed-numeric.manifest.json') != proposal['preserved_files_sha256'][case + 'numeric-evidence.tar.gz.manifest.json']:
        raise ValueError('Actual partial-export sidecar changed.')
    data = expected_input(old['data'], proposal, authorization, json.loads((packet / 'failed-numeric.manifest.json').read_text()))
    if digest(packet / 'INPUT.json') != op['inputs_file_sha256'] or json.loads((packet / 'INPUT.json').read_text()) != data:
        raise ValueError('Recovery import, approval or preservation map changed.')
    return {'packet': packet, 'op': op, 'data': data, 'core': core, 'core_packet': core_packet,
        'authorization': authorization, 'proposal': proposal, 'prior': prior,
        'sync_op': old['op'], 'sync_input': old['data'], 'sync_receipt': receipt}


def trusted(packet, pin, package, reference):
    c = boundary(packet, pin)
    previous = c['core'].trusted(c['core_packet'], CORE_OP, package, reference)
    for name in ('prior', 'ops', 'revision', 'manifest', 'numeric', 'old', 'package', 'reference'):
        c[name] = previous[name]
    c['previous'] = previous
    return c


def load(packet, pin, home):
    transfer = (Path(home) / BASE).resolve()
    packet = Path(packet).resolve()
    if packet != (transfer / PREFIX / 'unpacked' / PREFIX).resolve():
        raise ValueError('Use the one fixed recovery namespace; no copied-root or time reset.')
    c = trusted(packet, pin, transfer / PACKAGE, transfer / REFERENCE)
    c.update(transfer=transfer, root=transfer / TARGET, legacy=transfer / LEGACY)
    old_packet = transfer / CORE_PREFIX / 'unpacked' / CORE_PREFIX
    c['previous'] = c['core'].load(old_packet, CORE_OP, home)
    if any(digest(old_packet / n) != digest(c['core_packet'] / n) for n in CORE_FILES):
        raise ValueError('Installed failed-attempt source packet changed.')
    check_ancestors(c)
    return c


def ancestor_pins(context):
    pins = dict(context['data']['preserved_files_sha256'])
    if len(pins) != PRESERVED:
        raise ValueError('Preserve all 470 files, including failed tensors, identities and consumed clocks.')
    return pins


def check_ancestors(context):
    context['core'].check_ancestors(context['previous'])
    registry = context['core'].Registry(context['previous'])
    with context['ops'].file_lock(context['previous']['root'] / '.registry.lock'):
        expected = {'kind': context['core'].KIND + '-progress', 'accepted_case_ids': ACCEPTED,
            'claimed_case_ids': ['clip-06-bank-1'], 'computed_case_ids': [], 'failed_case_ids': ['clip-06-bank-1'],
            'stop_dispatch': True, 'complete': False, 'scientific_aggregate': None, 'research_admitted': False}
        if registry.status() != expected:
            raise ValueError('The prior failed coordinator changed; preserve the consumed claim and STOP.')
        return context['prior'].check_preserved(context['transfer'], ancestor_pins(context))


def recovery_grant():
    return {'kind': KIND + '-grant', 'proposal_sha256': PROPOSAL, 'authorization_file_sha256': AUTH_FILE,
        'grant_id': GRANT_ID, 'case_id': 'clip-06-bank-1', 'host': 'linux10603', 'attempt': ATTEMPT,
        'previous_charged_seconds': CARRIED, 'old_remaining_seconds': OLD_REMAINING,
        'fresh_attempt_seconds': LIMIT, 'old_remainder_added': False, 'additional_numeric_attempts': 1,
        'prior_numeric_attempts': 1, 'failed_STOP_file_sha256': FAILED_STOP,
        'old_claim_and_STOP_preserved': True, 'fresh_matrix_and_replay': True,
        'strict_GPU_guard_unchanged': True, 'paid_compute': False, 'research_admitted': False}


def activation_records(op, data, prior):
    imports = {'kind': KIND + '-imports', 'operation_sha256': canonical(op), 'case_origins': data['case_index'],
        'prior_receipt_file_sha256': PRIOR_RECEIPT, 'original_execution_identities_preserved': True, 'research_admitted': False}
    resolution = {'kind': KIND + '-resolution', 'operation_sha256': canonical(op),
        'proposal_sha256': PROPOSAL, 'authorization_file_sha256': AUTH_FILE, 'grant_id': GRANT_ID,
        'historical_STOP_sha256': STOP, 'failed_STOP_file_sha256': FAILED_STOP,
        'historical_STOP_records_deleted': False, 'reason': 'Explicitly approved one fresh recovery attempt. Preserve the failed claim, STOP, all previous time and fourteen origins. Never reuse the partial matrix.',
        'first_requests': TARGETS, 'automatic_GPU_retries': 0, 'research_admitted': False}
    grant = recovery_grant()
    genesis = {'kind': KIND + '-genesis', 'operation_sha256': canonical(op),
        'imports_sha256': canonical(imports), 'resolution_sha256': canonical(resolution),
        'recovery_grant_sha256': canonical(grant), 'prior_receipt_file_sha256': PRIOR_RECEIPT, 'research_admitted': False}
    values = {'imports.json': imports, 'resolution.json': resolution, 'recovery-grant.json': grant, 'genesis.json': genesis}
    values.update({'requests/' + cid + '.json': request(op, host) for host, cid in TARGETS.items()})
    return {n: prior.record_bytes(v) for n, v in values.items()}


def verify_activation(receipt, op, data, prior):
    validate_operation(op, canonical(op))
    expected = activation_records(op, data, prior)
    pins = ancestor_pins({'data': data})
    if (receipt['kind'] != KIND + '-activation-receipt' or receipt['operation_sha256'] != canonical(op)
            or receipt['prior_receipt_file_sha256'] != PRIOR_RECEIPT
            or set(receipt['created']) & set(receipt['preexisting_identical'])
            or set(receipt['created']) | set(receipt['preexisting_identical']) != set(expected)
            or len(receipt['created']) + len(receipt['preexisting_identical']) != len(expected)
            or receipt['record_texts'] != {n: v.decode() for n, v in expected.items()}
            or receipt['preserved_files_sha256'] != pins or receipt['accepted_case_ids'] != ACCEPTED
            or receipt['next_case_ids'] != TARGETS or receipt['both_historical_STOP_records_unchanged'] is not True
            or any(receipt[k] is not False for k in ('models_started', 'automatic_queue', 'research_admitted'))):
        raise ValueError('Activation differs from five exact records and the fourteen-case boundary.')
    return {'verified': True, 'accepted_cases': 14, 'previous_charged_seconds': CARRIED, 'fresh_attempt_seconds': LIMIT, 'grant_id': GRANT_ID, 'new_requests': TARGETS,
        'preserved_files': PRESERVED, 'historical_STOP_records_unchanged': True, 'models_started': False}


def request(op, host):
    if host not in TARGETS:
        raise ValueError('Unknown assigned recovery host.')
    return {'kind': KIND + '-request', 'operation_sha256': canonical(op), 'host': host,
        'case_id': TARGETS[host], 'case_seconds': LIMIT, 'kill_grace_seconds': 15,
        'previous_charged_seconds': CARRIED, 'old_remaining_seconds': OLD_REMAINING,
        'recovery_grant_sha256': canonical(recovery_grant()), 'authorization_file_sha256': AUTH_FILE,
        'grant_id': GRANT_ID, 'prior_numeric_attempts': 1, 'additional_numeric_attempts': 1,
        'attempt': ATTEMPT, 'max_attempts': 1, 'extra_case_seconds': 4800, 'old_remainder_added': False,
        'one_case_only': True, 'automatic_queue': False, 'paid_compute': False, 'research_admitted': False}


def validate_request(value, pin, op, host):
    if canonical(value) != pin or value != request(op, host):
        raise ValueError('Wrong or altered fresh-case request.')
    return value['case_id']


def elapsed(started, now=None):
    if type(started) not in (int, float) or not math.isfinite(started):
        raise ValueError('Unknown launcher start; no allowance may be reset.')
    spent = (time.time() if now is None else now) - started
    if not math.isfinite(spent) or not 0 <= spent < LIMIT:
        raise TimeoutError('Invalid or exhausted original launcher allowance.')
    return spent


def publish_activation(context):
    """Both old host leases held by caller; publication never alters ancestor STOPs."""
    prior, ops, root = context['prior'], context['ops'], context['root']
    targets = activation_records(context['op'], context['data'], prior)
    if root.is_symlink():
        raise ValueError('Continuation coordinator is redirected.')
    root.mkdir(exist_ok=True)
    with ops.file_lock(root / '.registry.lock'):
        for path in root.rglob('*'):
            name = str(path.relative_to(root))
            if path.is_symlink() or (path.is_dir() and name != 'requests') or (path.is_file() and name not in set(targets) | {'.registry.lock'}):
                raise ValueError('Unknown continuation state; never initialize over an attempt.')
        for name, content in targets.items():
            path = prior.bound(root, name)
            if path.exists() and path.read_bytes() != content:
                raise ValueError('Conflicting continuation activation record.')
        check_ancestors(context)
        created, existing = [], []
        for name in sorted(set(targets) - {'genesis.json'}) + ['genesis.json']:
            path = prior.bound(root, name)
            if path.exists():
                existing.append(name)
            else:
                prior.exclusive_bytes(path, targets[name])
                created.append(name)
        observed = check_ancestors(context)
        readback = {name: prior.bound(root, name).read_text() for name in targets}
    return {'kind': KIND + '-activation-receipt', 'operation_sha256': canonical(context['op']),
        'prior_receipt_file_sha256': PRIOR_RECEIPT, 'created': created, 'preexisting_identical': existing,
        'record_texts': readback, 'preserved_files_sha256': observed,
        'accepted_case_ids': ACCEPTED, 'next_case_ids': TARGETS,
        'both_historical_STOP_records_unchanged': True, 'models_started': False,
        'automatic_queue': False, 'research_admitted': False}


def activate(context, host):
    if host != 'linux10609':
        raise ValueError('Use the metadata activation host.')
    with ExitStack() as stack:
        for assigned in context['ops'].HOSTS:
            stack.enter_context(context['ops'].file_lock(context['legacy'] / 'leases' / assigned, wait_seconds=0))
        result = publish_activation(context)
    verify_activation(result, context['op'], context['data'], context['prior'])
    return result


class Registry:
    """A new append-only ledger imports fourteen original identities and never rewrites the ancestors."""
    def __init__(self, context):
        self.context = context
        if context['root'].is_symlink():
            raise ValueError('Continuation root is redirected.')
        self.root = context['root'].resolve()
        self.op, self.ops, self.prior = context['op'], context['ops'], context['prior']
        self.targets = activation_records(self.op, context['data'], self.prior)
        self.check_metadata()

    def check_metadata(self):
        for name, content in self.targets.items():
            if self.prior.bound(self.root, name).read_bytes() != content:
                raise ValueError('Activated continuation metadata changed.')
        allowed = {'recovery-grant.json', 'genesis.json', 'imports.json', 'resolution.json', 'requests', '.registry.lock',
                   'claims', 'cases', 'outcomes', 'exports', 'reviews', 'failures', 'STOP.json'}
        if any(p.name not in allowed or p.is_symlink() for p in self.root.iterdir()):
            raise ValueError('Unknown or redirected coordinator state.')
        for folder in ('requests', 'claims', 'cases', 'outcomes', 'exports', 'reviews'):
            path = self.prior.bound(self.root, folder)
            names = set(TARGETS.values()) if folder in ('claims', 'cases') else {c + '.json' for c in TARGETS.values()}
            if path.exists() and any(p.name not in names or p.is_symlink() for p in path.iterdir()):
                raise ValueError('Unknown continuation case state.')

    def status(self):
        self.check_metadata()
        accepted, claimed, computed, failed = set(ACCEPTED), [], [], []
        for host, cid in TARGETS.items():
            claim_path = self.prior.bound(self.root, 'claims/' + cid + '/claim.json')
            if claim_path.parent.exists():
                if not claim_path.is_file() or self.ops.read_record(claim_path) != self.expected_claim(host):
                    raise ValueError('Unknown or mixed case claim; no allowance reset.')
                claimed.append(cid)
            outcome_path = self.prior.bound(self.root, 'outcomes/' + cid + '.json')
            if outcome_path.exists():
                body = self.ops.read_record(outcome_path)
                if (cid not in claimed or body['kind'] != KIND + '-outcome' or body['case_id'] != cid
                        or body['host'] != host or body['operation_sha256'] != canonical(self.op)
                        or type(body['complete']) is not bool or body['research_admitted'] is not False):
                    raise ValueError('Invalid or unowned continuation outcome.')
                (computed if body['complete'] else failed).append(cid)
            marker_path = self.prior.bound(self.root, 'reviews/' + cid + '.json')
            if marker_path.exists():
                marker = self.ops.read_record(marker_path)
                export = self.ops.read_record(self.prior.bound(self.root, 'exports/' + cid + '.json'))
                if (cid not in computed or marker['kind'] != KIND + '-independent-review'
                        or marker['operation_sha256'] != canonical(self.op) or marker['case_id'] != cid
                        or marker['host'] != host or marker['reviewed_export'] is not True
                        or marker['outcome_file_sha256'] != digest(outcome_path)
                        or marker['archive_sha256'] != export['archive_sha256']
                        or marker['numeric_receipt_sha256'] != body['numeric_receipt_sha256']
                        or marker['execution_sha256'] != body['execution_sha256'] or marker['research_admitted'] is not False):
                    raise ValueError('Review does not match its actual outcome and export.')
                accepted.add(cid)
        return {'kind': KIND + '-progress', 'accepted_case_ids': [c for c in ORDER if c in accepted],
            'claimed_case_ids': [c for c in ORDER if c in claimed], 'computed_case_ids': [c for c in ORDER if c in computed],
            'failed_case_ids': [c for c in ORDER if c in failed], 'stop_dispatch': (self.root / 'STOP.json').exists(),
            'complete': len(accepted) == 20, 'scientific_aggregate': None, 'research_admitted': False}

    def expected_claim(self, host):
        return claim_body(self.op, host)

    def claim(self, host):
        cid = request(self.op, host)['case_id']
        check_ancestors(self.context)
        with self.ops.file_lock(self.root / '.registry.lock'):
            state = self.status()
            if state['stop_dispatch'] or cid in state['claimed_case_ids'] or cid in state['accepted_case_ids']:
                raise ValueError('Stopped, completed or recovery grant already consumed; no retry.')
            candidates = [self.root / n / cid for n in ('claims', 'cases')]
            candidates += [self.root / n / (cid + '.json') for n in ('outcomes', 'exports', 'reviews', 'failures')]
            if any(p.exists() or p.is_symlink() for p in candidates):
                raise ValueError('This recovery already has pending artifacts; never reset it.')
            path = self.prior.bound(self.root, 'claims/' + cid)
            path.mkdir(parents=True, exist_ok=False)
            self.ops.write_record(path / 'claim.json', self.expected_claim(host))
            return self.expected_claim(host)

    def fail(self, host, error, spent):
        cid = TARGETS[host]
        body = {'kind': KIND + '-outcome', 'operation_sha256': canonical(self.op),
            'case_id': cid, 'host': host, 'complete': False, 'error': str(error),
            'launcher_elapsed_seconds': spent, 'prior_plus_new_elapsed_seconds': CARRIED + spent,
            'resource_grant': recovery_grant(), 'research_admitted': False}
        with self.ops.file_lock(self.root / '.registry.lock'):
            path = self.root / 'outcomes' / (cid + '.json')
            if not path.exists():
                self.ops.write_record(path, body)
            elif self.ops.read_record(path)['complete']:
                self.ops.write_record(self.root / 'failures' / (cid + '.json'), body)
            if not (self.root / 'STOP.json').exists():
                self.ops.write_record(self.root / 'STOP.json', body)
        return body


def execution_identity(context, host):
    result = context['ops'].execution_identity(context['revision'], context['manifest'], host)
    reference = canonical(result)
    result.update(kind=KIND + '-numeric-execution', origin='explicit-Archery-additional-attempt',
        reference_execution_sha256=reference, prior_rejected_execution_sha256=FAILED_EXECUTION,
        continuation_operation_sha256=canonical(context['op']), continuation_sources_sha256=context['op']['sources_sha256'],
        prior_activation_receipt_file_sha256=PRIOR_RECEIPT, resource_grant=recovery_grant())
    return result


def claim_body(op, host):
    return {'kind': KIND + '-claim', 'operation_sha256': canonical(op),
        'request_sha256': canonical(request(op, host)), 'case_id': TARGETS[host], 'host': host,
        'attempt': ATTEMPT, 'case_seconds': LIMIT, 'max_attempts': 1,
        'grant_id': GRANT_ID, 'authorization_file_sha256': AUTH_FILE, 'research_admitted': False}


def strict_gate(context, host, directory, stage, *, allow_self=False):
    ops = context['ops']
    event = {'kind': KIND + '-GPU-check', 'host': host, 'stage': stage,
        'utc': datetime.now(timezone.utc).isoformat(), 'strict_guard_source_unchanged': True,
        'snapshot_is_not_a_reservation': True, 'research_admitted': False}
    try:
        identity = ops.hardware_gate(host, allow_self=allow_self)
        event.update(passed=True, GPU_identity=identity)
        return identity
    except Exception as error:
        event.update(passed=False, error=type(error).__name__ + ': ' + str(error))
        for label, command in [('nvidia_smi', ['nvidia-smi']), ('compute_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])]:
            try:
                snapshot = subprocess.run(command, capture_output=True, text=True, timeout=10)
                event[label] = {'exit_status': snapshot.returncode, 'stdout': snapshot.stdout, 'stderr': snapshot.stderr}
            except Exception as query_error:
                event[label] = {'query_error': type(query_error).__name__ + ': ' + str(query_error)}
        raise
    finally:
        ops.write_record(directory / ('hardware-' + stage + '.json'), event)


def worker(context, host, role):
    if host not in TARGETS or platform.node() != host or role not in ('matrix', 'replay'):
        raise ValueError('Wrong assigned worker host or role.')
    registry = Registry(context)
    cid = TARGETS[host]
    directory = context['prior'].bound(registry.root, 'cases/' + cid + '/' + ATTEMPT)
    claim = context['ops'].read_record(registry.root / 'claims' / cid / 'claim.json')
    control = context['ops'].read_record(directory / 'case-envelope.json')
    if (claim != registry.expected_claim(host) or control['operation_sha256'] != canonical(context['op'])
            or control['active_role'] != role or control['complete'] is not False):
        raise ValueError('Worker requires the live matching role and single-use claim.')
    check_ancestors(context)
    (registry.root / 'claims' / cid / 'workers' / role).mkdir(parents=True, exist_ok=False)
    strict_gate(context, host, directory, role + '-before')
    import numpy as np
    import torch
    from models.evaluation import EvaluationMaskSampler
    from models.execution import apply_model_policy, configure_execution
    from models.matched_panel_case import KIND as NUMERIC_KIND, run_worker
    from models.matched_panel_io import load_clip
    from models.precision_diagnostic import native_probe
    from models.runtime import MemoryMonitor
    from models.training import runtime_contract
    from models.vjepa2 import build_vitl, load_checkpoint
    execution = execution_identity(context, host)
    expected = execution['expected_runtime']
    configure_execution('shared-fp64-native', device='cuda')
    torch.set_num_threads(expected['threads'])
    if (platform.python_version() != expected['python'] or str(torch.__version__) != expected['torch']
            or np.__version__ != expected['numpy'] or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError('Actual worker runtime differs from the frozen reference.')
    total = torch.cuda.get_device_properties(0).total_memory
    allocator = min(7 * 2**30, int(.9 * total))
    torch.cuda.set_per_process_memory_fraction(min(.9, 7 * 2**30 / total))
    probe = native_probe('cuda')
    strict_gate(context, host, directory, role + '-probe', allow_self=True)
    torch.manual_seed(20261024)
    package = context['package']
    clean = load_clip(package, context['numeric'], cid)
    model = build_vitl(package / 'external/vjepa2', activation_checkpointing=True)
    checkpoint = Path.home() / 'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/checkpoints/vitl.pt'
    load_checkpoint(model, checkpoint, execution['checkpoint_sha256'])
    apply_model_policy(model, 'shared-fp64-native')
    model.to('cuda')
    if canonical(runtime_contract('cuda', model=model)) != canonical(expected):
        raise ValueError('Actual model/runtime policy changed.')
    with MemoryMonitor('cuda') as memory:
        run_worker(model, clean.to('cuda'), EvaluationMaskSampler(package / 'external/vjepa2'),
            plan=context['numeric'], case_id=cid, execution=execution, output=directory / role, role=role)
    identity = strict_gate(context, host, directory, role + '-after', allow_self=True)
    context['ops'].write_record(directory / role / 'observation.json', {
        'kind': NUMERIC_KIND, 'case_id': cid, 'role': role, 'execution_sha256': canonical(execution),
        'actual_host': platform.node(), 'actual_GPU_identity': identity, 'memory': memory.report(),
        'allocator_limit_bytes': allocator, 'native_probe': probe, 'origin': KIND,
        'research_admitted': False, 'A40_cost_pool_eligible': False})


def numerical_case(context, host, directory, budget):
    from models.matched_panel_case import KIND as NUMERIC_KIND, verify_case, verify_pair
    from models.matched_panel_io import export_case, verify_archive
    ops, cid = context['ops'], TARGETS[host]
    execution = execution_identity(context, host)
    started = time.monotonic()
    session = {'kind': NUMERIC_KIND, 'case_id': cid, 'plan_sha256': ops.NUMERIC_PLAN_SHA,
        'execution_sha256': canonical(execution), 'complete': False, 'passed': False,
        'research_admitted': False, 'tasks': {}, 'elapsed_seconds': 0.0,
        'started_utc': datetime.now(timezone.utc).isoformat()}
    ops.write_record(directory / 'session.json', session)
    control = ops.read_record(directory / 'case-envelope.json')
    old_handler = signal.getsignal(signal.SIGTERM)
    def stop(*_):
        raise TimeoutError('Case terminated; retain partial evidence and consumed claim.')
    signal.signal(signal.SIGTERM, stop)
    try:
        for role, cap in (('matrix', 3600), ('replay', 900)):
            remaining = budget - (time.monotonic() - started) - 30
            if remaining < 1:
                raise TimeoutError('No time remains for the required fresh worker.')
            control['active_role'] = role
            ops.write_record(directory / 'case-envelope.json', control, replace=True)
            command = [sys.executable, '-B', '-u', str(context['packet'] / 'scripts/recover_archery.py'),
                'worker', '--operation-sha256', canonical(context['op']), '--host', host, '--role', role]
            print(host, cid, 'starting', role, 'worker', flush=True)
            session['tasks'][role] = ops.logged_child(command, directory / (role + '.log'), min(cap, remaining))
            session['elapsed_seconds'] = time.monotonic() - started
            ops.write_record(directory / 'session.json', session, replace=True)
            if not session['tasks'][role]['complete']:
                raise RuntimeError('Worker failed or timed out; no retry or next case.')
        verify_pair(directory, plan=context['numeric'], case_id=cid, execution=execution)
        session.update(complete=True, passed=True, elapsed_seconds=time.monotonic() - started)
        ops.write_record(directory / 'session.json', session, replace=True)
        receipt = verify_case(directory, plan=context['numeric'], case_id=cid, execution=execution)
        ops.write_record(directory / 'receipt.json', receipt)
    except BaseException as error:
        session.update(complete=False, passed=False, error=type(error).__name__ + ': ' + str(error),
                       elapsed_seconds=time.monotonic() - started)
        ops.write_record(directory / 'session.json', session, replace=True)
        raise
    finally:
        control['active_role'] = None
        ops.write_record(directory / 'case-envelope.json', control, replace=True)
        try:
            archive = export_case(context['package'], directory, context['numeric'], execution,
                                  destination=directory / 'numeric-evidence.tar.gz')
            verify_archive(directory / 'numeric-evidence.tar.gz', archive, archive['sha256'])
        finally:
            signal.signal(signal.SIGTERM, old_handler)
    return receipt, archive


def outer_files(context, host, directory):
    packet, root = context['packet'], context['root']
    files = {'CONTINUATION.json': packet / 'CONTINUATION.json', 'INPUT.json': packet / 'INPUT.json',
        'prior-OPERATION.json': packet / 'prior-OPERATION.json', 'prior-INPUT.json': packet / 'prior-INPUT.json',
        'prior-receipt.json': packet / 'prior-receipt.json', 'reference-implementation.json': context['reference'] / 'IMPLEMENTATION.json',
        'case-request.json': root / 'requests' / (TARGETS[host] + '.json'),
        'case-claim.json': root / 'claims' / TARGETS[host] / 'claim.json',
        'case-outcome.json': root / 'outcomes' / (TARGETS[host] + '.json'),
        'case-envelope.json': directory / 'case-envelope.json', 'ledger-genesis.json': root / 'genesis.json',
        'ledger-imports.json': root / 'imports.json', 'resolution.json': root / 'resolution.json', 'recovery-grant.json': root / 'recovery-grant.json',
        'preservation-before-export.json': directory / 'preservation-before-export.json'}
    files.update({n: packet / n for n in METADATA})
    files.update({'source/' + n: packet / n for n in SOURCES})
    files.update({'reference-source/' + n: context['reference'] / n for n in context['ops'].SOURCES})
    files.update({'hardware/hardware-' + stage + '.json': directory / ('hardware-' + stage + '.json') for stage in HARDWARE_STAGES})
    for name in ('numeric-evidence.tar.gz', 'numeric-evidence.tar.gz.manifest.json'):
        if (directory / name).is_file():
            files['numeric/' + name] = directory / name
    return files


def export_case(context, host, directory):
    from models.pilot_bundle import make_archive
    files = outer_files(context, host, directory)
    if any(p.is_symlink() for p in files.values()) or sum(p.stat().st_size for p in files.values()) > 256 * 2**20:
        raise ValueError('Outer evidence exceeds the fixed regular-file bound.')
    path = directory / 'continuation-evidence.tar.gz'
    sidecar = make_archive(path, files)
    Path(str(path) + '.manifest.json').write_text(json.dumps(sidecar, sort_keys=True, indent=2) + '\n')
    event = {'kind': KIND + '-export', 'operation_sha256': canonical(context['op']), 'case_id': TARGETS[host],
        'archive_path': str(path.relative_to(context['root'])), 'archive_sha256': sidecar['sha256'],
        'archive_manifest_path': str(Path(str(path) + '.manifest.json').relative_to(context['root'])),
        'archive_manifest_sha256': digest(Path(str(path) + '.manifest.json')),
        'bytes': sidecar['bytes'], 'members': len(sidecar['files'])}
    context['ops'].write_record(context['root'] / 'exports' / (TARGETS[host] + '.json'), event)
    print('Continuation export:', json.dumps(event, sort_keys=True), flush=True)
    return event


def run(context, host, launcher_started):
    if host not in TARGETS or platform.node() != host:
        raise ValueError('Use the assigned workstation.')
    spent = elapsed(launcher_started)
    ops, root, cid = context['ops'], context['root'], TARGETS[host]
    from models.matched_panel_runner import case_deadline
    with case_deadline(LIMIT - spent, kill_grace_seconds=15):
        with ops.file_lock(context['legacy'] / 'leases' / host, wait_seconds=0):
            check_ancestors(context)
            registry = Registry(context)
            ops.hardware_gate(host)
            if __import__('shutil').disk_usage(root).free < 4 * 2**30:
                raise OSError('Insufficient artifact/export space.')
            registry.claim(host)
            directory = context['prior'].bound(root, 'cases/' + cid + '/' + ATTEMPT)
            directory.mkdir(parents=True, exist_ok=False)
            control = {'kind': KIND + '-case-envelope', 'operation_sha256': canonical(context['op']),
                'case_id': cid, 'host': host, 'complete': False, 'active_role': None,
                'case_seconds': LIMIT, 'resource_grant': recovery_grant(), 'launcher_started_epoch': launcher_started,
                'launcher_elapsed_seconds': elapsed(launcher_started), 'research_admitted': False}
            ops.write_record(directory / 'case-envelope.json', control)
            try:
                budget = min(4620, LIMIT - elapsed(launcher_started) - 120)
                if budget < 1:
                    raise TimeoutError('Preflight consumed the original case allowance.')
                receipt, inner = numerical_case(context, host, directory, budget)
                observed = check_ancestors(context)
                ops.write_record(directory / 'preservation-before-export.json', {
                    'kind': KIND + '-preservation', 'stage': 'before-export', 'case_id': cid,
                    'operation_sha256': canonical(context['op']), 'observed_files_sha256': observed,
                    'utc': datetime.now(timezone.utc).isoformat(), 'research_admitted': False})
                control.update(complete=True, active_role=None, launcher_elapsed_seconds=elapsed(launcher_started))
                control['prior_plus_new_elapsed_seconds'] = CARRIED + control['launcher_elapsed_seconds']
                ops.write_record(directory / 'case-envelope.json', control, replace=True)
                body = {'kind': KIND + '-outcome', 'operation_sha256': canonical(context['op']),
                    'case_id': cid, 'host': host, 'complete': True, 'execution_sha256': receipt['execution_sha256'],
                    'numeric_receipt_sha256': canonical(receipt), 'numeric_receipt_file_sha256': digest(directory / 'receipt.json'),
                    'numeric_archive_sha256': inner['sha256'], 'launcher_elapsed_seconds': control['launcher_elapsed_seconds'],
                    'prior_plus_new_elapsed_seconds': control['prior_plus_new_elapsed_seconds'], 'resource_grant': recovery_grant(),
                    'research_admitted': False}
                ops.write_record(root / 'outcomes' / (cid + '.json'), body)
                result = export_case(context, host, directory)
                observed = check_ancestors(context)
                ops.write_record(directory / 'preservation-after-export.json', {
                    'kind': KIND + '-preservation', 'stage': 'after-export', 'case_id': cid,
                    'operation_sha256': canonical(context['op']), 'observed_files_sha256': observed,
                    'utc': datetime.now(timezone.utc).isoformat(), 'research_admitted': False})
                elapsed(launcher_started)
                print(host, cid, 'complete; independent export review required. No next case starts.', flush=True)
                return result
            except BaseException as error:
                control.update(complete=False, active_role=None, launcher_elapsed_seconds=time.time() - launcher_started,
                    error=type(error).__name__ + ': ' + str(error))
                ops.write_record(directory / 'case-envelope.json', control, replace=True)
                registry.fail(host, control['error'], control['launcher_elapsed_seconds'])
                raise


def review(context, archive, sidecar, archive_pin, output):
    """Rebuild acceptance from saved evidence only; never start a model."""
    from models.matched_panel_io import review_export, verify_archive
    from models.pilot_bundle import extract_verified
    output = Path(output)
    verify_archive(archive, sidecar, archive_pin)
    extract_verified(archive, sidecar, output)
    op, prior, ops = context['op'], context['prior'], context['ops']
    if json.loads((output / 'CONTINUATION.json').read_text()) != op:
        raise ValueError('Captured operation changed.')
    for name, sha in op['sources_sha256'].items():
        if digest(prior.bound(output, 'source/' + name)) != sha:
            raise ValueError('Captured continuation source changed.')
    for name, sha in context['manifest']['source_sha256'].items():
        if digest(prior.bound(output, 'reference-source/' + name)) != sha:
            raise ValueError('Captured original reference source changed.')
    if (digest(output / 'prior-receipt.json') != PRIOR_RECEIPT
            or json.loads((output / 'prior-OPERATION.json').read_text()) != context['sync_op']
            or json.loads((output / 'prior-INPUT.json').read_text()) != context['sync_input']
            or json.loads((output / 'INPUT.json').read_text()) != context['data']
            or canonical(json.loads((output / 'reference-implementation.json').read_text())) != MP):
        raise ValueError('Captured actual failed activation boundary changed.')
    for name in METADATA:
        if digest(output / name) != digest(context['packet'] / name):
            raise ValueError('Captured proposal, authorization or incident anchor changed.')
    preservation = ops.read_record(output / 'preservation-before-export.json')
    if (preservation['kind'] != KIND + '-preservation' or preservation['stage'] != 'before-export'
            or preservation['operation_sha256'] != canonical(op)
            or preservation['observed_files_sha256'] != ancestor_pins(context)
            or preservation['research_admitted'] is not False):
        raise ValueError('Captured ancestor preservation observation changed.')
    envelope = ops.read_record(output / 'case-envelope.json')
    host = envelope['host']
    if host not in TARGETS:
        raise ValueError('Unknown export host.')
    cid = TARGETS[host]
    q = request(op, host)
    expected_claim = claim_body(op, host)
    if (ops.read_record(output / 'case-request.json') != q or ops.read_record(output / 'case-claim.json') != expected_claim
            or envelope['operation_sha256'] != canonical(op) or envelope['case_id'] != cid
            or envelope['complete'] is not True or envelope['active_role'] is not None
            or envelope.get('resource_grant') != recovery_grant()
            or envelope.get('prior_plus_new_elapsed_seconds') != CARRIED + envelope['launcher_elapsed_seconds']
            or envelope['case_seconds'] != LIMIT or envelope['research_admitted'] is not False
            or type(envelope['launcher_elapsed_seconds']) not in (int, float)
            or not 0 < envelope['launcher_elapsed_seconds'] <= LIMIT):
        raise ValueError('Incomplete, mixed or over-limit attended case.')
    expected_records = activation_records(op, context['data'], prior)
    for source, target in [('genesis.json', 'ledger-genesis.json'), ('imports.json', 'ledger-imports.json'), ('resolution.json', 'resolution.json'), ('recovery-grant.json', 'recovery-grant.json')]:
        if (output / target).read_bytes() != expected_records[source]:
            raise ValueError('Captured continuation import or stop resolution changed.')
    execution = execution_identity(context, host)
    for stage in HARDWARE_STAGES:
        event = ops.read_record(output / 'hardware' / ('hardware-' + stage + '.json'))
        if (event['kind'] != KIND + '-GPU-check' or event['host'] != host or event['stage'] != stage
                or event['passed'] is not True or event['GPU_identity'] != ops.HOSTS[host]
                or event['strict_guard_source_unchanged'] is not True or event['research_admitted'] is not False):
            raise ValueError('Missing or rejected strict GPU boundary check.')
    outcome = ops.read_record(output / 'case-outcome.json')
    inner = output / 'numeric/numeric-evidence.tar.gz'
    inner_manifest = json.loads(Path(str(inner) + '.manifest.json').read_text())
    receipt = review_export(inner, inner_manifest, outcome['numeric_archive_sha256'], output / 'numeric-reviewed',
                            plan=context['numeric'], execution=execution)
    if (outcome['kind'] != KIND + '-outcome' or outcome['operation_sha256'] != canonical(op)
            or outcome['case_id'] != cid or outcome['host'] != host or outcome['complete'] is not True
            or outcome['execution_sha256'] != canonical(execution) or outcome['numeric_receipt_sha256'] != canonical(receipt)
            or outcome['numeric_receipt_file_sha256'] != digest(output / 'numeric-reviewed/receipt.json')
            or outcome.get('resource_grant') != recovery_grant()
            or outcome.get('prior_plus_new_elapsed_seconds') != envelope['prior_plus_new_elapsed_seconds']
            or outcome.get('launcher_elapsed_seconds') != envelope['launcher_elapsed_seconds']
            or outcome['research_admitted'] is not False):
        raise ValueError('Saved continuation outcome differs from retained numerical evidence.')
    expected_files = set(outer_files(dict(context, transfer=Path('/unused'), root=Path('/unused')), host, Path('/unused')))
    expected_files |= {'numeric/numeric-evidence.tar.gz', 'numeric/numeric-evidence.tar.gz.manifest.json'}
    if set(sidecar['files']) != expected_files:
        raise ValueError('Unexpected continuation export inventory.')
    return {'kind': KIND + '-independent-review', 'operation_sha256': canonical(op), 'case_id': cid, 'host': host,
        'archive_sha256': archive_pin, 'outcome_file_sha256': digest(output / 'case-outcome.json'),
        'numeric_receipt_sha256': canonical(receipt), 'execution_sha256': canonical(execution),
        'reviewed_export': True, 'research_admitted': False, 'automatic_queue_authorized': False}


