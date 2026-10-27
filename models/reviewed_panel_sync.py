"""Append two independently verified case reviews; no numerical dispatch or queue."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile

KIND = 'matched-panel-reviewed-case-sync-v1'
BASE = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PREFIX = 'reviewed-pair-sync-20261027'
CORE_PREFIX = 'reviewed-continuation-20261026'
CORE_ROOT = 'reviewed-execution-20261026'
CORE_PIN = '0714e035422d4e21ba41c37734997138aa3249abb0966cf8b8324b6c96ac7300'
CORE_SOURCE = '51f78c2b4ff4904f4492e4ff055d58eb7f219a4a3b1f440dd502c43725c71b96'
ACTIVATION_RECEIPT = '32acded6836409b294caf3d57f562e1b1d7055c27574d4ad99d9ba612cce0dce'
CASES = {'linux10609': 'clip-05-bank-1', 'linux10603': 'clip-06-bank-0'}
MARKER_PINS = {'clip-05-bank-1': 'bda8ce30a7e6d44801ead63cb0f95402a4f9632a59cb0480ef1f4785e66fa351',
               'clip-06-bank-0': '6b5ef5fef2fd26797831928b35dfe59b040d1e4826da81e02a947c73fad87517'}
ORDER = [f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0, 1)]
TARGETS = {'reviews/' + cid + '.json' for cid in CASES.values()}
SOURCES = ('models/reviewed_panel_sync.py', 'scripts/sync_reviewed_panel.py',
           'scripts/prepare_reviewed_panel_sync.py', 'tests/test_reviewed_panel_sync.py',
           'tests/fixtures/reviewed_panel_sync_metadata.json', 'support/reviewed_panel_continuation.py')


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
    fixed = {'kind': KIND, 'host': 'linux10609', 'core_operation_sha256': CORE_PIN,
        'core_module_sha256': CORE_SOURCE, 'actual_activation_receipt_file_sha256': ACTIVATION_RECEIPT,
        'coordinator_namespace': CORE_ROOT, 'case_ids': list(CASES.values()),
        'accepted_after_sync': ORDER[:13], 'preserved_file_count': 275, 'max_seconds': 300,
        'kill_grace_seconds': 15, 'metadata_only': True, 'models_started': False,
        'test_suite_started': False, 'new_GPU_dispatch_authorized': False,
        'historical_STOP_records_unchanged': True, 'automatic_queue': False, 'research_admitted': False}
    if canonical(op) != pin or any(k not in op or canonical(op[k]) != canonical(v) for k, v in fixed.items()):
        raise ValueError('Use the pinned two-review CPU-only synchronization.')
    if set(op['sources_sha256']) != set(SOURCES) or op['sources_sha256']['support/reviewed_panel_continuation.py'] != CORE_SOURCE:
        raise ValueError('Incomplete or changed synchronization source inventory.')


def marker_payload(row):
    value = json.loads(row['text'])
    if (set(value) != {'payload', 'sha256'} or canonical(value['payload']) != value['sha256']
            or hashlib.sha256(row['text'].encode()).hexdigest() != row['sha256']):
        raise ValueError('Independent marker bytes or checksum changed.')
    return value['payload']


def validate_input(data):
    if (data['kind'] != KIND + '-input' or data['core_operation_sha256'] != CORE_PIN
            or data['actual_activation_receipt_file_sha256'] != ACTIVATION_RECEIPT
            or set(data['reviews']) != set(CASES.values()) or len(data['preserved_files_sha256']) != 275
            or data['models_started'] is not False or data['research_admitted'] is not False):
        raise ValueError('Wrong reviewed evidence or preservation scope.')
    for host, cid in CASES.items():
        row = data['reviews'][cid]
        marker = marker_payload(row)
        if (row['sha256'] != MARKER_PINS[cid] or marker['kind'] != 'matched-panel-reviewed-continuation-v1-independent-review'
                or marker['operation_sha256'] != CORE_PIN or marker['case_id'] != cid or marker['host'] != host
                or marker['reviewed_export'] is not True or marker['research_admitted'] is not False
                or marker['automatic_queue_authorized'] is not False):
            raise ValueError('Marker is not the original independently verified completed case.')
        case_root = CORE_ROOT + '/cases/' + cid + '/attempt-01/'
        pins = data['preserved_files_sha256']
        if pins.get(case_root + 'continuation-evidence.tar.gz') != marker['archive_sha256']:
            raise ValueError('Original reviewed archive is unpinned.')
        if pins.get(CORE_ROOT + '/outcomes/' + cid + '.json') != marker['outcome_file_sha256']:
            raise ValueError('Original reviewed outcome is unpinned.')
        if any(n not in pins for n in [case_root + 'receipt.json', case_root + 'session.json',
                case_root + 'preservation-after-export.json', case_root + 'case-envelope.json',
                CORE_ROOT + '/claims/' + cid + '/claim.json', CORE_ROOT + '/exports/' + cid + '.json']):
            raise ValueError('Missing claim, completion, elapsed-time or post-export evidence.')
    return data


def targets(data):
    validate_input(data)
    return {'reviews/' + cid + '.json': row['text'].encode() for cid, row in data['reviews'].items()}


def load(packet, pin, home):
    packet = Path(packet).resolve()
    transfer = (Path(home) / BASE).resolve()
    if packet != (transfer / PREFIX / 'unpacked' / PREFIX).resolve():
        raise ValueError('Use the fixed source-only synchronization packet.')
    op = json.loads((packet / 'SYNC.json').read_text())
    validate_operation(op, pin)
    for name, sha in op['sources_sha256'].items():
        if digest(packet / name) != sha:
            raise ValueError('Synchronization source changed: ' + name)
    if digest(packet / 'INPUT.json') != op['inputs_file_sha256']:
        raise ValueError('Externally pinned synchronization input changed.')
    data = validate_input(json.loads((packet / 'INPUT.json').read_text()))
    if digest(packet / 'activation-receipt.json') != ACTIVATION_RECEIPT:
        raise ValueError('Actual independent activation boundary changed.')
    core = load_module('completed_reviewed_continuation', packet / 'support/reviewed_panel_continuation.py')
    core_packet = transfer / CORE_PREFIX / 'unpacked' / CORE_PREFIX
    context = core.load(core_packet, CORE_PIN, home)
    for name, sha in core.ancestor_pins(context).items():
        if data['preserved_files_sha256'].get(name) != sha:
            raise ValueError('Original 148-file ancestor boundary changed.')
    original_records = core.activation_records(context['op'], context['data'], context['prior'])
    actual_receipt = json.loads((packet / 'activation-receipt.json').read_text())
    if actual_receipt['record_texts'] != {n: b.decode() for n, b in original_records.items()}:
        raise ValueError('Actual activated five-record boundary differs.')
    for name, content in original_records.items():
        if data['preserved_files_sha256'].get(CORE_ROOT + '/' + name) != hashlib.sha256(content).hexdigest():
            raise ValueError('Activated request/import identity is not preserved.')
    return {'packet': packet, 'op': op, 'data': data, 'core': core, 'context': context,
            'ops': context['ops'], 'prior': context['prior'], 'root': context['root'], 'transfer': transfer}


def check_preserved(c):
    return c['prior'].check_preserved(c['transfer'], c['data']['preserved_files_sha256'])


def check_targets(root, values, prior):
    if set(values) != TARGETS:
        raise ValueError('Only the two original independent markers may be published.')
    existing = []
    for name, content in values.items():
        path = prior.bound(root, name)
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError('Conflicting existing independent review.')
            existing.append(name)
    return existing


def check_state(status, reviews):
    accepted = ORDER[:11] + [cid for cid in ORDER[11:13] if 'reviews/' + cid + '.json' in reviews]
    expected = {'kind': 'matched-panel-reviewed-continuation-v1-progress', 'accepted_case_ids': accepted,
        'claimed_case_ids': list(CASES.values()), 'computed_case_ids': list(CASES.values()),
        'failed_case_ids': [], 'stop_dispatch': False, 'complete': False,
        'scientific_aggregate': None, 'research_admitted': False}
    if status != expected:
        raise ValueError('Coordinator differs from the two reviewed completed cases; preserve it for inspection.')


def verify_saved(c, temporary):
    """Rebuild both original markers and live receipts from existing saved tensors on CPU."""
    from models.matched_panel_case import verify_case
    core, context, data = c['core'], c['context'], c['data']
    for host, cid in CASES.items():
        directory = c['prior'].bound(c['root'], 'cases/' + cid + '/attempt-01')
        archive = directory / 'continuation-evidence.tar.gz'
        sidecar = json.loads(Path(str(archive) + '.manifest.json').read_text())
        marker = marker_payload(data['reviews'][cid])
        rebuilt = core.review(context, archive, sidecar, marker['archive_sha256'], Path(temporary) / cid)
        if rebuilt != marker:
            raise ValueError('Saved outer evidence does not rebuild its original independent marker.')
        receipt = verify_case(directory, plan=context['numeric'], case_id=cid,
                              execution=core.execution_identity(context, host))
        if canonical(receipt) != marker['numeric_receipt_sha256']:
            raise ValueError('Live numerical evidence differs from the reviewed archive.')
    return 2


def publish(root, values, prior, verify_old):
    verify_old()
    existing = check_targets(root, values, prior)
    created = []
    for name in sorted(values):
        path = prior.bound(root, name)
        if name not in existing:
            prior.exclusive_bytes(path, values[name])
            created.append(name)
        if path.read_bytes() != values[name]:
            raise ValueError('Independent review readback mismatch.')
    verify_old()
    return created, existing


def apply(packet, pin, home, host):
    if host != 'linux10609':
        raise ValueError('Use the declared metadata-only host.')
    c = load(packet, pin, home)
    import torch
    torch.set_num_threads(4)
    values = targets(c['data'])
    with ExitStack() as locks:
        for assigned in CASES:
            locks.enter_context(c['ops'].file_lock(c['context']['legacy'] / 'leases' / assigned, wait_seconds=0))
        registry = c['core'].Registry(c['context'])
        with c['ops'].file_lock(c['root'] / '.registry.lock'):
            existing_before = check_targets(c['root'], values, c['prior'])
            before = registry.status()
            check_state(before, existing_before)
            check_preserved(c)
            with tempfile.TemporaryDirectory(prefix='reviewed-pair-sync-CPU-') as temporary:
                checked = verify_saved(c, temporary)
            if checked != 2 or registry.status() != before:
                raise ValueError('Incomplete saved-case check or concurrent state change.')
            created, existing = publish(c['root'], values, c['prior'], lambda: check_preserved(c))
            after = registry.status()
            check_state(after, TARGETS)
            observed = check_preserved(c)
            readback = {name: c['prior'].bound(c['root'], name).read_text() for name in values}
    receipt = {'kind': KIND + '-receipt', 'operation_sha256': pin,
        'inputs_file_sha256': c['op']['inputs_file_sha256'], 'created': created, 'preexisting_identical': existing,
        'records_readback': readback, 'fresh_preserved_files_sha256': observed,
        'status_before': before, 'status_after': after, 'saved_cases_reverified': checked,
        'historical_STOP_records_unchanged': True, 'models_started': False,
        'test_suite_started': False, 'new_GPU_dispatch_authorized': False, 'research_admitted': False}
    verify_receipt(receipt, c['op'], c['data'])
    return receipt


def verify_receipt(receipt, op, data):
    validate_operation(op, canonical(op))
    values = targets(data)
    created, existing = receipt['created'], receipt['preexisting_identical']
    if (receipt['kind'] != KIND + '-receipt' or receipt['operation_sha256'] != canonical(op)
            or receipt['inputs_file_sha256'] != op['inputs_file_sha256'] or set(created) & set(existing)
            or set(created) | set(existing) != TARGETS or len(created) + len(existing) != 2
            or receipt['records_readback'] != {n: b.decode() for n, b in values.items()}
            or receipt['fresh_preserved_files_sha256'] != data['preserved_files_sha256']
            or receipt['saved_cases_reverified'] != 2 or receipt['historical_STOP_records_unchanged'] is not True
            or any(receipt[k] is not False for k in ('models_started', 'test_suite_started', 'new_GPU_dispatch_authorized', 'research_admitted'))):
        raise ValueError('Receipt differs from the two markers and 275 preserved files.')
    check_state(receipt['status_before'], existing)
    check_state(receipt['status_after'], TARGETS)
    return {'verified': True, 'accepted_cases': 13, 'complete_clip_pairs': 6, 'remaining_cases': 7,
            'preserved_files': 275, 'saved_cases_reverified': 2, 'models_started': False,
            'new_GPU_dispatch_authorized': False, 'historical_STOP_records_unchanged': True}
