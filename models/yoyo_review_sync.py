"""Register the one original YoYo review and preserve the unused Archery case; no numerical dispatch or queue."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING

KIND = 'matched-panel-yoyo-single-review-sync-v1'
BASE = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PREFIX = 'yoyo-review-sync-20261027'
CORE_PREFIX = 'archery-yoyo-pair-20261027'
CORE_ROOT = 'archery-yoyo-execution-20261027'
CORE_PIN = '4e5993581a76b1c62d809c9d038e8428694a41e8d78304a7fe0247583f347d70'
CORE_SOURCE = '7989283aca4be98e53929d0761110d0896e8d8a3324e1751fa4141641e067651'
ACTIVATION_RECEIPT = '7d5c519983419ddb455e509f94d534743c82478957791e4e386680f48ae8ca6d'
CASES = {'linux10609': 'clip-07-bank-0'}
MARKER_PINS = {'clip-07-bank-0': '35281514c0c2ce991b0d6ce1390dd1ba1853886af6529e1e24aab868f2452f99'}
ARCHERY = 'clip-06-bank-1'
ARCHERY_HOST = 'linux10603'
ACCOUNTING_PIN = 'ab600f0150874c584cb3f9560a7a349cdea7243dd7a2e5be1253b61e53f47224'
PRESERVED = 382
ORDER = [f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0, 1)]
TARGETS = {'reviews/' + cid + '.json' for cid in CASES.values()}
SOURCES = ('models/yoyo_review_sync.py', 'scripts/sync_yoyo_review.py',
           'scripts/prepare_yoyo_review_sync.py', 'tests/test_yoyo_review_sync.py',
           'tests/fixtures/yoyo_review_sync_metadata.json', 'support/reviewed_panel_batch.py')


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
        'accepted_after_sync': ORDER[:13] + ['clip-07-bank-0'], 'preserved_file_count': PRESERVED, 'max_seconds': 300,
        'kill_grace_seconds': 15, 'metadata_only': True, 'models_started': False,
        'test_suite_started': False, 'new_GPU_dispatch_authorized': False,
        'historical_STOP_records_unchanged': True, 'automatic_queue': False, 'research_admitted': False,
        'archery_unclaimed': True, 'archery_startup_charge_seconds': 43,
        'archery_remaining_original_allowance_seconds': 4757, 'archery_accounting_file_sha256': ACCOUNTING_PIN}
    if canonical(op) != pin or any(k not in op or canonical(op[k]) != canonical(v) for k, v in fixed.items()):
        raise ValueError('Use the pinned single-review CPU-only synchronization.')
    if set(op['sources_sha256']) != set(SOURCES) or op['sources_sha256']['support/reviewed_panel_batch.py'] != CORE_SOURCE:
        raise ValueError('Incomplete or changed synchronization source inventory.')


def marker_payload(row):
    value = json.loads(row['text'])
    if (set(value) != {'payload', 'sha256'} or canonical(value['payload']) != value['sha256']
            or hashlib.sha256(row['text'].encode()).hexdigest() != row['sha256']):
        raise ValueError('Independent marker bytes or checksum changed.')
    return value['payload']


def accounting_payload(row):
    if hashlib.sha256(row['text'].encode()).hexdigest() != row['sha256'] or row['sha256'] != ACCOUNTING_PIN:
        raise ValueError('Use the original independently verified startup accounting bytes.')
    a = json.loads(row['text'])
    fixed = {'kind': 'archery-unclaimed-startup-accounting-v1', 'case_id': ARCHERY, 'host': ARCHERY_HOST,
        'operation_sha256': CORE_PIN, 'conservative_startup_charge_seconds': 43,
        'remaining_original_allowance_seconds': 4757, 'original_case_allowance_seconds': 4800,
        'additional_allowance_seconds': 0, 'numeric_attempts_started': 0,
        'claim_and_case_absent_at_snapshot': True, 'new_GPU_execution_authorized': False, 'research_admitted': False}
    if any(k not in a or canonical(a[k]) != canonical(v) for k,v in fixed.items()):
        raise ValueError('Changed unused-case accounting or new execution authority.')
    return a


def archery_readback(account):
    return {'case_id': ARCHERY, 'host': ARCHERY_HOST, 'accounting_file_sha256': ACCOUNTING_PIN,
        'claim_case_attempt_outcome_export_review_failure_STOP_absent': True,
        'started_epoch_decimal': account['started_epoch_decimal'], 'started_utc_floor': account['started_utc_floor'],
        'ended_utc_floor': account['ended_utc_floor'], 'conservative_startup_charge_seconds': 43,
        'remaining_original_allowance_seconds': 4757, 'additional_allowance_seconds': 0,
        'new_GPU_execution_authorized': False}


def check_archery(c):
    account = accounting_payload(c['data']['archery_accounting'])
    paths = ['claims/' + ARCHERY, 'cases/' + ARCHERY] + [n + '/' + ARCHERY + '.json'
        for n in ('outcomes', 'exports', 'reviews', 'failures')] + ['STOP.json']
    for n in paths:
        p = c['prior'].bound(c['root'], n)
        if p.exists() or p.is_symlink():
            raise ValueError('Archery must remain unclaimed and no current STOP may be cleared.')
    p = c['context']['packet'] / 'launchers' / (ARCHERY_HOST + '-' + ARCHERY)
    start = Decimal((p / 'started-epoch.txt').read_text().strip())
    start_floor = datetime.strptime((p / 'started-utc.txt').read_text().strip(), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    end_floor = datetime.strptime((p / 'ended-utc.txt').read_text().strip(), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    if not start.is_finite() or int(start) != int(start_floor.timestamp()):
        raise ValueError('Unknown or mixed original launcher clock.')
    upper = Decimal(int(end_floor.timestamp())) + 1 - start
    if not Decimal(0) < upper < Decimal(4800):
        raise ValueError('Invalid or exhausted stopped startup interval.')
    charged = int(upper.to_integral_value(rounding=ROUND_CEILING))
    if (str(start) != account['started_epoch_decimal'] or start_floor.isoformat() != account['started_utc_floor']
            or end_floor.isoformat() != account['ended_utc_floor'] or charged != 43 or 4800 - charged != 4757
            or (p / 'exit-status.txt').read_text().strip() != '1'
            or 'ValueError: Another CUDA compute process is active on this host.' not in (p / 'case.log').read_text()):
        raise ValueError('Original guard refusal or conservative startup charge changed.')
    return archery_readback(account)


def validate_input(data):
    if (data['kind'] != KIND + '-input' or data['core_operation_sha256'] != CORE_PIN
            or data['actual_activation_receipt_file_sha256'] != ACTIVATION_RECEIPT
            or set(data['reviews']) != set(CASES.values()) or len(data['preserved_files_sha256']) != PRESERVED
            or data['models_started'] is not False or data['research_admitted'] is not False):
        raise ValueError('Wrong reviewed evidence or preservation scope.')
    for host, cid in CASES.items():
        row = data['reviews'][cid]
        marker = marker_payload(row)
        if (row['sha256'] != MARKER_PINS[cid] or marker['kind'] != 'matched-panel-reviewed-batch-v1-independent-review'
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
    accounting_payload(data['archery_accounting'])
    launcher = CORE_PREFIX + '/unpacked/' + CORE_PREFIX + '/launchers/' + ARCHERY_HOST + '-' + ARCHERY + '/'
    if any(launcher + n not in data['preserved_files_sha256'] for n in
        ('started-epoch.txt', 'started-utc.txt', 'ended-utc.txt', 'exit-status.txt', 'case.log', 'cpu-results.xml', 'cpu-summary.log')):
        raise ValueError('Original refused startup and elapsed-time files must remain pinned.')
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
    core = load_module('completed_reviewed_continuation', packet / 'support/reviewed_panel_batch.py')
    core_packet = transfer / CORE_PREFIX / 'unpacked' / CORE_PREFIX
    context = core.load(core_packet, CORE_PIN, home)
    for name, sha in core.ancestor_pins(context).items():
        if data['preserved_files_sha256'].get(name) != sha:
            raise ValueError('Original 286-file ancestor boundary changed.')
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
        raise ValueError('Only the one original independent marker may be published.')
    existing = []
    for name, content in values.items():
        path = prior.bound(root, name)
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError('Conflicting existing independent review.')
            existing.append(name)
    return existing


def check_state(status, reviews):
    accepted = ORDER[:13] + [cid for cid in CASES.values() if 'reviews/' + cid + '.json' in reviews]
    expected = {'kind': 'matched-panel-reviewed-batch-v1-progress', 'accepted_case_ids': accepted,
        'claimed_case_ids': list(CASES.values()), 'computed_case_ids': list(CASES.values()),
        'failed_case_ids': [], 'stop_dispatch': False, 'complete': False,
        'scientific_aggregate': None, 'research_admitted': False}
    if status != expected:
        raise ValueError('Coordinator differs from the one completed YoYo case and unused Archery request; preserve it for inspection.')


def verify_saved(c, temporary):
    """Rebuild the original YoYo marker and live receipt from existing saved tensors on CPU."""
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
    return 1


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
        for assigned in c['core'].TARGETS:
            locks.enter_context(c['ops'].file_lock(c['context']['legacy'] / 'leases' / assigned, wait_seconds=0))
        registry = c['core'].Registry(c['context'])
        with c['ops'].file_lock(c['root'] / '.registry.lock'):
            existing_before = check_targets(c['root'], values, c['prior'])
            before = registry.status()
            check_state(before, existing_before)
            check_preserved(c)
            archery_before = check_archery(c)
            with tempfile.TemporaryDirectory(prefix='yoyo-single-review-sync-CPU-') as temporary:
                checked = verify_saved(c, temporary)
            if checked != 1 or registry.status() != before:
                raise ValueError('Incomplete saved-case check or concurrent state change.')
            created, existing = publish(c['root'], values, c['prior'], lambda: check_preserved(c))
            after = registry.status()
            check_state(after, TARGETS)
            observed = check_preserved(c)
            archery_after = check_archery(c)
            if archery_after != archery_before:
                raise ValueError('Unused Archery case or elapsed-time accounting changed during sync.')
            readback = {name: c['prior'].bound(c['root'], name).read_text() for name in values}
    receipt = {'kind': KIND + '-receipt', 'operation_sha256': pin,
        'inputs_file_sha256': c['op']['inputs_file_sha256'], 'created': created, 'preexisting_identical': existing,
        'records_readback': readback, 'fresh_preserved_files_sha256': observed,
        'status_before': before, 'status_after': after, 'saved_cases_reverified': checked,
        'historical_STOP_records_unchanged': True, 'models_started': False,
        'test_suite_started': False, 'new_GPU_dispatch_authorized': False, 'research_admitted': False,
        'archery_before': archery_before, 'archery_after': archery_after}
    verify_receipt(receipt, c['op'], c['data'])
    return receipt


def verify_receipt(receipt, op, data):
    validate_operation(op, canonical(op))
    values = targets(data)
    created, existing = receipt['created'], receipt['preexisting_identical']
    if (receipt['kind'] != KIND + '-receipt' or receipt['operation_sha256'] != canonical(op)
            or receipt['inputs_file_sha256'] != op['inputs_file_sha256'] or set(created) & set(existing)
            or set(created) | set(existing) != TARGETS or len(created) + len(existing) != 1
            or receipt['records_readback'] != {n: b.decode() for n, b in values.items()}
            or receipt['fresh_preserved_files_sha256'] != data['preserved_files_sha256']
            or receipt['saved_cases_reverified'] != 1 or receipt['historical_STOP_records_unchanged'] is not True
            or any(receipt[k] is not False for k in ('models_started', 'test_suite_started', 'new_GPU_dispatch_authorized', 'research_admitted'))):
        raise ValueError('Receipt differs from the original YoYo marker and 382 preserved files.')
    expected_archery = archery_readback(accounting_payload(data['archery_accounting']))
    if receipt['archery_before'] != expected_archery or receipt['archery_after'] != expected_archery:
        raise ValueError('Receipt changes Archery claim absence, startup charge or remaining original allowance.')
    check_state(receipt['status_before'], existing)
    check_state(receipt['status_after'], TARGETS)
    return {'verified': True, 'accepted_cases': 14, 'complete_clip_pairs': 6, 'remaining_cases': 6,
            'preserved_files': PRESERVED, 'saved_cases_reverified': 1, 'models_started': False,
            'new_GPU_dispatch_authorized': False, 'historical_STOP_records_unchanged': True,
            'archery_unclaimed': True, 'archery_startup_charge_seconds': 43, 'archery_remaining_original_allowance_seconds': 4757}
