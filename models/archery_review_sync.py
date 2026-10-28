"""Append the original approved recovery review; preserve failures and consumed time."""
from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

KIND = 'matched-panel-archery-recovery-review-sync-v1'
BASE = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PREFIX = 'archery-review-sync-20261027'
CORE_PREFIX = 'archery-recovery-20261027'
CORE_ROOT = 'archery-recovery-execution-20261027'
CORE_PIN = '1757892c83921e0cee8391b2468989680bcb44c57b6fa686f3528de9142eb4b5'
CORE_SOURCE = '8e29c1931c1ecb9734fa31a969be0524efe2545f08df15127bc7eb73f7db4215'
ACTIVATION_RECEIPT = '5417f0b5db0cff2306a646a8c40e4d3e8642134abca315ec8aba7da856ec1a52'
MARKER = 'ba9adb7e31588dd85a697436c637a3c602997fb4f6c701957ee775af0c9f401a'
ACCOUNTING = '31f1ee6803722e70121cc6d7bd020c11b6c6a7546a207b76e308715b68bb271a'
EXECUTION = '87bd519d87e4aae442d4837dac6f165d9459f4fcb2903328b928488c69db4760'
RECEIPT = 'a923a1838d11215c2a33f59de59e155806fdf8e3eda2698d94db5a63e6d3dfc1'
GRANT = 'fd4b4cb8d42e442f8d9ee8ac7130945e'
AUTH = '41141726758e31b4babf76feadfe48618b4399a5147ace7dcea614ea14feabc3'
CASE, HOST, ATTEMPT = 'clip-06-bank-1', 'linux10603', 'recovery-01'
PRESERVED = 578
ORDER = [f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0, 1)]
BEFORE = ORDER[:13] + ['clip-07-bank-0']
TARGET = 'reviews/' + CASE + '.json'
SOURCES = ('models/archery_review_sync.py', 'scripts/sync_archery_review.py',
           'scripts/prepare_archery_review_sync.py', 'tests/test_archery_review_sync.py',
           'tests/fixtures/archery_review_sync_metadata.json', 'support/archery_recovery.py')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def policy():
    return {'kind': KIND, 'host': 'linux10609', 'core_operation_sha256': CORE_PIN,
        'core_module_sha256': CORE_SOURCE, 'actual_activation_receipt_file_sha256': ACTIVATION_RECEIPT,
        'coordinator_namespace': CORE_ROOT, 'case_ids': [CASE], 'attempt': ATTEMPT,
        'accepted_after_sync': ORDER[:15], 'original_review_marker_file_sha256': MARKER,
        'preserved_file_count': PRESERVED, 'max_seconds': 300, 'kill_grace_seconds': 15,
        'accounting_file_sha256': ACCOUNTING, 'consumed_grant_id': GRANT,
        'previous_charged_seconds': 2348, 'recovery_charged_seconds': 2599,
        'total_charged_seconds': 4947, 'old_remainder_added': False,
        'metadata_only': True, 'models_started': False, 'test_suite_started': False,
        'new_GPU_dispatch_authorized': False, 'additional_attempt_authorized': False,
        'historical_STOP_records_unchanged': True, 'automatic_queue': False, 'research_admitted': False}


def validate_operation(op, pin):
    if canonical(op) != pin or any(k not in op or canonical(op[k]) != canonical(v) for k, v in policy().items()):
        raise ValueError('Use the pinned metadata-only registration of this original recovery review.')
    if set(op['sources_sha256']) != set(SOURCES) or op['sources_sha256']['support/archery_recovery.py'] != CORE_SOURCE:
        raise ValueError('Incomplete or changed registration source inventory.')


def marker(data):
    row = data['review']; value = json.loads(row['text'])
    if (set(value) != {'payload', 'sha256'} or canonical(value['payload']) != value['sha256']
            or hashlib.sha256(row['text'].encode()).hexdigest() != row['sha256'] or row['sha256'] != MARKER):
        raise ValueError('Use the exact original independent review bytes.')
    m = value['payload']
    fixed = {'kind': 'matched-panel-archery-recovery-v1-independent-review', 'operation_sha256': CORE_PIN,
        'case_id': CASE, 'host': HOST, 'execution_sha256': EXECUTION, 'numeric_receipt_sha256': RECEIPT,
        'reviewed_export': True, 'research_admitted': False, 'automatic_queue_authorized': False}
    if any(k not in m or canonical(m[k]) != canonical(v) for k, v in fixed.items()):
        raise ValueError('Mixed recovery execution or unsupported marker authority.')
    return m


def accounting(data):
    row = data['accounting']
    if row['sha256'] != ACCOUNTING or hashlib.sha256(row['text'].encode()).hexdigest() != ACCOUNTING:
        raise ValueError('Original independent attempt accounting changed.')
    a = json.loads(row['text'])
    fixed = {'prior_charged_seconds': 2348, 'new_launcher_conservative_charged_seconds': 2599,
        'total_conservative_charged_seconds_including_prior': 4947, 'old_remaining_seconds': 2452,
        'old_remainder_added': False, 'fresh_attempt_limit_seconds': 4800, 'kill_grace_seconds': 15,
        'new_window_unspent_seconds_not_a_retry_grant': 2201}
    if any(k not in a or canonical(a[k]) != canonical(v) for k, v in fixed.items()):
        raise ValueError('Do not reset elapsed time or grant another attempt.')
    return a


def validate_input(data):
    fixed = {'kind': KIND + '-input', 'core_operation_sha256': CORE_PIN,
        'actual_activation_receipt_file_sha256': ACTIVATION_RECEIPT, 'models_started': False, 'research_admitted': False}
    if any(k not in data or canonical(data[k]) != canonical(v) for k, v in fixed.items()):
        raise ValueError('Wrong registration input scope.')
    m = marker(data); accounting(data); pins = data['preserved_files_sha256']
    if len(pins) != PRESERVED or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in pins.values()):
        raise ValueError('Preserve all 578 existing source, failure, success and clock hashes.')
    case = CORE_ROOT + '/cases/' + CASE + '/' + ATTEMPT + '/'
    required = [case + n for n in ('case-envelope.json', 'receipt.json', 'session.json',
        'preservation-before-export.json', 'preservation-after-export.json', 'numeric-evidence.tar.gz',
        'numeric-evidence.tar.gz.manifest.json', 'continuation-evidence.tar.gz.manifest.json')]
    required += [CORE_ROOT + '/' + n for n in ('recovery-grant.json', 'claims/' + CASE + '/claim.json', 'exports/' + CASE + '.json')]
    launcher = CORE_PREFIX + '/unpacked/' + CORE_PREFIX + '/launchers/' + HOST + '-' + CASE + '/'
    required += [launcher + n for n in ('started-epoch.txt', 'started-utc.txt', 'ended-utc.txt', 'exit-status.txt',
        'cpu-results.xml', 'cpu-summary.log', 'case.log', 'source-launch.sh', 'source-pipeline.sh')]
    if (any(n not in pins for n in required) or CORE_ROOT + '/' + TARGET in pins
            or pins.get(case + 'continuation-evidence.tar.gz') != m['archive_sha256']
            or pins.get(CORE_ROOT + '/outcomes/' + CASE + '.json') != m['outcome_file_sha256']):
        raise ValueError('Missing successful claim, grant, clock, export or original outcome pin.')
    for name in pins:
        p = Path(name)
        if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
            raise ValueError('Unsafe preservation path.')
    return data


def targets(data):
    validate_input(data)
    return {TARGET: data['review']['text'].encode()}


def load(packet, pin, home):
    packet, transfer = Path(packet).resolve(), (Path(home) / BASE).resolve()
    if packet != (transfer / PREFIX / 'unpacked' / PREFIX).resolve():
        raise ValueError('Use the fixed source-only review registration packet.')
    op = json.loads((packet / 'SYNC.json').read_text()); validate_operation(op, pin)
    for name, h in op['sources_sha256'].items():
        if digest(packet / name) != h:
            raise ValueError('Registration source changed: ' + name)
    if digest(packet / 'INPUT.json') != op['inputs_file_sha256'] or digest(packet / 'activation-receipt.json') != ACTIVATION_RECEIPT:
        raise ValueError('Pinned input or actual recovery activation changed.')
    data = validate_input(json.loads((packet / 'INPUT.json').read_text()))
    core = module('original_completed_archery_recovery', packet / 'support/archery_recovery.py')
    core_packet = transfer / CORE_PREFIX / 'unpacked' / CORE_PREFIX
    # Verify immutable history before any older Registry constructor can run.
    checked = core.boundary(core_packet, CORE_PIN)
    checked['prior'].check_preserved(transfer, data['preserved_files_sha256'])
    context = core.load(core_packet, CORE_PIN, home)
    expected = core.activation_records(context['op'], context['data'], context['prior'])
    r = json.loads((packet / 'activation-receipt.json').read_text())
    core.verify_activation(r, context['op'], context['data'], context['prior'])
    for name, text in expected.items():
        if data['preserved_files_sha256'].get(CORE_ROOT + '/' + name) != hashlib.sha256(text).hexdigest():
            raise ValueError('Original activated imports, grant or request changed.')
    for name, h in core.ancestor_pins(context).items():
        if data['preserved_files_sha256'].get(name) != h:
            raise ValueError('Historical 470-file boundary changed.')
    return {'packet': packet, 'op': op, 'data': data, 'core': core, 'context': context,
        'ops': context['ops'], 'prior': context['prior'], 'root': context['root'], 'transfer': transfer}


def check_preserved(c):
    return c['prior'].check_preserved(c['transfer'], c['data']['preserved_files_sha256'])


def time_readback(data):
    a = accounting(data)
    return {'case_id': CASE, 'attempt': ATTEMPT, 'accounting_file_sha256': ACCOUNTING,
        'consumed_grant_id': GRANT, 'authorization_file_sha256': AUTH,
        'previous_charged_seconds': 2348, 'recovery_charged_seconds': 2599, 'total_charged_seconds': 4947,
        'old_remaining_seconds': 2452, 'old_remainder_added': False,
        'unused_seconds_authorize_no_further_attempt': True, 'started_epoch_decimal': a['started_epoch_decimal'],
        'started_UTC_floor': a['started_UTC_floor'], 'ended_UTC_floor': a['ended_UTC_floor']}


def check_time(c):
    a = accounting(c['data']); launcher = c['context']['packet'] / 'launchers' / (HOST + '-' + CASE)
    start = Decimal((launcher / 'started-epoch.txt').read_text().strip())
    start_floor = datetime.fromisoformat((launcher / 'started-utc.txt').read_text().strip().replace('Z', '+00:00'))
    end = datetime.fromisoformat((launcher / 'ended-utc.txt').read_text().strip().replace('Z', '+00:00'))
    if not start.is_finite() or start_floor.utcoffset() != timezone.utc.utcoffset(None) or end.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError('Unknown recovery clock.')
    upper = Decimal(int(end.timestamp())) + 1 - start
    if (str(start) != a['started_epoch_decimal'] or start_floor.isoformat() != a['started_UTC_floor']
            or end.isoformat() != a['ended_UTC_floor'] or not 0 < upper < 4800
            or int(upper.to_integral_value(rounding=ROUND_CEILING)) != 2599
            or Decimal(int(start_floor.timestamp())) + 1 <= start
            or (launcher / 'exit-status.txt').read_text().strip() != '0'):
        raise ValueError('Original consumed recovery clock changed.')
    envelope = c['ops'].read_record(c['root'] / 'cases' / CASE / ATTEMPT / 'case-envelope.json')
    if (envelope['resource_grant'] != c['core'].recovery_grant() or envelope['complete'] is not True
            or envelope['active_role'] is not None or Decimal(str(envelope['launcher_started_epoch'])) != start
            or envelope['prior_plus_new_elapsed_seconds'] != 2348 + envelope['launcher_elapsed_seconds']
            or not 0 < envelope['launcher_elapsed_seconds'] < float(upper)):
        raise ValueError('Recovery grant or saved elapsed time changed.')
    return time_readback(c['data'])


def check_targets(root, values, prior):
    if set(values) != {TARGET}:
        raise ValueError('Only the exact original Archery recovery review may be appended.')
    p = prior.bound(root, TARGET)
    if p.exists():
        if not p.is_file() or p.read_bytes() != values[TARGET]:
            raise ValueError('Conflicting original review; preserve it for inspection.')
        return [TARGET]
    return []


def check_state(status, reviews):
    expected = {'kind': 'matched-panel-archery-recovery-v1-progress',
        'accepted_case_ids': ORDER[:15] if TARGET in reviews else BEFORE,
        'claimed_case_ids': [CASE], 'computed_case_ids': [CASE], 'failed_case_ids': [],
        'stop_dispatch': False, 'complete': False, 'scientific_aggregate': None, 'research_admitted': False}
    if status != expected:
        raise ValueError('Coordinator differs from the one completed recovery and original fourteen imports.')


def verify_saved(c, temporary):
    """Review existing saved tensors on CPU; never load a model or run an attack."""
    from models.matched_panel_case import verify_case
    directory = c['prior'].bound(c['root'], 'cases/' + CASE + '/' + ATTEMPT)
    archive = directory / 'continuation-evidence.tar.gz'
    spec = json.loads(Path(str(archive) + '.manifest.json').read_text()); original = marker(c['data'])
    rebuilt = c['core'].review(c['context'], archive, spec, original['archive_sha256'], Path(temporary) / CASE)
    if rebuilt != original:
        raise ValueError('Saved outer export does not reconstruct the exact original recovery review.')
    receipt = verify_case(directory, plan=c['context']['numeric'], case_id=CASE,
                          execution=c['core'].execution_identity(c['context'], HOST))
    if canonical(receipt) != RECEIPT:
        raise ValueError('Live saved numerical receipt differs from the independently reviewed archive.')
    return 1


def publish(root, values, prior, verify_old):
    verify_old(); existing = check_targets(root, values, prior); created = []
    if not existing:
        prior.exclusive_bytes(prior.bound(root, TARGET), values[TARGET]); created.append(TARGET)
    if prior.bound(root, TARGET).read_bytes() != values[TARGET]:
        raise ValueError('Original review readback mismatch.')
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
        for assigned in c['ops'].HOSTS:
            locks.enter_context(c['ops'].file_lock(c['context']['legacy'] / 'leases' / assigned, wait_seconds=0))
        registry = c['core'].Registry(c['context'])
        with c['ops'].file_lock(c['root'] / '.registry.lock'):
            existing_before = check_targets(c['root'], values, c['prior'])
            before = registry.status(); check_state(before, existing_before); check_preserved(c)
            time_before = check_time(c)
            with tempfile.TemporaryDirectory(prefix='archery-review-sync-CPU-') as temporary:
                checked = verify_saved(c, temporary)
            if checked != 1 or registry.status() != before:
                raise ValueError('Incomplete saved-case review or concurrent coordinator change.')
            created, existing = publish(c['root'], values, c['prior'], lambda: check_preserved(c))
            after = registry.status(); check_state(after, [TARGET]); observed = check_preserved(c)
            time_after = check_time(c)
            if time_after != time_before:
                raise ValueError('Consumed grant or attempt accounting changed during registration.')
            readback = {TARGET: c['prior'].bound(c['root'], TARGET).read_text()}
    receipt = {'kind': KIND + '-receipt', 'operation_sha256': pin, 'inputs_file_sha256': c['op']['inputs_file_sha256'],
        'created': created, 'preexisting_identical': existing, 'records_readback': readback,
        'fresh_preserved_files_sha256': observed, 'status_before': before, 'status_after': after,
        'saved_cases_reverified': checked, 'time_before': time_before, 'time_after': time_after,
        'historical_STOP_records_unchanged': True, 'models_started': False, 'test_suite_started': False,
        'new_GPU_dispatch_authorized': False, 'additional_attempt_authorized': False, 'research_admitted': False}
    verify_receipt(receipt, c['op'], c['data'])
    return receipt


def verify_receipt(receipt, op, data):
    validate_operation(op, canonical(op)); values = targets(data)
    created, existing = receipt['created'], receipt['preexisting_identical']
    if (receipt['kind'] != KIND + '-receipt' or receipt['operation_sha256'] != canonical(op)
            or receipt['inputs_file_sha256'] != op['inputs_file_sha256'] or set(created) & set(existing)
            or set(created) | set(existing) != {TARGET} or len(created) + len(existing) != 1
            or receipt['records_readback'] != {n: b.decode() for n, b in values.items()}
            or receipt['fresh_preserved_files_sha256'] != data['preserved_files_sha256']
            or receipt['saved_cases_reverified'] != 1 or receipt['historical_STOP_records_unchanged'] is not True
            or any(receipt[k] is not False for k in ('models_started', 'test_suite_started',
                'new_GPU_dispatch_authorized', 'additional_attempt_authorized', 'research_admitted'))):
        raise ValueError('Receipt differs from the original recovery review and preserved history.')
    if receipt['time_before'] != time_readback(data) or receipt['time_after'] != time_readback(data):
        raise ValueError('Receipt changes the consumed grant or original attempt accounting.')
    check_state(receipt['status_before'], existing); check_state(receipt['status_after'], [TARGET])
    return {'verified': True, 'accepted_cases': 15, 'complete_clip_pairs': 7, 'remaining_cases': 5,
        'preserved_files': PRESERVED, 'saved_cases_reverified': 1, 'consumed_grant_id': GRANT,
        'previous_charged_seconds': 2348, 'recovery_charged_seconds': 2599, 'total_charged_seconds': 4947,
        'historical_STOP_records_unchanged': True, 'models_started': False, 'new_GPU_dispatch_authorized': False,
        'additional_attempt_authorized': False, 'research_admitted': False}
