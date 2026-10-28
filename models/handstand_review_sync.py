"""Register the original HandStandPushups review without changing attempts or numerical work."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

KIND = 'matched-panel-handstand-review-sync-v1'
BASE = 'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
PREFIX = 'handstand-review-sync-20261028'
CORE_PREFIX = 'bowling-handstand-pair-20261028'
CORE_ROOT = 'bowling-handstand-execution-20261028'
CORE_PIN = 'b899e6d9b2e6824804a035807ff61ea3517182fbbaabf7910d2dada5a34c82e1'
CORE_SOURCE = '6d435b42a87c51c3992ea022ebdb4e366cb3cb7c021e1213c2b15cfa2b8372ad'
ACTIVATION_RECEIPT = 'c78105922962b3e4aff228bf135ef7c5df1514a6c35d38572eaa0fb20ad43d83'
CASES = {'linux10609': 'clip-09-bank-0'}
LEASE_HOSTS = ('linux10609', 'linux10603')
MARKERS = {'clip-09-bank-0': '50aee20e6386d958f293c414fb076fcef4bf09c545a5449cffc4b86378325765'}
EXECUTIONS = {'clip-09-bank-0': '2c5408b45416be7151da07f5c6ddf9c69ddfb0a4347499d6afacc680cc87d0a7'}
RECEIPTS = {'clip-09-bank-0': 'ea77905e4544077ceb204a8c411f5e640e81600ec966c3e6b5350af66a465cad'}
CHARGES = {'clip-09-bank-0': 2951}
PRESERVED = 918
ORDER = [f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0, 1)]
BEFORE = ORDER[:17]
AFTER = [c for c in ORDER if c in set(BEFORE) | {'clip-09-bank-0'}]
TARGETS = {'reviews/' + cid + '.json' for cid in CASES.values()}
SOURCES = ('models/handstand_review_sync.py', 'scripts/sync_handstand_review.py',
           'scripts/prepare_handstand_review_sync.py', 'tests/test_handstand_review_sync.py',
           'tests/fixtures/handstand_review_sync_metadata.json', 'support/reviewed_panel_final_pair.py')
FIXED_CLOCKS = {'clip-09-bank-0': {'host': 'linux10609', 'attempt': 'attempt-01', 'started_epoch_decimal': '1793214611.436341', 'started_UTC_floor': '2026-10-28T19:10:11+00:00', 'ended_UTC_floor': '2026-10-28T19:59:21+00:00', 'conservative_charge_upper_seconds': 2951, 'original_allowance_seconds': 4800, 'kill_grace_seconds': 15, 'additional_attempt_authorized': False, 'unused_time_authorizes_no_retry': True}}
PRIOR_TIME = {'additional_attempt_authorized': False, 'completed_original_attempts': {'clip-07-bank-1': {'additional_attempt_authorized': False, 'attempt': 'attempt-01', 'conservative_charge_upper_seconds': 2794, 'ended_UTC_floor': '2026-10-28T06:48:44+00:00', 'host': 'linux10609', 'kill_grace_seconds': 15, 'original_allowance_seconds': 4800, 'started_UTC_floor': '2026-10-28T06:02:11+00:00', 'started_epoch_decimal': '1793167331.0951025', 'unused_time_authorizes_no_retry': True}, 'clip-08-bank-0': {'additional_attempt_authorized': False, 'attempt': 'attempt-01', 'conservative_charge_upper_seconds': 2634, 'ended_UTC_floor': '2026-10-28T06:45:29+00:00', 'host': 'linux10603', 'kill_grace_seconds': 15, 'original_allowance_seconds': 4800, 'started_UTC_floor': '2026-10-28T06:01:37+00:00', 'started_epoch_decimal': '1793167296.9180472', 'unused_time_authorizes_no_retry': True}}, 'elapsed_time_reset': False, 'previous_consumed_history': {'accounting_file_sha256': '31f1ee6803722e70121cc6d7bd020c11b6c6a7546a207b76e308715b68bb271a', 'attempt': 'recovery-01', 'authorization_file_sha256': '41141726758e31b4babf76feadfe48618b4399a5147ace7dcea614ea14feabc3', 'case_id': 'clip-06-bank-1', 'consumed_grant_id': 'fd4b4cb8d42e442f8d9ee8ac7130945e', 'ended_UTC_floor': '2026-10-28T04:32:47+00:00', 'old_remainder_added': False, 'old_remaining_seconds': 2452, 'previous_charged_seconds': 2348, 'recovery_charged_seconds': 2599, 'started_UTC_floor': '2026-10-28T03:49:29+00:00', 'started_epoch_decimal': '1793159369.0821397', 'total_charged_seconds': 4947, 'unused_seconds_authorize_no_further_attempt': True}}
BOWLING_STARTUP = {'additional_allowance_seconds': 0, 'basis': 'Original whole-bootstrap epoch; ended UTC has one-second resolution, so round its upper bound upward. No allowance reset.', 'case_id': 'clip-08-bank-1', 'claim_and_case_absent_at_snapshot': True, 'conservative_startup_charge_seconds': 206, 'elapsed_lower_bound_seconds': '204.6155326', 'elapsed_upper_exclusive_seconds': '205.6155326', 'ended_utc_floor': '2026-10-28T19:13:37+00:00', 'host': 'linux10603', 'kind': 'bowling-unclaimed-startup-accounting-v1', 'new_GPU_execution_authorized': False, 'numeric_attempts_started': 0, 'operation_sha256': 'b899e6d9b2e6824804a035807ff61ea3517182fbbaabf7910d2dada5a34c82e1', 'original_case_allowance_seconds': 4800, 'remaining_original_allowance_seconds': 4594, 'research_admitted': False, 'started_epoch_decimal': '1793214612.3844674', 'started_utc_display': '2026-10-28T19:10:12+00:00', 'verified_transport_sha256': 'b8011463210c3764a960e6fbaf0130d58b8f707316c43774c94122140d60aaa0'}
BOWLING_CAPTURE = 'b8011463210c3764a960e6fbaf0130d58b8f707316c43774c94122140d60aaa0'


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
    return deepcopy({'kind': KIND, 'host': 'linux10609', 'core_operation_sha256': CORE_PIN,
        'core_module_sha256': CORE_SOURCE, 'actual_activation_receipt_file_sha256': ACTIVATION_RECEIPT,
        'coordinator_namespace': CORE_ROOT, 'case_ids': list(CASES.values()), 'attempt': 'attempt-01',
        'accepted_after_sync': AFTER, 'original_review_markers_file_sha256': MARKERS,
        'preserved_file_count': PRESERVED, 'max_seconds': 300, 'kill_grace_seconds': 15,
        'original_attempt_charge_upper_seconds': CHARGES, 'previous_consumed_history_seconds': 4947,
        'bowling_startup_charge_seconds': 206, 'bowling_remaining_original_allowance_seconds': 4594, 'metadata_only': True, 'models_started': False, 'test_suite_started': False,
        'new_GPU_dispatch_authorized': False, 'additional_attempt_authorized': False,
        'historical_STOP_records_unchanged': True, 'automatic_queue': False, 'research_admitted': False})


def validate_operation(op, pin):
    if canonical(op) != pin or any(k not in op or canonical(op[k]) != canonical(v) for k, v in policy().items()):
        raise ValueError('Use the pinned metadata-only registration of this original review.')
    if set(op['sources_sha256']) != set(SOURCES) or op['sources_sha256']['support/reviewed_panel_final_pair.py'] != CORE_SOURCE:
        raise ValueError('Incomplete or changed registration source inventory.')


def marker(data, cid):
    if cid not in CASES.values():
        raise ValueError('Unknown completed case.')
    row = data['reviews'][cid]; value = json.loads(row['text'])
    if (set(value) != {'payload', 'sha256'} or canonical(value['payload']) != value['sha256']
            or hashlib.sha256(row['text'].encode()).hexdigest() != row['sha256'] or row['sha256'] != MARKERS[cid]):
        raise ValueError('Use the exact original independent review bytes.')
    host = next(h for h, c in CASES.items() if c == cid)
    fixed = {'kind': 'matched-panel-reviewed-seventeen-pair-v1-independent-review',
        'operation_sha256': CORE_PIN, 'case_id': cid, 'host': host,
        'execution_sha256': EXECUTIONS[cid], 'numeric_receipt_sha256': RECEIPTS[cid],
        'reviewed_export': True, 'research_admitted': False, 'automatic_queue_authorized': False}
    if any(k not in value['payload'] or canonical(value['payload'][k]) != canonical(v) for k, v in fixed.items()):
        raise ValueError('Mixed completed case identity or unsupported marker authority.')
    return value['payload']


def validate_input(data):
    fixed = {'kind': KIND + '-input', 'core_operation_sha256': CORE_PIN,
        'actual_activation_receipt_file_sha256': ACTIVATION_RECEIPT,
        'clocks': FIXED_CLOCKS, 'previous_consumed_history': PRIOR_TIME, 'bowling_startup_accounting': BOWLING_STARTUP, 'bowling_capture_transport_sha256': BOWLING_CAPTURE,
        'models_started': False, 'research_admitted': False}
    if (any(k not in data or canonical(data[k]) != canonical(v) for k, v in fixed.items())
            or set(data['reviews']) != set(CASES.values())):
        raise ValueError('Wrong review, clock, consumed history or registration scope.')
    pins = data['preserved_files_sha256']
    if len(pins) != PRESERVED or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in pins.values()):
        raise ValueError('Preserve all 918 existing source, failure, success and clock hashes.')
    for name in pins:
        p = Path(name)
        if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
            raise ValueError('Unsafe preservation path.')
    for host, cid in CASES.items():
        m = marker(data, cid); case = CORE_ROOT + '/cases/' + cid + '/attempt-01/'
        required = [case + n for n in ('case-envelope.json', 'receipt.json', 'session.json',
            'preservation-before-export.json', 'preservation-after-export.json', 'numeric-evidence.tar.gz',
            'numeric-evidence.tar.gz.manifest.json', 'continuation-evidence.tar.gz.manifest.json')]
        required += [CORE_ROOT + '/' + n for n in ('claims/' + cid + '/claim.json', 'exports/' + cid + '.json')]
        launcher = CORE_PREFIX + '/unpacked/' + CORE_PREFIX + '/launchers/' + host + '-' + cid + '/'
        required += [launcher + n for n in ('started-epoch.txt', 'started-utc.txt', 'ended-utc.txt', 'exit-status.txt',
            'cpu-results.xml', 'cpu-summary.log', 'case.log', 'source-launch.sh', 'source-pipeline.sh')]
        if (any(n not in pins for n in required) or CORE_ROOT + '/reviews/' + cid + '.json' in pins
                or pins.get(case + 'continuation-evidence.tar.gz') != m['archive_sha256']
                or pins.get(CORE_ROOT + '/outcomes/' + cid + '.json') != m['outcome_file_sha256']):
            raise ValueError('Missing original successful claim, clock, export or outcome pin.')
    stopped = CORE_PREFIX + '/unpacked/' + CORE_PREFIX + '/launchers/linux10603-clip-08-bank-1/'
    required_startup = ('started-epoch.txt', 'started-utc.txt', 'ended-utc.txt', 'exit-status.txt',
        'case.log', 'cpu.log', 'cpu-old.log', 'cpu-new.log', 'cpu-summary.log',
        'cpu-old.xml', 'cpu-new.xml', 'cpu-results.xml', 'source-launch.sh', 'source-pipeline.sh')
    if any(stopped + n not in pins for n in required_startup):
        raise ValueError('Preserve all fourteen Bowling startup evidence files.')
    return data


def targets(data):
    validate_input(data)
    return {'reviews/' + cid + '.json': data['reviews'][cid]['text'].encode() for cid in CASES.values()}


def load(packet, pin, home):
    packet, transfer = Path(packet).resolve(), (Path(home) / BASE).resolve()
    if packet != (transfer / PREFIX / 'unpacked' / PREFIX).resolve():
        raise ValueError('Use the fixed source-only review registration packet.')
    op = json.loads((packet / 'SYNC.json').read_text()); validate_operation(op, pin)
    for name, h in op['sources_sha256'].items():
        if digest(packet / name) != h:
            raise ValueError('Registration source changed: ' + name)
    if digest(packet / 'INPUT.json') != op['inputs_file_sha256'] or digest(packet / 'activation-receipt.json') != ACTIVATION_RECEIPT:
        raise ValueError('Pinned input or actual activation changed.')
    data = validate_input(json.loads((packet / 'INPUT.json').read_text()))
    core = module('original_completed_handstand', packet / 'support/reviewed_panel_final_pair.py')
    core_packet = transfer / CORE_PREFIX / 'unpacked' / CORE_PREFIX
    # Fail closed before an older Registry constructor could initialize missing history.
    checked = core.boundary(core_packet, CORE_PIN)
    checked['prior'].check_preserved(transfer, data['preserved_files_sha256'])
    context = core.load(core_packet, CORE_PIN, home)
    expected = core.activation_records(context['op'], context['data'], context['prior'])
    receipt = json.loads((packet / 'activation-receipt.json').read_text())
    core.verify_activation(receipt, context['op'], context['data'], context['prior'])
    for name, content in expected.items():
        if data['preserved_files_sha256'].get(CORE_ROOT + '/' + name) != hashlib.sha256(content).hexdigest():
            raise ValueError('Original activated import, resolution or request changed.')
    for name, h in core.ancestor_pins(context).items():
        if data['preserved_files_sha256'].get(name) != h:
            raise ValueError('Historical 774-file boundary changed.')
    if context['sync_receipt']['time_after'] != PRIOR_TIME:
        raise ValueError('Previous completed attempts or consumed Archery grant history changed.')
    return {'packet': packet, 'op': op, 'data': data, 'core': core, 'context': context,
        'ops': context['ops'], 'prior': context['prior'], 'root': context['root'], 'transfer': transfer}


def check_preserved(c):
    return c['prior'].check_preserved(c['transfer'], c['data']['preserved_files_sha256'])


def time_readback(data):
    validate_input(data)
    return deepcopy({'completed_original_attempts': data['clocks'], 'previous_consumed_history': PRIOR_TIME, 'bowling_startup_accounting': BOWLING_STARTUP, 'bowling_capture_transport_sha256': BOWLING_CAPTURE,
        'elapsed_time_reset': False, 'additional_attempt_authorized': False})


def check_time(c):
    expected = time_readback(c['data'])
    for host, cid in CASES.items():
        a = c['data']['clocks'][cid]
        launcher = c['context']['packet'] / 'launchers' / (host + '-' + cid)
        start = Decimal((launcher / 'started-epoch.txt').read_text().strip())
        shown = datetime.fromisoformat((launcher / 'started-utc.txt').read_text().strip().replace('Z', '+00:00'))
        end = datetime.fromisoformat((launcher / 'ended-utc.txt').read_text().strip().replace('Z', '+00:00'))
        if not start.is_finite() or shown.utcoffset() != timezone.utc.utcoffset(None) or end.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError('Unknown original case clock.')
        upper = Decimal(int(end.timestamp())) + 1 - start
        if (str(start) != a['started_epoch_decimal'] or shown.isoformat() != a['started_UTC_floor']
                or end.isoformat() != a['ended_UTC_floor'] or not 0 < upper <= 4800
                or int(upper.to_integral_value(rounding=ROUND_CEILING)) != CHARGES[cid]
                or Decimal(int(shown.timestamp())) + 1 <= start
                or (launcher / 'exit-status.txt').read_text().strip() != '0'):
            raise ValueError('Original completed attempt clock changed.')
        envelope = c['ops'].read_record(c['root'] / 'cases' / cid / 'attempt-01' / 'case-envelope.json')
        if (envelope['complete'] is not True or envelope['active_role'] is not None
                or envelope['case_seconds'] != 4800 or Decimal(str(envelope['launcher_started_epoch'])) != start
                or not 0 < envelope['launcher_elapsed_seconds'] < float(upper)):
            raise ValueError('Original completed attempt or consumed elapsed time changed.')
    check_bowling(c)
    return expected


def check_bowling(c):
    expected = c['data']['bowling_startup_accounting']
    launcher = c['context']['packet'] / 'launchers/linux10603-clip-08-bank-1'
    start = Decimal((launcher / 'started-epoch.txt').read_text().strip())
    shown = datetime.fromisoformat((launcher / 'started-utc.txt').read_text().strip().replace('Z', '+00:00'))
    end = datetime.fromisoformat((launcher / 'ended-utc.txt').read_text().strip().replace('Z', '+00:00'))
    upper = Decimal(int(end.timestamp())) + 1 - start
    if (not start.is_finite() or str(start) != expected['started_epoch_decimal']
            or shown.isoformat() != expected['started_utc_display'] or end.isoformat() != expected['ended_utc_floor']
            or int(upper.to_integral_value(rounding=ROUND_CEILING)) != 206
            or 4800 - 206 != expected['remaining_original_allowance_seconds']
            or (launcher / 'exit-status.txt').read_text().strip() != '1'):
        raise ValueError('Original Bowling preclaim startup time changed.')
    if 'ValueError: Another CUDA compute process is active on this host.' not in (launcher / 'case.log').read_text():
        raise ValueError('Original Bowling preclaim refusal is missing.')
    candidates = []
    for cid in ('clip-08-bank-1', 'clip-09-bank-1'):
        for namespace in (*c['core'].HISTORICAL_ROOTS, CORE_ROOT):
            root = c['transfer'] / namespace
            candidates += [root / n / cid for n in ('claims', 'cases')]
            candidates += [root / n / (cid + '.json') for n in ('outcomes', 'exports', 'reviews')]
    v1 = c['context']['package'] / 'runs/panel.qIXeDn'
    for cid in ('clip-08-bank-1', 'clip-09-bank-1'):
        candidates += [v1 / 'cases' / cid, v1 / 'operator-attended' / cid]
        candidates += list((v1 / 'exports').glob(cid + '-*'))
    if any(p.exists() or p.is_symlink() for p in candidates):
        raise ValueError('Pending final-case artifacts exist; preserve evidence before continuing.')
    return deepcopy(expected)


def check_targets(root, values, prior):
    if set(values) != TARGETS:
        raise ValueError('Only the exact original HandStandPushups review may be appended.')
    existing = []
    for name in sorted(values):
        p = prior.bound(root, name)
        if p.exists() or p.is_symlink():
            if p.is_symlink() or not p.is_file() or p.read_bytes() != values[name]:
                raise ValueError('Conflicting or redirected original review; preserve it for inspection.')
            existing.append(name)
    return existing


def check_state(status, reviews):
    if not set(reviews) <= TARGETS:
        raise ValueError('Unknown review target.')
    accepted = set(BEFORE) | {cid for cid in CASES.values() if 'reviews/' + cid + '.json' in reviews}
    expected = {'kind': 'matched-panel-reviewed-seventeen-pair-v1-progress',
        'accepted_case_ids': [c for c in ORDER if c in accepted],
        'claimed_case_ids': list(CASES.values()), 'computed_case_ids': list(CASES.values()), 'failed_case_ids': [],
        'stop_dispatch': False, 'complete': False, 'scientific_aggregate': None, 'research_admitted': False}
    if status != expected:
        raise ValueError('Coordinator differs from the one completed case and seventeen original imports.')


def verify_saved(c, temporary):
    """Reverify existing tensors on CPU; do not construct a model or run an attack."""
    from models.matched_panel_case import verify_case
    for host, cid in CASES.items():
        directory = c['prior'].bound(c['root'], 'cases/' + cid + '/attempt-01')
        archive = directory / 'continuation-evidence.tar.gz'
        spec = json.loads(Path(str(archive) + '.manifest.json').read_text()); original = marker(c['data'], cid)
        rebuilt = c['core'].review(c['context'], archive, spec, original['archive_sha256'], Path(temporary) / cid)
        if rebuilt != original:
            raise ValueError('Saved outer export does not reconstruct its exact original independent review.')
        receipt = verify_case(directory, plan=c['context']['numeric'], case_id=cid,
                              execution=c['core'].execution_identity(c['context'], host))
        if canonical(receipt) != RECEIPTS[cid]:
            raise ValueError('Live numerical receipt differs from the independently reviewed archive.')
    return 1


def publish(root, values, prior, verify_old):
    verify_old(); existing = check_targets(root, values, prior); created = []
    for name in sorted(values):
        if name not in existing:
            verify_old()
            prior.exclusive_bytes(prior.bound(root, name), values[name]); created.append(name)
        if prior.bound(root, name).read_bytes() != values[name]:
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
        for assigned in LEASE_HOSTS:
            locks.enter_context(c['ops'].file_lock(c['context']['legacy'] / 'leases' / assigned, wait_seconds=0))
        registry = c['core'].Registry(c['context'])  # Outside its corresponding mutex.
        with c['ops'].file_lock(c['root'] / '.registry.lock'):
            existing_before = check_targets(c['root'], values, c['prior'])
            before = registry.status(); check_state(before, existing_before); check_preserved(c)
            time_before = check_time(c)
            with tempfile.TemporaryDirectory(prefix='handstand-review-sync-CPU-') as temporary:
                checked = verify_saved(c, temporary)
            if checked != 1 or registry.status() != before:
                raise ValueError('Incomplete saved-case check or concurrent coordinator change.')
            created, existing = publish(c['root'], values, c['prior'], lambda: check_preserved(c))
            after = registry.status(); check_state(after, TARGETS); observed = check_preserved(c)
            time_after = check_time(c)
            if time_after != time_before:
                raise ValueError('Original attempt or consumed history changed during registration.')
            readback = {name: c['prior'].bound(c['root'], name).read_text() for name in sorted(values)}
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
            or set(created) | set(existing) != TARGETS or len(created) + len(existing) != 1
            or receipt['records_readback'] != {n: b.decode() for n, b in values.items()}
            or receipt['fresh_preserved_files_sha256'] != data['preserved_files_sha256']
            or receipt['saved_cases_reverified'] != 1 or receipt['historical_STOP_records_unchanged'] is not True
            or any(receipt[k] is not False for k in ('models_started', 'test_suite_started',
                'new_GPU_dispatch_authorized', 'additional_attempt_authorized', 'research_admitted'))):
        raise ValueError('Receipt differs from the original review and preserved history.')
    if receipt['time_before'] != time_readback(data) or receipt['time_after'] != time_readback(data):
        raise ValueError('Receipt changes original clocks or consumed attempt history.')
    check_state(receipt['status_before'], existing); check_state(receipt['status_after'], TARGETS)
    return {'verified': True, 'accepted_cases': 18, 'complete_clip_pairs': 8, 'remaining_cases': 2,
        'preserved_files': PRESERVED, 'saved_cases_reverified': 1,
        'original_attempt_charge_upper_seconds': CHARGES, 'previous_consumed_history_seconds': 4947,
        'historical_STOP_records_unchanged': True, 'models_started': False, 'new_GPU_dispatch_authorized': False,
        'additional_attempt_authorized': False, 'research_admitted': False}
