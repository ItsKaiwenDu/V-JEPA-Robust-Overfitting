"""Read-only descriptive analysis of the twenty originally registered diagnostic cases."""
import hashlib
import json
from pathlib import Path

KIND = 'registered-matched-panel-analysis-v1'
AUTHORITY = 'report/checks/2026-10-28-final-review-sync-verification.json'
AUTHORITY_SHA = '06e5d8ec8873b41a29cce535212c140c075b95bfcbf8e96bc426abcec8ea5f5a'
REGISTRATION_SHA = 'dd1644fad315867fa9a990047c59163dfc28716995aa299d96a2c4f93935a980'
REGISTRATION_OP = 'e3f12b9d5f88816e22bed8aeaefb20eded97904692e1a1cbf65b3e37a8d571c5'
PLAN_SHA = 'f6b81dec841aeee703413dba23ea16afe3dbf14736edecadc0f7830f3ded34ae'
ORDER = tuple(f'clip-{i:02d}-bank-{b}' for i in range(10) for b in (0, 1))
BANKS = ('main', 'audit_0', 'audit_1')
CONTRASTS = ('B-A', 'D-C', 'C-A', 'D-B')
SNAPSHOT = 'runs/pretrained/diagnostic/matched-panel-handoff-20261024/snapshot'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def path_in(root, relative):
    root, name = Path(root).resolve(), Path(relative)
    path = root / name
    if (not relative or name.is_absolute() or '..' in name.parts or str(name) != relative
            or '\\' in relative or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in (path, *(root / Path(*name.parts[:i]) for i in range(1, len(name.parts)))))):
        raise ValueError('Use an unredirected repository-relative artifact path.')
    return path


def json_file(path):
    return json.loads(Path(path).read_text())


def record(path):
    value = json_file(path)
    if set(value) != {'payload', 'sha256'} or canonical(value['payload']) != value['sha256']:
        raise ValueError('Invalid checksummed evidence record: ' + str(path))
    return value['payload']


def require_complete(rows):
    if set(rows) != set(ORDER):
        raise ValueError('All twenty original registered cases are required; no partial aggregate.')


def authority(root):
    report_path = path_in(root, AUTHORITY)
    if digest(report_path) != AUTHORITY_SHA:
        raise ValueError('Use the independently verified actual twenty-case registration.')
    report = json_file(report_path)
    if (report['verified'] is not True or report['registered_university_case_count'] != 20
            or report['complete_clip_pairs'] != 10 or report['research_admitted'] is not False
            or report['scientific_aggregate'] is not None):
        raise ValueError('Registration authority does not admit this complete diagnostic collection.')
    require_complete(report['registered_case_index'])
    session = path_in(root, report['actual_session'])
    if digest(session / 'receipt.json') != REGISTRATION_SHA or (session / 'connection-exit-status.txt').read_text().strip() != '0':
        raise ValueError('Actual registration receipt or successful connection changed.')
    receipt = json_file(session / 'receipt.json')
    if (receipt['operation_sha256'] != REGISTRATION_OP
            or receipt['status_after']['accepted_case_ids'] != list(ORDER)
            or receipt['status_after']['complete'] is not True
            or receipt['saved_cases_reverified'] != 2 or receipt['time_before'] != receipt['time_after']
            or any(receipt[k] is not False for k in ('models_started', 'test_suite_started',
                'new_GPU_dispatch_authorized', 'additional_attempt_authorized', 'research_admitted'))):
        raise ValueError('Actual registered state or preserved time differs.')
    fingerprints = {AUTHORITY: AUTHORITY_SHA}
    for field, sha_field in [('independent_proof_path', 'independent_proof_sha256'),
                           ('review_provenance_path', 'review_provenance_sha256')]:
        path = path_in(root, report[field])
        if digest(path) != report[sha_field]:
            raise ValueError('Independent actual-registration proof changed.')
        fingerprints[report[field]] = report[sha_field]
    provenance = json_file(path_in(root, report['review_provenance_path']))
    for name, expected in provenance['session_files_sha256'].items():
        path = path_in(root, report['actual_session'] + '/' + name)
        if digest(path) != expected:
            raise ValueError('Actual registration session provenance changed: ' + name)
        fingerprints[str(path.relative_to(root))] = expected
    return report, fingerprints


def validate_manifest(manifest, report):
    if (manifest['kind'] != KIND + '-inputs' or manifest['authority_path'] != AUTHORITY
            or manifest['authority_sha256'] != AUTHORITY_SHA or manifest['plan_sha256'] != PLAN_SHA
            or manifest['research_admitted'] is not False):
        raise ValueError('Input manifest changed the frozen diagnostic scope or registration authority.')
    require_complete(manifest['cases'])
    for cid in ORDER:
        if manifest['cases'][cid]['registered_identity'] != report['registered_case_index'][cid]:
            raise ValueError('Original case origin or execution identity changed: ' + cid)


def bind_identity(cid, identity, execution, receipt):
    if (identity['case_id'] != cid or execution['domain'] != 'university-cuda'
            or execution['research_admitted'] is not False or canonical(execution) != identity['execution_sha256']
            or execution['expected_host'] != identity['host'] or receipt['case_id'] != cid
            or receipt['execution_sha256'] != identity['execution_sha256'] or receipt['plan_sha256'] != PLAN_SHA
            or receipt['fresh_A_replay_exact'] is not True or receipt['research_admitted'] is not False):
        raise ValueError('Mixed case, host, execution identity or unsupported fixture evidence.')


def source_location(root, directory, name):
    from models.matched_panel_io import RUNNER_SOURCES
    # Exports deliberately include only RUNNER_SOURCES; original dependencies
    # remain in the single checksummed scientific handoff. Never choose a fallback.
    return path_in(directory, 'source/' + name) if name in RUNNER_SOURCES else path_in(root, SNAPSHOT + '/' + name)


def load_case(root, plan, cid, item):
    from models.matched_panel_case import verify_case
    identity = item['registered_identity']
    directory = path_in(root, item['numeric_directory'])
    receipt_path = directory / 'receipt.json'
    if digest(receipt_path) != identity['receipt_file_sha256']:
        raise ValueError('Original registered numerical receipt bytes changed: ' + cid)
    saved = record(receipt_path)
    execution = record(directory / 'execution.json')
    bind_identity(cid, identity, execution, saved)
    if digest(directory / 'PANEL-HANDOFF.json') != execution['handoff_manifest_sha256']:
        raise ValueError('Original shared scientific handoff identity changed.')
    marker_path = path_in(root, item['review_marker'])
    if digest(marker_path) != identity['review_file_sha256']:
        raise ValueError('Original registered review marker changed: ' + cid)
    marker = record(marker_path)
    if marker['reviewed_export'] is not True or marker['research_admitted'] is not False:
        raise ValueError('Unreviewed or research-admitting marker.')
    if identity['origin'] == 'frozen-v1':
        if cid not in ORDER[:7] or marker['case_id'] != 'clip-03-bank-0' or marker['completed_case_count'] != 7:
            raise ValueError('The seven-case original bootstrap marker was applied to another case.')
    elif marker['case_id'] != cid or marker['execution_sha256'] != identity['execution_sha256']:
        raise ValueError('Review marker does not belong to this original case.')
    archive = path_in(root, item['archive'])
    sidecar = path_in(root, item['sidecar'])
    if digest(archive) != identity['archive_sha256'] or digest(sidecar) != identity['archive_manifest_sha256']:
        raise ValueError('Original independently reviewed archive or sidecar changed: ' + cid)
    manifest = json_file(sidecar)
    if manifest['sha256'] != identity['archive_sha256'] or manifest['bytes'] != archive.stat().st_size:
        raise ValueError('Archive size or manifest disagrees with the registered case.')
    fingerprints = {str(p.relative_to(root)): digest(p) for p in (receipt_path, directory / 'execution.json', directory / 'PANEL-HANDOFF.json', marker_path, archive, sidecar)}
    # Verify the actual imported verifier matches each case's frozen scientific source.
    import importlib
    for name in ('models.matched_panel_case', 'models.matched_panel', 'models.matched_panel_io'):
        active = importlib.import_module(name)
        if digest(active.__file__) != execution['source_sha256'][name.replace('.', '/') + '.py']:
            raise ValueError('Current verifier or aggregation library differs from original scientific source.')
    for name, expected in execution['source_sha256'].items():
        path = source_location(root, directory, name)
        if digest(path) != expected:
            raise ValueError('Captured original scientific source changed: ' + name)
        fingerprints[str(path.relative_to(root))] = expected
    rebuilt = verify_case(directory, plan=plan, case_id=cid, execution=execution)
    if canonical(rebuilt) != canonical(saved):
        raise ValueError('Current saved tensors do not reconstruct the original registered receipt: ' + cid)
    for name, expected in saved['worker_files_sha256'].items():
        path = path_in(directory, name)
        if digest(path) != expected:
            raise ValueError('Original numerical worker artifact changed: ' + cid)
        fingerprints[str(path.relative_to(root))] = expected
    fingerprints[str((directory / 'session.json').relative_to(root))] = saved['session_sha256']
    return saved, fingerprints


def verify_pairs(plan, receipts):
    require_complete(receipts)
    checks = []
    for a, b in zip(plan['cases'][::2], plan['cases'][1::2]):
        ca, cb = (receipts[case['case_id']]['contract'] for case in (a, b))
        for name in ('clip', 'runtime', 'weights', 'settings', 'clean_sha256', 'targets_sha256', 'starts_sha256'):
            if ca[name] != cb[name]:
                raise ValueError('Technical banks changed shared clip/weights/runtime/targets/starts: ' + name)
        shared_a = {n: v for n, v in ca['mask_tensors_sha256'].items() if not n.startswith('attack_')}
        shared_b = {n: v for n, v in cb['mask_tensors_sha256'].items() if not n.startswith('attack_')}
        if shared_a != shared_b:
            raise ValueError('Independent scoring mask tensors differ within a clip.')
        if a['attack_seeds'] == b['attack_seeds'] or a['attack_bank'] != 0 or b['attack_bank'] != 1:
            raise ValueError('Two distinct technical attack banks are required.')
        for arm in 'ABCD':
            for bank in BANKS:
                if receipts[a['case_id']]['scores'][arm][bank]['clean'] != receipts[b['case_id']]['scores'][arm][bank]['clean']:
                    raise ValueError('Paired clean scoring baseline changed.')
        checks.append(dict(clip_index=a['clip_index'], video_id=a['video_id'], case_ids=[a['case_id'], b['case_id']],
            common_targets_starts_runtime_weights_and_scoring_masks=True, distinct_attack_banks=True,
            original_execution_ids=[receipts[a['case_id']]['execution_sha256'], receipts[b['case_id']]['execution_sha256']]))
    if len(checks) != 10:
        raise ValueError('The complete ten-clip matched panel is required.')
    return checks


def collect(plan, items, loader):
    require_complete(items)
    receipts, fingerprints = {}, {}
    for cid in ORDER:
        receipts[cid], observed = loader(cid, items[cid])
        fingerprints.update(observed)
    checks = verify_pairs(plan, receipts)
    return receipts, checks, fingerprints


def analyze(root, manifest_path, expected_sha):
    from models.matched_panel import aggregate_scores
    from models.matched_panel_io import load_plan
    from models.matched_panel_plan import plan_digest
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_relative_to(root) or digest(manifest_path) != expected_sha:
        raise ValueError('Use the externally pinned complete-panel input manifest.')
    manifest = json_file(manifest_path)
    report, fingerprints = authority(root)
    validate_manifest(manifest, report)
    plan = load_plan(root)
    if plan_digest(plan) != PLAN_SHA:
        raise ValueError('Frozen panel plan changed.')
    receipts, pairs, observations = collect(plan, manifest['cases'], lambda cid, item: load_case(root, plan, cid, item))
    fingerprints.update(observations)
    fingerprints[str(manifest_path.relative_to(root))] = expected_sha
    fingerprints['configs/matched_panel_diagnostic_v1.json'] = digest(root / 'configs/matched_panel_diagnostic_v1.json')
    scores = {cid: receipts[cid]['scores'] for cid in ORDER}
    aggregate = aggregate_scores(plan, scores)
    # An independent arithmetic implementation is checked by the caller before publication.
    for name, expected in fingerprints.items():
        if digest(path_in(root, name)) != expected:
            raise ValueError('An input changed during read-only analysis: ' + name)
    import torch
    return dict(kind=KIND, complete=True, diagnostic_only=True, plan_sha256=PLAN_SHA,
        authority_path=AUTHORITY, authority_sha256=AUTHORITY_SHA, actual_registration_receipt_sha256=REGISTRATION_SHA,
        input_manifest_path=str(manifest_path.relative_to(root)), input_manifest_sha256=expected_sha,
        independently_reviewed_and_registered_cases=20, distinct_clips=10, technical_banks_per_clip=2,
        original_registered_case_identities=report['registered_case_index'], paired_bank_checks=pairs,
        scores=scores, case_receipt_sha256={cid: canonical(receipts[cid]) for cid in ORDER},
        saved_evidence_files_sha256=fingerprints, scientific_aggregate=aggregate,
        analysis_rules=plan['analysis'], numerical_settings=plan['settings'],
        CPU_random_starts_regenerated_in_matching_torch_version=all(str(torch.__version__) == row['contract']['runtime']['torch'] for row in receipts.values()),
        random_start_note='Saved starts and hashes are verified; CPU starts are regenerated only when the reader torch version matches the recorded runtime. No cross-version RNG requirement is added.',
        new_model_forward_or_GPU_work=False, new_pixel_decode=False, held_out_pixels_opened=False,
        paid_compute=False, automatic_primary_choice=False, automatic_convergence_pass=False,
        research_admitted=False, canonical_evaluation_changed=False,
        source_sha256={name: digest(root / name) for name in ('models/registered_panel_analysis.py',
            'scripts/analyze_registered_panel.py', 'tests/test_registered_panel_analysis.py',
            'models/matched_panel.py', 'models/matched_panel_case.py', 'models/matched_panel_io.py')})


def markdown(result):
    lines = ['# Complete matched diagnostic panel', '',
        'All20 originally reviewed and registered cases pass the read-only evidence checks.',
        'The analysis unit is the clip (n=10); its two attack banks are technical replicates.',
        'The main and two audit scoring banks remain separate. No attack choice or convergence pass is automatic.', '',
        'A=20steps/EOT4; B=50steps/EOT4; C=20steps/EOT8; D=50steps/EOT8. All use two restarts, step2/255 and epsilon4/255.', '',
        'Positive contrasts mean the higher-setting attack produced greater independent-scoring native L1 loss.',
        'Relative values average case-level percentages within clip, then average ten clips; they are not ratios of pooled means.', '',
        '| Contrast | Scoring bank | Mean L1 difference | Mean relative % | Median | Min | Max | Positive / zero / negative clips |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for contrast in CONTRASTS:
        for bank in BANKS:
            row = result['scientific_aggregate']['summary'][contrast][bank]
            percent = 'undefined' if row['relative_percent_mean'] is None else f"{row['relative_percent_mean']:.4f}"
            lines.append(f"| {contrast} | {bank} | {row['mean']:.8f} | {percent} | {row['median']:.8f} | {row['minimum']:.8f} | {row['maximum']:.8f} | {row['positive_clips']} / {row['zero_clips']} / {row['negative_clips']} |")
    for bank in BANKS:
        lines += ['', '## Per-clip absolute differences: ' + bank, '', '| Clip | B-A | D-C | C-A | D-B |', '| --- | ---: | ---: | ---: | ---: |']
        for row in result['scientific_aggregate']['per_clip']:
            values = ' | '.join(f"{row['contrasts'][contrast][bank]['absolute']:.8f}" for contrast in CONTRASTS)
            lines.append(f"| {row['video_id']} | {values} |")
    lines += ['', 'The structured JSON retains all case-level clean/adversarial losses, paired contrasts, relative gaps, exact original identities and evidence hashes.',
        'This pretrained training-only diagnostic is separate from research step-size calibration, numerical-policy adoption, optimizer validation and budget admission.',
        'It contains no training trajectory, held-out evaluation, statistical significance claim or robust-overfitting conclusion.', '']
    return '\n'.join(lines)
