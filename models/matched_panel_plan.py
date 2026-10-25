"""Metadata-only predeclaration for a broader, separately named attack check.

This module does not read pixel/weight tensors, query a device or launch work.
The plan is a specification; a future runner needs separate qualification.
"""

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from models.evaluation import evaluation_seed

KIND = 'training-only-matched-panel-plan-v1'
BASE_SEED = 20261024
CLIPS, BANKS, RESTARTS = 10, 2, 2
MODULUS = 2**31 - 1
ARMS = {'A': {'steps': 20, 'attack_masks': 4}, 'B': {'steps': 50, 'attack_masks': 4},
        'C': {'steps': 20, 'attack_masks': 8}, 'D': {'steps': 50, 'attack_masks': 8}}
REFERENCE = 'report/checks/2026-10-24-university-matched.json'
REFERENCE_SHA256 = 'a34565a9c08c635df8b7f8f560f6a8bb28e7ca4b70c7c5781631fabb2154decf'
STUDY_SHA256 = '4a16565ead0e3929a0a707fd71cfff95c0c99da93bef7510a3b0f385c4efe3b1'
PROPOSAL_SHA256 = '70de4b44a71efe8e7ee45094f525992c9ea8f8ef45ec0de999ea6047a5f888fa'
NEW_SOURCES = ('models/matched_panel_plan.py', 'scripts/prepare_matched_panel_plan.py')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def plan_digest(plan):
    data = json.dumps(plan, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode('ascii')).hexdigest()


def clip_seeds(video_id):
    def draws(namespace, count):
        return [evaluation_seed(BASE_SEED, 'matched_panel_v1_' + namespace, video_id, draw=i)
                for i in range(count)]
    return {'attack_banks': {str(b): draws('attack_bank_' + str(b), 8) for b in range(BANKS)},
            'scoring_banks': {name: draws(name, 4) for name in ('main', 'audit_0', 'audit_1')},
            'start': evaluation_seed(BASE_SEED, 'matched_panel_v1_start', video_id)}


def validate_seeds(seed_specs, prior_seeds):
    full = [value for spec in seed_specs for group in ('attack_banks', 'scoring_banks')
            for values in spec[group].values() for value in values]
    if (len(full) != CLIPS * (BANKS * 8 + 12)
            or any(type(s) is not int or not 0 <= s < 2**63 for s in full)
            or len(set(full)) != len(full) or len({s % MODULUS for s in full}) != len(full)):
        raise ValueError('Panel mask seeds collide in the full or actual upstream seed space.')
    historical = [s for group in prior_seeds.values() if isinstance(group, list) for s in group]
    if {s % MODULUS for s in full} & {s % MODULUS for s in historical}:
        raise ValueError('New panel mask draws collide with the historical one-clip bank.')
    starts = [s['start'] for s in seed_specs]
    if (any(type(s) is not int or not 0 <= s < 2**63 for s in starts)
            or len(set(starts)) != CLIPS or prior_seeds.get('start') in starts):
        raise ValueError('Panel start seeds collide or reuse the historical start.')


def compile_plan(training_rows, monitor_rows, pixel_entries, *, identity, reference):
    """Compile fixed metadata and scheduling scenarios; never select from losses."""
    if len(training_rows) != 100 or len(monitor_rows) != 20:
        raise ValueError('Keep the original 100-training/20-monitor, 20-class design.')
    training = {r['video_id']: r for r in training_rows}
    if (len(training) != 100 or len({r['video_id'] for r in monitor_rows}) != 20
            or len({r['class_id'] for r in monitor_rows}) != 20
            or any(training.get(r['video_id']) != r for r in monitor_rows)
            or any(n != 5 for n in Counter(r['class_id'] for r in training_rows).values())
            or set(r['class_id'] for r in training_rows) != set(r['class_id'] for r in monitor_rows)):
        raise ValueError('Monitor identities/class membership differ from actual training.')
    if len({r['source_group_id'] for r in monitor_rows}) != 20:
        raise ValueError('Monitoring rows must keep distinct original source groups.')
    by_id = {r['video_id']: r for r in pixel_entries}
    if len(pixel_entries) != 100 or len(by_id) != 100 or set(by_id) != set(training):
        raise ValueError('Training cache index identities are incomplete or duplicated.')
    selected, specs = [], []
    for index, row in enumerate(monitor_rows[:CLIPS]):
        entry = by_id[row['video_id']]
        name = PurePosixPath(entry['file'])
        if (any(entry.get(k) != value for k, value in row.items()) or name.name != str(name)
                or name.suffix != '.pt' or '\\' in str(name)
                or not re.fullmatch('[0-9a-f]{64}', entry['pixel_sha256'])):
            raise ValueError('Selected monitor cache metadata, pixel hash or filename is invalid.')
        selected.append({'clip_index': index, **row, 'prepared_file': 'train/' + str(name),
                         'pixel_sha256': entry['pixel_sha256']})
        specs.append(clip_seeds(row['video_id']))
    validate_seeds(specs, reference['seeds'])
    cases = []
    for row, spec in zip(selected, specs):
        for bank in range(BANKS):
            cases.append({'case_id': f"clip-{row['clip_index']:02d}-bank-{bank}",
                          'clip_index': row['clip_index'], 'video_id': row['video_id'], 'attack_bank': bank,
                          'attack_seeds': spec['attack_banks'][str(bank)],
                          'scoring_seeds': spec['scoring_banks'], 'start_seed': spec['start'],
                          'arm_order': list(ARMS), 'fresh_replay_arm': 'A'})
    seconds = reference['session_seconds']
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Use a finite positive verified one-case scheduling reference.')
    return {'kind': KIND, 'schema_version': 1, 'status_at_freeze': 'specified_not_implemented_or_run',
        'diagnostic_only': True, 'research_result': False, 'research_admitted': False,
        'launch_authorized_by_this_file': False, 'paid_compute_allowed': False,
        'identity': copy.deepcopy(identity), 'panel': 'train_monitor', 'selection': 'first-10-original-monitor-rows-in-order',
        'study_training_classes': 20, 'diagnostic_classes': 10, 'ordered_clips': selected, 'cases': cases,
        'seeds': {'base': BASE_SEED, 'schema': 'evaluation-v1-json-sha256-63bit-mask-mod31',
                  'upstream_modulus': MODULUS, 'distinct_panel_mask_draws': 280,
                  'globally_checked_for_collisions': True,
                  'historical_one_clip_draws_and_starts_reused': False},
        'settings': {'arms': copy.deepcopy(ARMS), 'epsilon_255': 4, 'step_size_255': 2, 'restarts': RESTARTS,
                     'scoring_draws_per_bank': 4, 'scoring_audit_banks': 2,
                     'checkpoint': 'released-pretrained-only', 'execution_policy': 'shared-fp64-native',
                     'attack_bank_fixed_across_steps_and_restarts': True,
                     'four_draw_bank': 'first-four-of-the-same-eight-draw-bank',
                     'shared_across_attack_bank_replicates': ['clean_targets', 'start_seed', 'scoring_seeds'],
                     'candidate_selection': 'attack-bank-loss-only-strict-greater-best-over-clean-starts-iterates'},
        'counts': {'clips': CLIPS, 'attack_banks_per_clip': BANKS, 'matrix_cases': 20,
                   'matrix_arms': 80, 'fresh_A_replays': 20, 'total_PGD_arms': 100,
                   'PGD_objective_calls_excluding_posthoc_scoring': 6700, 'input_gradient_calls': 6400},
        'analysis': {'case_contrasts': ['B-A', 'D-C', 'C-A', 'D-B'],
            'bank_order': ['main', 'audit_0', 'audit_1'], 'aggregate': 'mean-two-attack-banks-within-clip-then-equal-mean-ten-clips',
            'report_also': ['all-case-paired-losses', 'per-clip-mean', 'median-min-max-across-clips',
                            'positive-zero-negative-clip-counts', 'relative-gap-to-lower-setting-adversarial-loss'],
            'score_banks_pooled_or_selected': False, 'historical_one_clip_result_included': False,
            'historical_clip_is_unseen': False, 'partial_panel_scientific_aggregate_allowed': False,
            'relative_gap_if_denominator_zero': 'undefined-report-absolute-difference',
            'statistical_independence_unit': 'clip; bank replicates are paired technical replicates',
            'automatic_primary_or_arm_choice': False, 'automatic_attack_convergence_pass': False,
            'numeric_strength_margin_invented': False, 'positive_step_gaps': 'report magnitude and bank consistency for review; no automatic setting change'},
        'integrity_requirements': ['all-20-cases-complete', 'all-four-arms-per-case', 'fresh-A-exact-replay-per-case',
            'exact-shared-starts-and-first-20-step-prefixes', 'attack-bank-best-nondecreasing-with-steps',
            'frozen-clean-target-weights-and-values', 'finite-nonzero-input-gradients', 'pixel-bounds',
            'model-modes-RNG-and-parameter-grads-preserved', 'matching-frozen-plan-source-runtime-data-weights',
            'new-native-CUDA-probe-in-each-fresh-worker', 'all-artifact-and-archive-hashes-verified'],
        'execution_requirements': {'site': 'reviewed-university-host-only', 'max_parallel_cases': 1,
            'fresh_host_GPU_compute_process_and_free_space_check_required': True,
            'same_host_GPU_UUID_driver_and_software_across_panel': True,
            'allocator_max_bytes': 7 * 2**30, 'allocator_device_fraction_max': .9,
            'matrix_worker_seconds': 3600, 'replay_worker_seconds': 900, 'parent_seconds': 4620,
            'outer_seconds': 4800, 'outer_kill_grace_seconds': 15, 'child_kill_grace_seconds': 10,
            'export_limit_bytes_per_case': 256 * 2**20, 'minimum_free_disk_bytes_before_panel': 12 * 2**30,
            'first_case': 'clip-00-bank-0', 'verify_first_case_before_remaining_queue': True,
            'failure_action': 'stop queue, preserve partial evidence; no automatic retry or tolerance/backend fallback',
            'resume': 'reuse only independently verified whole-case receipts with exact plan/runtime/source identity; new attempt for partial cases',
            'no_PGD_iteration_resume': True, 'case_journal': 'atomic and bound to frozen plan and per-case identities'},
        'scheduling_scenarios': {'reference_seconds': seconds, 'reference_is_one_clip_one_bank_only': True,
            'repeat_reference_for_20_cases_hours': seconds * 20 / 3600,
            'repeat_reference_with_25pct_scheduling_margin_hours': seconds * 20 * 1.25 / 3600,
            'configured_case_limits_sum_hours': 20 * 4800 / 3600,
            'including_all_outer_kill_graces_hours': 20 * 4815 / 3600,
            'uncompressed_exports_ceiling_bytes': 20 * 256 * 2**20,
            'calendar_completion_bound': False, 'runtime_or_memory_bound_inferred_for_other_clips': False,
            'instrumented_diagnostic_reference_only': True, 'A40_cost_pool_eligible': False,
            'study_cost_revised': False},
        'preserved': {'canonical_evaluation_unchanged': True, 'study_yaml_unchanged': True,
            'held_out_pixels_closed': True, 'research_step_calibration_still_required': True,
            'no_optimizer_or_trajectory': True, 'no_new_pixel_decode': True,
            'new_training_pixels_or_weights_opened_during_planning': False,
            'scientific_or_replay_tolerances_relaxed': False}}


def build_plan(root):
    """Verify real manifest/cache metadata and prior receipts, then compile."""
    from datasets.staged_files import verify_manifests
    from datasets.video import RECIPE
    root = Path(root).resolve()
    if sha(root / REFERENCE) != REFERENCE_SHA256:
        raise ValueError('Verified one-clip reference changed; preserve the original evidence.')
    prior = json.loads((root / REFERENCE).read_text())
    if (not prior['verified'] or prior['research_admitted'] or not prior['verification']['replay']['passed']
            or prior['archive']['sha256'] != '3e9d17e10c9cdd1fa814ad2ccdbceeef71921158c3b34634023a79e145352f2b'):
        raise ValueError('Use the independently accepted one-clip diagnostic as scheduling reference only.')
    inputs = {'configs/study.yaml': STUDY_SHA256, 'configs/budget_study_plan.yaml': PROPOSAL_SHA256,
              REFERENCE: REFERENCE_SHA256}
    sources = dict(prior['source_sha256'])
    for name in NEW_SOURCES:
        sources[name] = sha(root / name)
    for name, expected in {**inputs, **sources}.items():
        if sha(root / name) != expected:
            raise ValueError('Frozen source/input changed: ' + name)
    for name, expected in prior['provenance']['upstream_sha256'].items():
        if sha(root / 'external/vjepa2' / name) != expected:
            raise ValueError('Pinned upstream source changed: ' + name)
    manifests = root / 'data/manifests/ucf101-staged-20x5-v1'
    rows, metadata = verify_manifests(root, manifests)
    prepared = root / 'data/processed/staged-exploration-v1'
    preparation = json.loads((prepared / 'preparation.json').read_text())
    index_path = prepared / 'train/index.json'
    index = json.loads(index_path.read_text())
    if (preparation['kind'] != 'exploration_preparation' or preparation['manifest_sha256'] != metadata['manifest_sha256']
            or sha(index_path) != preparation['clip_indices_sha256']['train'] or index['split'] != 'train'
            or index['source_manifest_sha256'] != metadata['files_sha256']['train.csv']
            or len(index['clips']) != len(rows['train'])
            or any(any(entry.get(k) != v for k, v in row.items()) or entry['recipe'] != RECIPE
                   for row, entry in zip(rows['train'], index['clips']))):
        raise ValueError('Training preparation/index metadata changed.')
    for name in ('metadata.json', 'train.csv', 'train_monitor.csv'):
        inputs[(manifests / name).relative_to(root).as_posix()] = sha(manifests / name)
    for path in (prepared / 'preparation.json', index_path):
        inputs[path.relative_to(root).as_posix()] = sha(path)
    identity = {'input_files_sha256': inputs, 'source_sha256': sources,
                'upstream_sha256': prior['provenance']['upstream_sha256'],
                'upstream_commit': prior['provenance']['upstream_commit'],
                'checkpoint_sha256': prior['verification']['released_checkpoint_sha256'],
                'component_weights': prior['verification']['released_component_weight_digests'],
                'reference_archive_sha256': prior['archive']['sha256'],
                'reference_runtime_for_future_match': prior['runtime'],
                'reference_host': prior['provenance']['host'],
                'reference_GPU_identity': prior['provenance']['nvidia_smi_identity']}
    return compile_plan(rows['train'], rows['train_monitor'], index['clips'], identity=identity,
                        reference={'session_seconds': prior['timings']['session_including_worker_setup_seconds'],
                                   'seeds': prior['settings']['seeds']})


def validate_plan(root, plan):
    expected = build_plan(root)
    # Python equality conflates False/0 and 4/4.0. A frozen JSON identity must
    # reject those edits as well as changed seed values or limits.
    if plan_digest(plan) != plan_digest(expected):
        raise ValueError('Frozen panel plan differs from reproducible metadata/seed/limit specification.')
    return {'kind': KIND, 'plan_sha256': plan_digest(plan), 'cases': len(plan['cases']),
            'mask_draws': plan['seeds']['distinct_panel_mask_draws'], 'launch_authorized': False}


def freeze_plan(root, output):
    plan = build_plan(root)
    data = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    output = Path(output)
    with output.open('xb') as handle:
        handle.write(data)
    return {'file_sha256': hashlib.sha256(data).hexdigest(), 'plan_sha256': plan_digest(plan),
            'case_count': len(plan['cases']), 'launch_authorized': False}
