"""Offline EOT8 research proposal and conditional affordability arithmetic; no execution."""
from copy import deepcopy
import hashlib
import json
import math

KIND = 'research-primary-eot8-candidate-v1'
STEP_CANDIDATES = (0.5, 1.0, 2.0)
VARIABLE_COMPONENTS = ('primary_train_validation_clips', 'primary_test_clips', 'calibration_three_candidate_clip_bundles')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seed(namespace, video_id):
    body = json.dumps(['evaluation-v1', 20261028, namespace, video_id, 0, 0], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return int.from_bytes(hashlib.sha256(body).digest()[:8], 'big') % 2**63


def draw(namespace, video_id, index):
    body = json.dumps(['evaluation-v1', 20261028, namespace, video_id, 0, index], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return int.from_bytes(hashlib.sha256(body).digest()[:8], 'big') % 2**63


def calibration_spec(ordered_clips, historical_cases):
    if len(ordered_clips) != 10 or len({r['video_id'] for r in ordered_clips}) != 10:
        raise ValueError('Use the original first ten training-monitor rows.')
    rows = []
    for row in ordered_clips:
        vid = row['video_id']
        rows.append(dict(video_id=vid, source_group_id=row['source_group_id'],
            start_seed=seed('research_candidate_eot8_v1_start', vid),
            attack_seeds=[draw('research_candidate_eot8_v1_attack', vid, i) for i in range(8)],
            scoring_seeds={bank: [draw('research_candidate_eot8_v1_' + bank, vid, i) for i in range(4)]
                           for bank in ('main', 'audit_0', 'audit_1')}))
    new = [s % (2**31 - 1) for row in rows for s in row['attack_seeds'] + sum(row['scoring_seeds'].values(), [])]
    old = {s % (2**31 - 1) for row in historical_cases for s in row['attack_seeds'] + sum(row['scoring_seeds'].values(), [])}
    if len(new) != 200 or len(set(new)) != 200 or set(new) & old:
        raise ValueError('Candidate masks collide internally or with the completed diagnostic panel.')
    return dict(status='specified_not_executed', checkpoint='released-pretrained-only', panel='train_monitor',
        rows=rows, candidate_step_sizes_255=list(STEP_CANDIDATES), evaluations_required=30,
        attack=dict(steps=20, restarts=2, attack_masks=8, scoring_masks=4, epsilon_255=4),
        selected_step_size_255=None, selection_bank='main', audit_banks_used_for_selection=False,
        selection_rule='highest_mean_independent_loss_then_smaller_step_on_exact_tie',
        candidate_weights_runtime_sources_rows_and_seeds_identical=True,
        both_restart_starts_shared_across_step_candidates=True,
        masks_fixed_across_steps_and_restarts=True, candidate_main_audit_masks_globally_distinct=200,
        mask_modulus=2**31 - 1, no_overlap_with_completed_panel_mask_draws=True,
        convergence_uses_identical_starts_and_eight_draw_bank_for_step_only_control=True,
        fresh_adversaries_required_at_every_checkpoint=True, exact_chosen_research_runtime_required=True,
        another_GPU_diagnostic_calibration_portable_to_research=False,
        original_four_draw_calibration_reusable=False, completed_panel_step_two_is_not_calibration=True,
        validation_or_test_used=False, training_PGD5_step_size_255=1,
        full_research_panel_seed_manifest_still_required=True)


def proposal(ordered_clips, historical_cases):
    return dict(kind=KIND, schema_version=1, status='proposed_not_adopted_not_runnable',
        rationale='The complete fixed panel shows positive independent-scoring gains from four to eight draws on all ten clips in all three scoring banks. Preserve the stronger fifty-step control; twenty steps is not declared converged.',
        primary_candidate=dict(steps=20, restarts=2, attack_masks=8, scoring_masks=4, epsilon_255=4, step_size_255=None),
        stronger_control=dict(steps=50, restarts=2, attack_masks=8, scoring_masks=4, scoring_audits=2,
            step_size='same_training_calibrated_step_as_primary', selection_uses_attack_bank_only=True),
        not_selected=dict(legacy_primary_20x4='Retained for historical comparability; not silently relabeled as the new candidate.',
            primary_50x8='Keep as stronger control; not a required primary replacement chosen by largest observed loss.'),
        preserved_study=dict(classes=20, training_clips=100, conditions=['clean-finetune', 'adversarial-finetune'],
            exploration_seeds=[0], confirmation_seeds=[11, 22, 33], runs=8, epochs=30, updates_per_run=390,
            effective_batch_size=8, micro_batch_size=1, fixed_clean_target=True, native_masked_L1=True,
            frames=16, crop_size=256, training_attack=dict(steps=5, restarts=1, attack_masks=1, epsilon_255=4, step_size_255=1),
            primary_epochs=[1, 15, 16, 23, 24, 30], initialization_required=True,
            checkpoint_selection='minimum_complete_primary_validation_anchor_including_initialization_earliest_tie',
            original_group_separated_splits_preserved=True, confirmation_and_test_pixels_closed=True),
        calibration=calibration_spec(ordered_clips, historical_cases),
        implementation=dict(candidate_execution_policy='shared-fp64-native', research_precision_adopted=False,
            current_EvaluationSettings_supports_candidate=False, current_canonical_primary_attack_masks=4,
            new_versioned_evaluation_calibration_and_artifact_contract_required=True,
            canonical_study_or_budget_YAML_changed=False, old_artifacts_and_numerical_rejections_preserved=True),
        budget=dict(total_cap_usd=200, compute_cap_usd=160, pilot_reserve_usd=10, noncompute_reserve_usd=40,
            runtime_headroom_factor=1.25, historical_untimed_allowance_hours_not_a_bound=8,
            primary_evaluation_count=2580, three_candidate_calibration_bundle_count=10,
            primary_eot8_native_seconds=None, three_candidate_eot8_bundle_seconds=None,
            current_live_quote_usd_per_hour=None, complete_candidate_cost_usd=None,
            complete_cost_estimate=False, budget_approved=False),
        next_compute_purpose='Qualify sufficient-memory research hardware and measure the proposed EOT8 workload before full ten-clip research calibration or any trajectory.',
        benchmark_proposal=dict(implemented=False, launch_authorized=False, proposed_maximum_billed_hours=2,
            inclusive_session_budget_usd=None, deadline_must_include_deployment_setup_and_idle=True,
            proposed_required_phases=['unchanged numerical preflight', 'clean-and-adversarial-two-update-fresh-process-exact-recovery',
                'three-pretrained-representative-EOT8-primary-units', 'one-clean-and-one-adversarial-trained-anchor-EOT8-primary-unit',
                'one-complete-three-candidate-training-clip-bundle', 'retained-full-stronger-controls-or-new-matching-timing-evidence',
                'checkpoint-state-save-load-and-space-boundaries'],
            small_bundle_cannot_replace_ten_clip_calibration=True,
            earlier_eighty_minute_case_windows_or_recovery_grants_reused=False),
        launch_authorized=False, new_GPU_job_started=False, paid_compute_authorized=False,
        automatic_attack_choice=False, attack_convergence_pass=False, research_admitted=False)


def validate_proposal(value, expected):
    if canonical(value) != canonical(expected):
        raise ValueError('The explicit candidate, preserved study, calibration or non-admission scope changed.')


def conditional_budget(review):
    """Necessary affordability conditions under explicitly fixed historical nonprimary units.

    p is a future measured full EOT8 primary clip; c is its complete three-step
    calibration bundle per clip. Both remain unknown here. Limits setting c=0
    are optimistic necessary ceilings, not a zero-cost calibration estimate.
    """
    econ = review['economics']
    if (econ['retained_design'] != dict(classes=20, training_clips=100, runs=8, updates_per_run=390)
            or (econ['project_cap_usd'], econ['planned_compute_cap_usd'], econ['pilot_reserve_usd'], econ['noncompute_reserve_usd'],
                econ['runtime_headroom_factor'], econ['allowance_hours']) != (200, 160, 10, 40, 1.25, 8)):
        raise ValueError('Preserve the existing design, headroom, allowance and both budget caps.')
    results = {}
    for label in ('median', 'maximum'):
        original = econ['scenarios'][label]
        rows = original['component_contributions']
        counts = {k: rows[k]['count'] for k in VARIABLE_COMPONENTS}
        if counts != dict(primary_train_validation_clips=1980, primary_test_clips=600, calibration_three_candidate_clip_bundles=10):
            raise ValueError('The full primary/test/calibration workload changed.')
        if any(type(r['count']) is not int or r['count'] < 1 or not isinstance(r['unit_seconds'], (int, float))
               or isinstance(r['unit_seconds'], bool) or not math.isfinite(r['unit_seconds']) or r['unit_seconds'] <= 0 for r in rows.values()):
            raise ValueError('Historical screen timings must remain positive and finite.')
        rebuilt = math.fsum(r['count'] * r['unit_seconds'] / 3600 for r in rows.values())
        if not math.isclose(rebuilt, original['timed_unit_hours'], rel_tol=1e-13):
            raise ValueError('Historical cost components do not reproduce their recorded total.')
        fixed_hours = math.fsum(r['count'] * r['unit_seconds'] / 3600 for k, r in rows.items() if k not in VARIABLE_COMPONENTS)
        historic_total = (rebuilt + 8) * 1.25 * 0.61 + 50
        if not math.isclose(historic_total, original['total_usd'], rel_tol=1e-13):
            raise ValueError('The historical A40 screen no longer reproduces.')
        scenarios = []
        for rate in (0.29, 0.40, 0.50, 0.61):
            hours_limit = min((200 - 40 - 10) / (1.25 * rate), (160 - 10) / (1.25 * rate)) - 8
            variable_hours = hours_limit - fixed_hours
            scenarios.append(dict(hypothetical_all_in_hourly_usd=rate,
                all_other_units_assumed_equal_to_historical_A40_only=True,
                variable_primary_plus_calibration_seconds_room=max(0.0, variable_hours * 3600),
                no_primary_time_can_fit_even_with_zero_calibration=variable_hours <= 0,
                optimistic_primary_clip_seconds_ceiling_if_calibration_free=max(0.0, variable_hours * 3600 / 2580),
                zero_calibration_is_only_a_necessary_condition_relaxation=True,
                current_quote=False, another_GPU_speed_assumed_or_predicted=False, budget_approved=False))
        results[label] = dict(historical_four_draw_total_usd=historic_total,
            historical_other_component_hours=fixed_hours,
            candidate_timed_hours_expression=f'{fixed_hours:.12f} + (2580*p + 10*c)/3600',
            candidate_screen_total_expression='50 + 1.25*rate*(8 + historical_other_hours + (2580*p+10*c)/3600)',
            candidate_total_usd=None, missing_required_units=['full_eot8_primary_clip_seconds', 'complete_eot8_three_candidate_bundle_seconds'],
            conditional_price_scenarios=scenarios, complete_cost_estimate=False, budget_approved=False)
    return dict(kind=KIND + '-conditional-affordability', scenarios=results,
        units_not_transferred_from_university_RTX3070=True, four_draw_or_PGD50_timings_not_relabeled_as_EOT8_primary=True,
        no_mask_ratio_or_unmeasured_speedup_used=True, current_quote_checked=False,
        unknown_cost_categories=list(('full-data-loading-integrity-cache', 'full-panel-hashing-transfers',
            'full-features-probes', 'checkpoint-retention-export-disk-peaks', 'billed-setup-idle-failure-repair')),
        historical_allowance_not_a_bound=True, prior_pilot_charges_counted_inside_reserve_once=True,
        complete_candidate_cost_usd=None, complete_cost_estimate=False, budget_approved=False, research_admitted=False)
