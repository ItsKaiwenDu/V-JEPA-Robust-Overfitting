#!/usr/bin/env python3
"""Local metadata-only decision after the clean EOT8 numerical/replay pass."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRIOR='runs/pretrained/diagnostic/predictor-missing-replay-handoff-v1-20261114/target-review/sealed-target-review.json'
NEW_REVIEW='report/checks/2026-11-14-missing-candidate-replay-v1-target-review.json'
OLD_REVIEW='report/checks/2026-11-14-forward-preserving-v2-target-review.json'
COST='report/checks/2026-10-29-rtx3090-partial-benchmark-review.json'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def descriptor(root,name):
    p=root/name;return dict(path=name,bytes=p.stat().st_size,sha256=digest(p))


def write_new(path,value):
    path=Path(path)
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def call_counts(steps,restarts,*,scoring_audit_banks=0):
    # Exact structure of evaluate_clip + pgd_linf for epsilon>0: evaluator attack
    # clean call, PGD clean baseline, S*R differentiations and R final iterates.
    return {'steps':steps,'restarts':restarts,'input_gradient_objectives':steps*restarts,
            'no_grad_attack_bank_forwards':2+restarts,
            'total_attack_bank_objective_forwards':2+restarts+steps*restarts,
            'independent_no_grad_scoring_forwards':2+2*scoring_audit_banks,
            'extra_scoring_banks':scoring_audit_banks,'source_structure_only_not_a_timing_estimate':True}


def prepare(root,plan_path,report_path):
    root=Path(root).resolve();pins=json.loads((root/PRIOR).read_bytes())
    if len(pins)!=3237:raise ValueError('Require the final successful replay preservation set.')
    for name,want in pins.items():
        if digest(root/name)!=want:raise ValueError('Consumed source/evidence changed: '+name)
    reviewed=json.loads((root/NEW_REVIEW).read_bytes());prior=json.loads((root/OLD_REVIEW).read_bytes())
    if (reviewed['candidate_fresh_process_replay_complete_and_exact'] is not True
            or reviewed['all_36_returns_exact_candidate0'] is not True
            or reviewed['original_comparison']['input']['failed_elements']!=0
            or reviewed['kernel_performance_optimizer_study_budget_or_full_training_qualified'] is not False):
        raise ValueError('Exact one-case qualification and remaining gate separation required.')
    candidate=json.loads((root/'configs/research_attack_candidate_v1.json').read_bytes())
    old_config=json.loads((root/'configs/predictor_missing_replay_handoff_v1.json').read_bytes())
    economics=json.loads((root/COST).read_bytes())
    if candidate['primary_candidate']!={'steps':20,'restarts':2,'attack_masks':8,'scoring_masks':4,'epsilon_255':4,'step_size_255':None}:
        raise ValueError('Preserve the proposed primary and unselected calibration step.')
    pgd=ast.parse((root/'attacks/pgd.py').read_text())
    function=next(n for n in pgd.body if isinstance(n,ast.FunctionDef) and n.name=='pgd_linf')
    code=ast.unparse(function)
    if any(token not in code for token in ['range(restarts)','range(steps)','torch.autograd.grad(losses.sum(), current)','consider(current, _losses(objective, current))']):
        raise ValueError('Revisit workload counts if the frozen PGD structure changes.')
    route_source=(root/'models/predictor_forward_preserving_target_v1.py').read_text()
    for token in ['if self._objective_used or self.prototype.calls:', 'One native EOT8 loss only; no retry.',
                  'One exact native loss/input derivative required.', 'len(banks) != 8']:
        if token not in route_source:raise ValueError('Revisit integration decision if the single-use route changes.')
    sources=[NEW_REVIEW,OLD_REVIEW,COST,'configs/research_attack_candidate_v1.json','configs/budget_study_plan.yaml',
             'configs/study.yaml','models/execution.py','models/training.py','attacks/pgd.py',
             'models/candidate_evaluation.py','models/candidate_contract.py','models/candidate_budget.py',
             'models/predictor_forward_preserving_target_v1.py','models/predictor_eot8_owned_v1.py',
             'models/private_collection_v1.py','tests/test_private_collection_v1.py']
    gate_map={
      'clean_EOT8_one_clip_numerical_and_fresh_replay':{'status':'passed_scoped','evidence':NEW_REVIEW,'research_admission':False},
      'private_collection_reconstruction':{'status':'new_local_receiver_verified_separately','consumed_clients_changed':False,'target_metadata_preserved':'files only; local directory policy explicit 0700'},
      'repeated_PGD_input_gradient_integration':{'status':'not_implemented','next_local_priority':True,'blocker':'TargetRoute.loss/gradient are single-use and native-owner-token bound; old PGD calls torch.autograd.grad directly.'},
      'iterated_same_input_numerical_native_ownership_and_replay':{'status':'not_qualified','clean_pass_does_not_substitute':True,'no_assumed_exact_original_attack_trajectory':True},
      'backend_kernel_and_end_to_end_performance':{'status':'not_measured_for_new_integrated_path','diagnostic_worker_wall_not_a_benchmark':True,'speedup':None},
      'research_hardware_optimizer_and_exact_recovery':{'status':'requires_chosen_runtime_and_explicit_new_attack_identity','parameter_update_gradient_route':'existing original path; custom VJP stays outside optimizer backward','university_8GiB_full_AdamW_training_qualified':False},
      'research_attack_calibration_and_stronger_control':{'status':'not_complete','ten_training_clips_three_steps_primary_units':30,'selected_step_size_255':None,'primary':candidate['primary_candidate'],'stronger_control':candidate['stronger_control']},
      'full_study_economics':{'status':'not_admitted','latest_existing_3090_primary_only_screen_passed':economics['primary_only_economic_screen']['sample_based_economic_screen_passed'],'new_candidate_complete_study_cost_USD':None},
      'full_training':{'status':'not_authorized','requires_all_preceding_relevant_gates_and_separate_adoption':True}}
    local_scope={'kind':'proposed_local_tiny_PGD_scope_not_a_target_grant','implemented':False,
      'device':'cpu','synthetic_tiny_only':True,'university_hosts_and_CUDA_refused_before_work':True,'input_shape':[1,3,4,16,16],
      'encoder_blocks':2,'predictor_blocks':2,'attack':{'steps':2,'restarts':2,'epsilon_255':4,'fixed_test_step_size_255':1,'step_is_not_research_calibration':True,'random_start':True,'draws':8,'families':2},
      'candidate_gradient_objectives_at_most':4,'same_input_original_gradient_checks_at_most':4,
      'conditional_fresh_candidate_rollout_gradient_objectives_at_most':4,'total_gradient_objectives_at_most':12,
      'fresh_candidate_rollout_only_after_initial_gates_pass':True,'completed_clean_cohort_repeats':0,
      'per_objective_native_root_and_distinct_math_tasks_required':True,'per_objective_frozen_tiny_caps_unchanged':True,
      'no_grad_attack_selection_and_independent_score_use_original_forward_outside_custom_scope':True,
      'parameter_gradients_absent_and_existing_optimizer_path_untouched':True,'training_PGD5_EOT1_and_routine_EOT1_paths_unchanged':True,'artifact_kind_must_declare_custom_attack_gradient_identity':True,
      'preserve_original_consider_before_backward_order_and_all_best_candidate_rules':True,
      'failure_stops_without_retry_or_scope_reuse':True,'no_held_out_pixels_paid_GPU_profile_optimizer_or_target_work':True}
    plan={'kind':'forward-preserving-attack-transition-local-plan-v1','date':'2026-11-14','proposal_only':True,
      'authorized':False,'new_target_execution_permitted':False,'research_policy_registered':False,'full_training_authorized':False,
      'decision':'Proceed to a separate local tiny EOT8 PGD input-gradient callback integration for primary/convergence evaluation. Retain existing parameter-update, training PGD5/EOT1 and routine/EOT1 paths; they are outside this candidate scope. Do not benchmark a clean-call-only adapter as if it were a research PGD implementation.',
      'sources':{n:descriptor(root,n) for n in sources},'completed_evidence_scope':{'training_clips':1,'clean_EOT8':True,'fresh_candidate_replay_exact':True,'parameter_update_attack_or_performance_qualification':False},
      'gates':gate_map,'source_control_flow_counts':{'local_integration':call_counts(2,2),'primary':call_counts(20,2),'stronger_control':call_counts(50,2,scoring_audit_banks=2)},
      'local_prototype_scope':local_scope,
      'integration_requirements':['Use a new opt-in PGD/evaluator artifact identity; leave existing PGD, execution policies, training and study YAML unchanged.',
       'An explicit per-iterate gradient context owns the actual loss/input token. Do not globally patch torch.autograd.grad or bypass native ownership.',
       'Create a fresh bounded route for each differentiation. Keep all no-grad clean/final-iterate/independent-scoring calls on original forward.',
       'Hold clean targets, masks, restart seeds, perturbation projection, all-candidate selection, per-example comparisons and consider-before-backward ordering fixed.',
       'Compare gradients at identical iterated inputs using unchanged bounds; exact clean replay does not imply identical PGD sign updates or final adversaries.',
       'Record per-objective native/checkpoint/inner-task evidence and restoration. Do not widen or reuse a consumed single-objective guard.',
       'Keep custom first-order input differentiation outside parameter accumulation, AdamW, checkpoint selection and independent scoring.',
       'Stop on any value/ownership/restoration/resource failure. No passing marker reconstructed from previous clean returns.'],
      'future_target_performance_prerequisites':['Local repeated-objective integration and conditional fresh rollout replay pass first.',
       'Prepare a separate bounded target iterated-input/ownership and backend-trace scope; completed clean cohorts are not rescheduled.',
       'Measure original and opt-in end-to-end units with identical inputs, scoring, runtime and bookkeeping; declare observer/transfer/serialization costs.',
       'Any lean observation variant needs separate explicit validation; do not disable guards or multiply instrumented single-EOT8 wall time into a primary-workload estimate.',
       'Measure all affected workload/recovery/checkpoint units on the selected research runtime before full cost admission. Historical screen prices and other-GPU timing are not current quotes or evidence for this route.'],
      'financial_state_unchanged':old_config['financial_state_unchanged'],
      'new_financial_hold_or_hold_release_USD':0,'new_SSH_upload_helper_authorization_attempt_GPU_or_scientific_work':False,
      'directory_fix_not_wired_into_consumed_clients':True,'next_local_step':'Implement only the declared tiny repeated-PGD gradient-context prototype and new local tests. No university request is ready at this checkpoint.'}
    walls=[{'role':w['mode']+str(w['replay']),'seconds':w['worker_wall_seconds'],'source_worker':w['worker'],'scope':'instrumented full-model verification wall, not a work-unit timing'} for w in prior['workers']]
    walls.append({'role':'candidate1','seconds':reviewed['worker_objective_wall_seconds'],'source_worker':reviewed['new_worker']['worker'],'scope':'instrumented full-model verification wall, not a work-unit timing'})
    report={'kind':'forward-preserving-attack-transition-metadata-review-v1','passed':True,'prior_sealed_files_verified':len(pins),'one_case_clean_replay_gate_closed':True,
      'PGD_integration_is_the_next_local_blocker':True,'planned_primary_input_differentiations_per_clip':40,'planned_primary_attack_bank_forward_calls_per_clip':44,
      'diagnostic_wall_samples_not_performance_units':walls,'speedup_estimated':False,'new_primary_unit_seconds':None,'new_full_study_cost_USD':None,
      'research_admitted':False,'study_training_execution_precision_or_attack_sources_changed':False,
      'no_model_tensor_deserialization_autograd_reference_or_remote_work':True,'next_local_scope':local_scope,'gates':gate_map,
      'financial_state_unchanged':plan['financial_state_unchanged'],'plan_output':str(plan_path)}
    os.umask(0o077);write_new(plan_path,plan);write_new(report_path,report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    r=prepare(ROOT,a.plan,a.report);print(json.dumps({'passed':r['passed'],'prior_pins':r['prior_sealed_files_verified'],'next':'local PGD gradient-context prototype','new_GPU_or_scientific_work':False},sort_keys=True))
