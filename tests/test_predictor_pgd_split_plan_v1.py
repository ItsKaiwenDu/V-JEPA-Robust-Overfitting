"""New local design/metadata arithmetic checks; no scientific execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
from models import predictor_pgd_split_plan_v1 as p

ROOT=Path(__file__).resolve().parents[1]
BASE=(ROOT/'configs/predictor_pgd_target_v1.json').read_bytes()
PIN_NAMES=('models/predictor_pgd_target_worker_v1.py','models/predictor_pgd_target_records_v1.py',
 'models/predictor_sdpa_observer_v1.py','models/predictor_forward_preserving_target_v1.py',
 'models/predictor_forward_preserving_cpu_v1.py','models/post_gradient_progress_v1.py',
 'models/predictor_pgd_progress_adapter_v1.py','models/predictor_sdpa_launch_evidence_v2.py',
 'models/predictor_sdpa_raw_capture_v1.py','models/predictor_sdpa_component_events_v2.py',
 'models/predictor_sdpa_launch_evidence_v1.py')
PINS={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in PIN_NAMES}


def design():return p.build(BASE,PINS)
def spec():return p.result_specs(p.base_contract(BASE))


def test_concrete_scope_keeps_eight_numerical_objectives_and_declares_two_extra_components():
    plan=design();n=plan['numerical'];g=plan['gradient_accounting']
    assert (n['same_input_original'],n['candidate'],n['full_model_gradient_objectives'])==(4,4,8)
    assert (n['EOT_draws'],n['families'],n['steps'],n['restarts'])==(8,2,2,2)
    assert n['profiled_full_model_objectives']==0 and n['no_grad_forwards']==6
    assert g['extra_component_VJP_evaluations']==2 and g['maximum_total_evaluation_units']==10
    assert g['initial_full_model_inner_math_VJPs']==768 and g['extra_component_inner_math_VJPs']==1
    assert g['CPU_gradient_or_FP64_reference_objectives']==g['warmups_or_untimed_profile_controls']==g['completed_cohort_repeats']==0
    assert plan['ordering'].index('eight numerical/ownership/restoration gates')<plan['ordering'].index('native component')<plan['ordering'].index('forward-preserving component')


def test_coverage_is_all_original_owners_draws_families_and_checkpoint_recomputations():
    c=design()['coverage']
    assert (c['owners'],c['primary_calls_per_owner'],c['checkpoint_calls_per_owner'])==(36,16,16)
    assert c['native_rows_per_objective']==1152
    assert (c['initial_math_VJPs_per_candidate'],c['predictor_forward_rows_per_candidate'])==(192,384)
    assert c['legacy_full_trace_counts_not_claimed_here']==dict(efficient_forward=1152,original_efficient_backward=576,candidate_efficient_backward=384,candidate_inner_math=192)


def test_selected_boundary_is_one_new_counted_original_native_call_at_full_geometry():
    plan=design();b=plan['boundary']
    assert (b['source_objective_index'],b['new_boundary_captures'],b['owner'],b['draw'],b['family'],b['native_forward_row_index'])==(0,1,'predictor.0',0,0,24)
    assert b['native_arrays']==['q','k','v','output','grad_output','dq','dk','dv']
    assert 'never historical' in b['source']
    assert all(probe['shape']==[1,12,2048,32] and probe['dtype']=='float32' for probe in plan['probes'])
    # Read the real frozen observer's selection expression, without importing it.
    source=ast.parse((ROOT/'models/predictor_sdpa_observer_v1.py').read_bytes())
    assignments=[n for n in ast.walk(source) if isinstance(n,ast.Assign)]
    index=next(n for n in assignments if any(isinstance(t,ast.Attribute) and t.attr=='row_index' for t in n.targets))
    assert ast.dump(index.value,include_attributes=False)==ast.dump(ast.parse('counts[0]',mode='eval').body,include_attributes=False)


def test_component_recipe_distinguishes_native_forward_from_public_math_backward():
    native,hybrid=design()['probes']
    assert native['role']=='native' and hybrid['role']=='forward-preserving'
    for r in (native,hybrid):
        assert r['efficient_forward_in_forward_phase']==1 and r['math_forward_in_forward_phase']==0
        assert r['profiled_forward_evaluations']==r['profiled_VJP_evaluations']==1
        assert not r['kernel_name_or_duration_is_identity'] and not r['component_VJP_accuracy_qualified']
    assert native['efficient_backward_in_backward_phase']==1 and native['math_forward_in_backward_phase']==0
    assert hybrid['efficient_backward_in_backward_phase']==0 and hybrid['math_forward_in_backward_phase']==1
    assert hybrid['custom_backward']=='_AttentionBackward' and hybrid['inner_math_root_and_three_leaf_hooks_required']
    assert {'math-GEMM','math-softmax-forward','math-softmax-backward'}<=set(hybrid['required_positive_correlated_kernels'])


def test_joint_raw_bound_includes_original_returns_scalars_boundary_components_and_one_partial():
    r=design()['resource']
    # Prior report omitted 8 objective loss scalars and 4 rollout scalars from
    # its formula; the separate new bound counts those 48 bytes explicitly.
    previous=json.loads((ROOT/'report/checks/2026-11-15-pretrained-pgd-target-v1-preparation-review.json').read_bytes())
    old=previous['payload']['storage_bound']['eight_objectives_plus_rollout_and_partial_bytes']
    assert old==1785532416
    assert r['raw_plus_one_16MiB_partial_bytes']==old+48+16*4*p.MiB==1852641328
    assert r['raw_cap_bytes']-r['raw_plus_one_16MiB_partial_bytes']==26406864


def test_joint_metadata_bound_includes_progress_snapshots_duplicate_boundary_proof_logs_and_both_traces():
    r=design()['resource']
    assert r['trace_bytes_included_in_metadata']==8*p.MiB
    assert r['complete_metadata_including_trace_bytes']==60178432
    assert r['metadata_plus_mutable_partials_bytes']==60702720
    assert r['shared_metadata_cap_bytes']-r['metadata_plus_mutable_partials_bytes']==6406144
    assert r['metadata_plus_mutable_partials_bytes']<64*p.MiB


def test_all_category_caps_and_allocation_reserves_fit_capture_and_setup_envelopes():
    r=design()['resource']
    assert r['capture_category_caps_bytes']==2264924160
    assert r['capture_category_caps_bytes']+r['capture_allocation_reserve_bytes']<=r['capture_ceiling_bytes']==9*p.GiB//4
    assert r['proof_category_caps_bytes']==2928*p.MiB
    assert r['proof_category_caps_bytes']+r['proof_allocation_reserve_bytes']<=r['setup_ceiling_bytes']==3*p.GiB
    assert r['allocated_fit_requires_live_observation'] is True


def test_forty_minute_budget_counts_setup_proof_components_finalization_and_overhead():
    plan=design();phases=plan['phase_seconds']
    assert phases['numerical']==1680 and phases['native-probe']==phases['candidate-probe']==120
    assert sum(phases.values())==plan['resource']['work_cap_seconds']==2400
    assert plan['resource']['collection_seconds']==600 and plan['resource']['connection_including_login_seconds']==3000
    assert plan['profile']['component_RSS_ceiling_bytes']==4*p.GiB
    assert plan['profile']['RSS_monitor_is_not_address_space_or_instantaneous_allocation_guarantee'] is True


@pytest.mark.parametrize('change',['old_numerical_budget','missing_proof','negative','float','bool','extra_phase'])
def test_time_accounting_cannot_reuse_old_32min_child_or_omit_phase(change):
    phases=copy.deepcopy(design()['phase_seconds'])
    if change=='old_numerical_budget':phases['numerical']=1920
    elif change=='missing_proof':phases.pop('proof')
    elif change=='negative':phases['setup']=-1
    elif change=='float':phases['proof']=60.0
    elif change=='bool':phases['proof']=True
    else:phases['unaccounted']=1
    with pytest.raises(ValueError):p.budget(spec(),phases)


def test_finite_inventory_counts_all_arrays_and_reserves_other_payload_files():
    s=spec();plan=design()
    assert len(s)==plan['inventory']['result_files']==419
    assert sum(v['category']=='raw' for v in s.values())==321
    assert len(s)+plan['inventory']['reserved_other_files']+3<=plan['inventory']['maximum_all_files']==1024
    assert hashlib.sha256(p.encoded(s)).hexdigest()==plan['inventory']['canonical_result_sha256']
    assert not any(name.endswith('/success.json') for name in s)


@pytest.mark.parametrize('name',['/results/numerical/clean.npy','results/../escape','results//numerical/clean.npy',
 'results/numerical/clean.npy.partial.partial','results/numerical/objective-8/input.npy',
 'results/numerical/objective-0/backend.json','results/numerical/objective-0/post-gradient-progress-v2.jsonl.partial',
 'results/components/math/events.json','results/components/native/events.json/../worker.json',
 'results\\numerical\\clean.npy'])
def test_unknown_or_ambiguous_result_paths_refuse(name):
    with pytest.raises(ValueError):p.classify(name,spec())


@pytest.mark.parametrize('size',[True,-1,1.0,4194305])
def test_typed_extents_refuse_invalid_sizes(size):
    with pytest.raises(ValueError):p.result_totals({'results/components/native/events.json':size},spec())


def test_completed_serial_arrays_and_mutable_metadata_partials_fit_together():
    s=spec();entries={name:row['max_bytes'] for name,row in s.items()}
    entries['results/numerical/objective-7/input.npy.partial']=s['results/numerical/objective-7/input.npy']['max_bytes']
    entries['results/numerical/objective-7/numerical.json.partial']=256*1024
    entries['results/run-status.json.partial']=256*1024
    # Raw final files are exclusive. An in-flight fresh raw partial has no final.
    entries.pop('results/numerical/objective-7/input.npy')
    totals=p.result_totals(entries,s)
    assert totals['metadata']+totals['trace']<=64*p.MiB and totals['raw']<=1792*p.MiB


@pytest.mark.parametrize('case',['raw_two','metadata_three','replace_static','replace_trace'])
def test_multiple_writers_or_static_result_replacement_refuse(case):
    s=spec()
    if case=='raw_two':names=['results/numerical/clean.npy.partial','results/numerical/adversarial.npy.partial']
    elif case=='metadata_three':names=['results/run-status.json.partial','results/supervisor.json.partial','results/collection-status.json.partial']
    elif case=='replace_static':names=['results/numerical/objective-0/native-calls.jsonl','results/numerical/objective-0/native-calls.jsonl.partial']
    else:names=['results/components/native/events.json','results/components/native/events.json.partial']
    with pytest.raises(ValueError):p.result_totals({n:1 for n in names},s)


def test_combined_metadata_budget_is_not_separate_64MiB_plus_8MiB_trace():
    s=spec();name='results/plan.json';s[name]=dict(s[name],max_bytes=8*p.MiB)
    with pytest.raises(ValueError,match='metadata'):p.result_totals({n:v['max_bytes'] for n,v in s.items()},s)


def test_original_receipt_limits_stay_bounded_and_original_empty_VJP_file_stays_empty():
    s=spec()
    assert s['results/numerical/objective-0/VJP-calls.jsonl']['max_bytes']==0
    assert s['results/numerical/objective-1/VJP-calls.jsonl']['max_bytes']==384*8192
    assert s['results/numerical/objective-0/native-calls.jsonl']['max_bytes']==1152*2048<8*p.MiB
    with pytest.raises(ValueError):p.result_totals({'results/numerical/objective-0/VJP-calls.jsonl':1},s)


@pytest.mark.parametrize('change',['authorize','full_coverage','all_math_forward','old_backend_counts','hidden_extra_objective','old_metadata_cap','financial_release'])
def test_local_plan_rejects_arming_scope_promotion_or_resource_and_financial_drift(change):
    plan=copy.deepcopy(design())
    if change=='authorize':plan['authorized']=True
    elif change=='full_coverage':plan['claims']['full_EOT8_kernel_counts_qualified']=True
    elif change=='all_math_forward':plan['probes'][1]['math_forward_in_forward_phase']=1
    elif change=='old_backend_counts':plan['probes'][0]['efficient_forward_in_forward_phase']=1152
    elif change=='hidden_extra_objective':plan['gradient_accounting']['extra_component_VJP_evaluations']=0
    elif change=='old_metadata_cap':plan['resource']['shared_metadata_cap_bytes']=72*p.MiB
    else:plan['financial_state_unchanged']['conservative_encumbrance_USD']=0
    with pytest.raises(ValueError):p.validate(plan,BASE,PINS)


def test_frozen_base_and_numerical_bounds_finances_are_preserved_exactly():
    plan=design();base=p.base_contract(BASE)
    assert plan['numerical_bounds']==base['numerical_bounds']=={'loss':{'rtol':1e-5,'atol':2e-6},'input':{'rtol':.002,'atol':2e-7}}
    assert plan['financial_state_unchanged']==base['financial_state_unchanged']
    with pytest.raises(ValueError):p.build(BASE+b'\n',PINS)
    assert p.validate(plan,BASE,PINS) is plan


def test_design_explicitly_blocks_target_until_hybrid_reader_route_lifecycle_inventory_and_scope_are_tested():
    plan=design()
    assert plan['local_only'] and not plan['authorized'] and not plan['target_runner_available']
    assert not plan['new_target_execution_permitted'] and not plan['actual_scientific_execution_in_this_step']
    assert len(plan['outstanding_before_target'])==5
    assert 'cross-thread' in plan['outstanding_before_target'][0]
    assert 'old all-math component entry is incompatible' in plan['outstanding_before_target'][1]
    assert plan['claims']['representative_backend_only'] and not plan['claims']['legacy_worker_or_backend_success_compatible']
    assert plan['claims']['all_numerical_native_root_checkpoint_restoration_gates_required']
    assert not any(plan['claims'][k] for k in ('full_EOT8_kernel_counts_qualified','component_VJP_accuracy_qualified','performance_or_cost_qualified','fresh_GPU_rollout_replay_qualified','training_or_research_admitted'))


def test_source_is_standard_library_planning_only_with_no_execution_entry_or_scientific_imports():
    source=ast.parse((ROOT/'models/predictor_pgd_split_plan_v1.py').read_bytes())
    imports={n.module for n in ast.walk(source) if isinstance(n,ast.ImportFrom)}|{a.name for n in ast.walk(source) if isinstance(n,ast.Import) for a in n.names}
    assert imports<=set(sys.stdlib_module_names)
    assert not any(isinstance(n,ast.FunctionDef) and n.name in {'run','main','worker'} for n in source.body)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def test_proof_budget_includes_input_and_cache_reservations_omitted_from_earlier_phase_formula():
    plan=design();base=p.base_contract(BASE);r=base['resource']
    old=json.loads((ROOT/'report/checks/2026-11-15-pretrained-pgd-target-v1-preparation-review.json').read_bytes())['payload']['storage_bound']['proof_phase_category_bound_bytes']
    assert plan['resource']['proof_category_caps_bytes']==old+r['input_bytes']+r['cache_bytes']
    assert plan['resource']['proof_category_caps_bytes']==sum(r[k] for k in ('proof_bytes','input_bytes','code_bytes','metadata_bytes','cache_bytes','emergency_bytes'))
    assert plan['resource']['setup_ceiling_bytes']-plan['resource']['proof_category_caps_bytes']==144*p.MiB
