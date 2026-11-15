"""Local-only split-profile design and budget calculator; no execution entry.

Consumes frozen JSON/source bytes, never tensors, model code, a profiler or SSH.
Result inventories are proposed typed namespaces, not an active target policy.
"""
import hashlib
import json
from pathlib import PurePosixPath

MiB=2**20
GiB=2**30
BASE_SHA='c7bb7f3af5e6b9e65ac746177b657ae966e2bc50171a6e5b516e30b848252960'
KIND='pretrained-pgd-split-profile-local-design-v1'
NATIVE=('q','k','v','output','grad_output','dq','dk','dv')
RETURNS=('output','dq','dk','dv')
PROBES=('native','forward-preserving')


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def base_contract(data):
    if type(data) is not bytes or hashlib.sha256(data).hexdigest()!=BASE_SHA:
        raise ValueError('Frozen predecessor JSON bytes required; no edited baseline.')
    return json.loads(data)


def coverage():
    draws,families,encoder,predictor=8,2,24,12
    each=draws*families;owners=encoder+predictor
    return {'draws':draws,'families':families,'owners':owners,
            'primary_calls_per_owner':each,'checkpoint_calls_per_owner':each,
            'native_rows_per_objective':2*owners*each,
            'initial_math_VJPs_per_candidate':predictor*each,
            'predictor_forward_rows_per_candidate':2*predictor*each,
            'legacy_full_trace_counts_not_claimed_here':{
                'efficient_forward':2*owners*each,'original_efficient_backward':owners*each,
                'candidate_efficient_backward':encoder*each,'candidate_inner_math':predictor*each}}


def recipe(role):
    if role not in PROBES:raise ValueError('Exactly two explicit component probe roles.')
    hybrid=role=='forward-preserving'
    return {'role':role,'shape':[1,12,2048,32],'dtype':'float32','device':'cuda:0',
            'profiled_forward_evaluations':1,'profiled_VJP_evaluations':1,
            'unprofiled_iterations':0,'warmups':0,'replay_workers':0,'FP64_reference_builds':0,
            'efficient_forward_in_forward_phase':1,
            'efficient_backward_in_backward_phase':0 if hybrid else 1,
            'math_forward_in_backward_phase':1 if hybrid else 0,
            'math_forward_in_forward_phase':0,
            'custom_backward':'_AttentionBackward' if hybrid else None,
            'inner_math_root_and_three_leaf_hooks_required':hybrid,
            'required_positive_correlated_kernels':
                ['efficient-forward','math-GEMM','math-softmax-forward','math-softmax-backward'] if hybrid
                else ['efficient-forward','efficient-backward'],
            'correlation_rule':'recorded host/GPU links plus CPU ancestry; cross-thread roots need unique recorded forward-thread/sequence/producer binding',
            'kernel_name_or_duration_is_identity':False,'full_EOT8_backend_coverage_qualified':False,
            'component_VJP_accuracy_qualified':False,'performance_qualified':False}


def result_specs(base):
    """Exact proposed result filenames; no broad suffix-only admission."""
    c=coverage();spec={}
    def add(name,category,cap,mutable=False,partial=True):
        if name in spec:raise ValueError('Duplicate proposed result name.')
        spec[name]={'category':category,'max_bytes':cap,'mutable':mutable,'partial_allowed':partial}
    header=4096
    lengths=[pair[1]['shape'][1] for draw in base['reference']['full_mask_tensor_identities'] for pair in draw]
    if len(lengths)!=16 or any(type(n) is not int or not 0<n<2048 for n in lengths):
        raise ValueError('Exactly sixteen original mask-target lengths required.')
    tensor_sizes={'pixels':3*16*256*256*4,'input':3*16*256*256*4,'targets':2048*1024*4,
                  'masks':8*2*2048*8,'loss':4}
    for i,n in enumerate(lengths):
        for label in ('prediction','cotangent'):tensor_sizes[label+'-'+str(i)]=n*1024*4
    for i in range(8):
        prefix='results/numerical/objective-'+str(i)+'/'
        for name,size in tensor_sizes.items():add(prefix+name+'.npy','raw',size+header)
        for name,cap,mutable in [('numerical.json',256*1024,True),('numerical-complete.json',16*1024,False),
                ('numerical-failed.json',16*1024,False),('native-calls.jsonl',c['native_rows_per_objective']*2048,False),
                ('VJP-calls.jsonl',c['predictor_forward_rows_per_candidate']*8192 if i%2 else 0,False),
                ('scopes.json',64*1024,False)]:add(prefix+name,'metadata',cap,mutable)
        add(prefix+'post-gradient-progress-v2.jsonl','metadata',64*1024,partial=False)
        if i<2:
            for name,cap in [('provisional-native-calls-v2.jsonl',c['native_rows_per_objective']*2048),
                    ('provisional-VJP-calls-v2.jsonl',c['predictor_forward_rows_per_candidate']*8192 if i%2 else 0),
                    ('provisional-scopes-v2.json',64*1024),('provisional-capture-v2.json',16*1024)]:
                add(prefix+name,'metadata',cap,partial=False)
    for n in ('clean','adversarial','restart-noise-0','restart-noise-1'):
        add('results/numerical/'+n+'.npy','raw',3*16*256*256*4+header)
    for n in ('attack-loss','clean-attack-loss','clean-score','adversarial-score'):
        add('results/numerical/'+n+'.npy','raw',4+header)
    add('results/numerical/scoring-masks.npy','raw',4*2*2048*8+header)
    for n in NATIVE:add('results/boundary/'+n+'.pt','raw',4*MiB)
    for name,cap,mutable in [('boundary.json',128*1024,True),('boundary-complete.json',16*1024,False),
            ('boundary-failed.json',16*1024,False),('attention-calls.jsonl',c['native_rows_per_objective']*2048,False),
            ('attention-scopes.json',64*1024,False)]:add('results/boundary/'+name,'metadata',cap,mutable)
    for role in PROBES:
        prefix='results/components/'+role+'/'
        for n in RETURNS:add(prefix+n+'.npy','raw',4*MiB)
        for name,cap,mutable in [('worker.json',256*1024,True),('complete.json',16*1024,False),('failed.json',16*1024,False),
                ('resource.json',16*1024,False),('ownership.json',64*1024,False),('backend.json',128*1024,False),
                ('trace-manifest.json',16*1024,False)]:add(prefix+name,'metadata',cap,mutable)
        add(prefix+'progress.jsonl','metadata',64*1024,partial=False)
        add(prefix+'events.json','trace',4*MiB)
    for name,cap in [('numerical',2*MiB),('native',MiB//2),('forward-preserving',MiB//2),('setup',MiB//2),('proof',MiB//2)]:
        add('logs/'+name+'.log','metadata',cap,partial=False)
    for name,cap in [('plan.json',256*1024),('run-status.json',256*1024),('supervisor.json',128*1024),
            ('collection-status.json',128*1024),('final-receipt.json',128*1024),('storage-proof-result.json',16*1024)]:
        add('results/'+name,'metadata',cap,name in ('run-status.json','supervisor.json','collection-status.json'))
    return spec


def classify(name,spec):
    if type(name) is not str or not name or name.startswith('/') or '\\' in name:
        raise ValueError('Canonical relative result filename required.')
    path=PurePosixPath(name)
    if str(path)!=name or any(part in ('..','.') for part in path.parts):
        raise ValueError('Canonical relative result filename required.')
    partial=name.endswith('.partial');plain=name.removesuffix('.partial')
    if plain not in spec or partial and not spec[plain]['partial_allowed']:
        raise ValueError('Unknown proposed result path or disallowed partial: '+name)
    return spec[plain],plain,partial


def result_totals(entries,spec):
    """Pure metadata simulation, not an actual filesystem/ownership inventory."""
    totals=dict(raw=0,metadata=0,trace=0);raw_partial=0;meta_partial=0;extra_metadata=0
    for name,size in entries.items():
        item,plain,partial=classify(name,spec)
        if type(size) is not int or not 0<=size<=item['max_bytes']:raise ValueError('Typed extent exceeded.')
        totals[item['category']]+=size
        if partial:
            if item['category']=='raw':raw_partial+=1
            else:meta_partial+=1
            if plain in entries:
                if not item['mutable']:raise ValueError('Exclusive publication cannot replace a completed result.')
                extra_metadata+=size
    if raw_partial>1 or meta_partial>2 or extra_metadata>512*1024:
        raise ValueError('Only one serial raw partial and two bounded mutable metadata partials.')
    if totals['raw']>1792*MiB or totals['trace']>8*MiB or totals['metadata']+totals['trace']>64*MiB:
        raise ValueError('Combined raw/trace/shared-metadata ceiling exceeded.')
    return totals


def budget(spec,phases):
    if any(type(v) is not int or v<0 for v in phases.values()):raise ValueError('Integer nonnegative phase budgets required.')
    if set(phases)!={'setup','proof','numerical','native-probe','candidate-probe','finalization','overhead'}:
        raise ValueError('Every setup/work/finalization interval must be counted.')
    if sum(phases.values())>2400:raise ValueError('Proposed work exceeds unchanged forty-minute envelope.')
    final_totals=result_totals({name:item['max_bytes'] for name,item in spec.items()},spec)
    raw=final_totals['raw']+16*MiB
    metadata=final_totals['metadata']+final_totals['trace']+512*1024
    if raw>1792*MiB or metadata>64*MiB:raise ValueError('Completed results plus allowed partials do not fit.')
    capture_caps=(1792+128+128+64+32+16)*MiB
    proof_caps=(2560+128+128+64+32+16)*MiB # Raw capture forbidden during proof.
    if capture_caps+96*MiB>9*GiB//4 or proof_caps+32*MiB>3*GiB:
        raise ValueError('Conservative category plus allocation reserves exceed aggregate envelope.')
    return {'complete_raw_bytes':final_totals['raw'],'raw_plus_one_16MiB_partial_bytes':raw,'raw_cap_bytes':1792*MiB,
            'complete_metadata_including_trace_bytes':final_totals['metadata']+final_totals['trace'],
            'metadata_plus_mutable_partials_bytes':metadata,'shared_metadata_cap_bytes':64*MiB,
            'trace_bytes_included_in_metadata':final_totals['trace'],'trace_cap_bytes':8*MiB,
            'capture_category_caps_bytes':capture_caps,'capture_allocation_reserve_bytes':96*MiB,'capture_ceiling_bytes':9*GiB//4,
            'proof_category_caps_bytes':proof_caps,'proof_allocation_reserve_bytes':32*MiB,'setup_ceiling_bytes':3*GiB,
            'allocated_fit_requires_live_observation':True,'result_files':len(spec),
            'proposed_work_seconds':sum(phases.values()),'work_cap_seconds':2400,
            'collection_seconds':600,'connection_including_login_seconds':3000}


def build(base_bytes,source_pins):
    base=base_contract(base_bytes);spec=result_specs(base)
    phases={'setup':180,'proof':60,'numerical':1680,'native-probe':120,'candidate-probe':120,'finalization':120,'overhead':120}
    resources=budget(spec,phases)
    if len(spec)+256+3>1024:raise ValueError('Results plus reserved input/code/control/in-flight files exceed old file cap.')
    if not source_pins or any(type(k) is not str or type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for k,v in source_pins.items()):
        raise ValueError('Pinned predecessor source descriptors required.')
    return {'kind':KIND,'date':'2026-11-15','authorized':False,'new_target_execution_permitted':False,'target_runner_available':False,
        'local_only':True,'base_contract_sha256':BASE_SHA,'source_sha256':source_pins,
        'coverage':coverage(),'numerical_bounds':base['numerical_bounds'],'financial_state_unchanged':base['financial_state_unchanged'],
        'numerical':{'full_model_gradient_objectives':8,'same_input_original':4,'candidate':4,'steps':2,'restarts':2,
            'EOT_draws':8,'families':2,'profiled_full_model_objectives':0,'no_grad_forwards':6,
            'old_targets_masks_seed_selection_projection_and_37_returns_unchanged':True,'fresh_rollout_replay':False},
        'boundary':{'source_objective_index':0,'new_boundary_captures':1,'owner':'predictor.0','draw':0,'family':0,
            'native_forward_row_index':24,'phase':'EOT8-before','native_arrays':list(NATIVE),
            'source':'new original objective zero only; never historical partial/clean/component arrays',
            'original_private_alias_hooks_graph_layout_flags_and_restoration_required':True},
        'probes':[recipe(role) for role in PROBES],
        'gradient_accounting':{'full_model_objectives':8,'extra_component_VJP_evaluations':2,
            'maximum_total_evaluation_units':10,'initial_full_model_inner_math_VJPs':4*192,
            'extra_component_inner_math_VJPs':1,'CPU_gradient_or_FP64_reference_objectives':0,
            'warmups_or_untimed_profile_controls':0,'completed_cohort_repeats':0},
        'ordering':['storage proof','eight numerical/ownership/restoration gates','native component','forward-preserving component','combined review'],
        'failure_policy':'Stop subsequent work; preserve numerical/provisional/trace evidence; no retry or old claim reset.',
        'profile':{'options':{'record_shapes':False,'with_stack':False,'profile_memory':False},
            'raw_API':'torch.profiler.profile.profiler.kineto_results','processed_profile_events_used':False,
            'raw_event_cap':20000,'raw_file_bytes':4*MiB,'combined_trace_bytes':8*MiB,
            'raw_trace_saved_before_semantic_validation':True,'event_limit_is_not_pre_materialization_memory_bound':True,
            'component_RSS_ceiling_bytes':4*GiB,'component_wall_seconds':120,'RSS_supervisor_required':True,
            'RSS_monitor_is_not_address_space_or_instantaneous_allocation_guarantee':True},
        'phase_seconds':phases,'resource':resources,'allocator_bytes':7*GiB,
        'inventory':{'canonical_result_sha256':hashlib.sha256(encoded(spec)).hexdigest(),'result_files':len(spec),
            'unknown_result_paths':'refuse','metadata_trace_is_shared_not_additive':True,
            'maximum_simultaneous_raw_partials':1,'maximum_metadata_partials':2,'maximum_extra_mutable_metadata_bytes':512*1024,
            'reserved_other_files':256,'maximum_all_files':1024},
        'claims':{'new_diagnostic_identity_required':True,'legacy_worker_or_backend_success_compatible':False,
            'all_numerical_native_root_checkpoint_restoration_gates_required':True,
            'representative_backend_only':True,'full_EOT8_kernel_counts_qualified':False,'component_VJP_accuracy_qualified':False,
            'performance_or_cost_qualified':False,'fresh_GPU_rollout_replay_qualified':False,'training_or_research_admitted':False},
        'outstanding_before_target':['New hybrid native-forward/math-backward raw-correlation reader, including owned cross-thread and inner-task association.',
            'New component route must retain exact candidate arithmetic and real root/leaf ownership; old all-math component entry is incompatible.',
            'New numerical worker/receipts and v2 progress lifecycle must separate provisional capture from detailed profiling, including boundary-observer cleanup.',
            'Implement/test actual typed inventory, serial partials, source/layout lineage, parent wall/RSS monitor and bounded collection.',
            'Verify complete source package and combined scope independently before requesting one new exact university authorization.'],
        'paid_compute':False,'cleanup':False,'held_out_pixels':False,'private_key_upload':False,
        'automatic_retries':0,'old_claims_or_clocks_reset':False,'actual_scientific_execution_in_this_step':False}


def validate(plan,base_bytes,source_pins):
    if encoded(plan)!=encoded(build(base_bytes,source_pins)):
        raise ValueError('Exact reviewed local design required; this validator cannot arm or authorize it.')
    return plan
