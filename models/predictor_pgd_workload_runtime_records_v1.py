"""Whole-unit metadata joins. Saved dictionaries do not authenticate computation."""
import math
from models.predictor_pgd_workload_records_v1 import _seeds

ROLES = ('original-unit', 'candidate-unit', 'candidate-replay')
FALSE = {'scientific_qualified': False, 'training_admitted': False,
         'performance_qualified': False, 'actual_CUDA_execution_admitted': False}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(v):
    return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)


def finite(v):
    return type(v) in (int,float) and math.isfinite(v)


def validate_work(row, plan):
    role=plan['role'];require(role in ROLES,'Unknown complete-unit role.')
    require(row.get('kind')=='whole-pgd-unit-metadata-v1' and row.get('role')==role
            and row.get('control_sha256')==plan['control_sha256']
            and row.get('source_manifest_sha256')==plan['source_manifest_sha256']
            and row.get('plan_sha256')==plan['plan_sha256']
            and type(row.get('pid')) is int and row['pid']>0
            and all(row.get(k) is False for k in FALSE),'New source/control/role/plan/process identity required.')
    baseline=row['baseline'];require(set(baseline)=={'weights_sha256','clean_sha256','video_id','checkpoint_sha256'}
        and all(sha(baseline[k]) for k in baseline if k!='video_id')
        and type(baseline['video_id']) is str and baseline['video_id'],'Pinned whole-model input identities required.')
    u=row['unit'];m=u['metrics'];settings=m['settings'];n=20 if plan['method']=='primary' else 50
    require(u['kind']=='target-whole-pgd-input-gradient-workload-v1' and u['status']=='complete'
        and u['execution_binding'] is None and u['research_admitted'] is False and u['restored'] is True
        and u['GPU_performance_qualified'] is False and u['overhead_subtracted'] is False,'No implicit adoption or partial-unit promotion.')
    require(settings=={'method':plan['method'],'steps':n,'restarts':2,'attack_masks':8,'scoring_masks':4,
        'epsilon_255':4.,'step_size_255':plan['step_size_255'],'profile':'research-primary-eot8-v1'}
        and m['method']==plan['method'] and m['video_id']==baseline['video_id']
        and m['seeds']==_seeds(m['video_id'],plan['sensitivity_banks'])
        and len(u['steps'])==2*n and len(m['scoring_audits'])==plan['sensitivity_banks'],'Exact complete primary/convergence settings/starts/banks required.')
    engine='full-eot8-original-owned-v1' if role=='original-unit' else 'full-eot8-forward-preserving-owned-v1'
    require(u['route']==engine,'Distinct original/candidate engine identity required.')
    for i,s in enumerate(u['steps']):
        p=s['route']
        require(s['index']==i and sha(s['pixels']) and sha(s['gradient'])
            and s['consider_before_backward'] is True and len(s['losses'])==1 and finite(s['losses'][0])
            and p['kind']==engine and p['native_root_hook_removed'] is True and p['native_scopes']==1
            and type(p['native_calls']) is int and 0<p['native_calls']<=4096 and p['native_refusals']==0
            and finite(s['context_seconds_including_observations']) and 0<=s['context_seconds_including_observations']<=120,
            'Unchanged fresh per-objective owner, calls, values and 120-second bound required.')
        if role!='original-unit':require(p['forward_calls']==384 and p['initial_predictor_VJPs']==192,'Original full-model candidate call caps required.')
    scores=u['scores'];audits=plan['sensitivity_banks']
    require([s['bank'] for s in scores]==[1,0,0,0,0,1]+[i for i in range(2,audits+2) for _ in range(2)],'Original scoring order required.')
    for s in scores:
        require(s['grad_enabled'] is s['route_active'] is False and sha(s['pixels'])
                and s['draws']==(8 if s['bank']==0 else 4) and len(s['losses'])==1 and finite(s['losses'][0]),'Original no-grad scoring required.')
    require(scores[0]['pixels']==scores[1]['pixels']==scores[2]['pixels']==baseline['clean_sha256']
        and scores[5]['pixels']==u['adversarial'] and sha(u['adversarial'])
        and scores[0]['losses'][0]==m['clean_loss'] and scores[1]['losses'][0]==m['attack_clean_loss']
        and scores[5]['losses'][0]==m['adversarial_loss'],'Clean/selected pixels and scoring metrics must join.')
    for i in range(audits):
        a,b=scores[6+2*i:8+2*i]
        require(a['pixels']==baseline['clean_sha256'] and b['pixels']==u['adversarial']
                and m['scoring_audits'][i]=={'clean_loss':a['losses'][0],'adversarial_loss':b['losses'][0]},'Original audit metrics must join.')
    require(all(finite(m[k]) for k in ('attack_loss','max_pixel_delta','seconds'))
        and 0<=m['max_pixel_delta']<=4/255+1e-7 and 0<=m['seconds']
        <=u['full_unit_seconds_including_guards_journal_and_observations']<=row['wall_seconds']<=720
        and type(row['peak_allocator_bytes']) is int and 0<row['peak_allocator_bytes']<=7*2**30,
        'Whole-unit clock/allocator design ceilings required.')
    return {'metadata_joined':True,'native_execution_authenticated_by_dictionary':False,**FALSE}


def join_units(original,candidate,replay):
    require([x['role'] for x in (original,candidate,replay)]==list(ROLES)
        and len({x['pid'] for x in (original,candidate,replay)})==3,'Three distinct whole-unit children required.')
    require(original['baseline']==candidate['baseline']==replay['baseline']
        and original['unit']['metrics']['settings']==candidate['unit']['metrics']['settings']==replay['unit']['metrics']['settings']
        and original['unit']['metrics']['seeds']==candidate['unit']['metrics']['seeds']==replay['unit']['metrics']['seeds'],
        'Matched complete units require identical original inputs/settings/banks/starts.')
    def logical(row):
        u=row['unit']
        return {'route':u['route'],'metrics':{k:v for k,v in u['metrics'].items() if k!='seconds'},
            'steps':[{k:v for k,v in s.items() if k!='context_seconds_including_observations'} for s in u['steps']],
            'scores':u['scores'],'adversarial':u['adversarial']}
    require(logical(candidate)==logical(replay),'Fresh candidate rollout replay differs.')
    return {'kind':'whole-pgd-matched-unit-replay-metadata-v1','candidate_replay_logical_exact':True,
            'original_candidate_numerical_agreement_asserted':False,
            'matched_inputs_settings_and_seeds':True,'metadata_authenticates_execution':False,**FALSE}


def wrap_work(unit, *, role, plan, baseline, pid, wall_seconds, peak_allocator_bytes):
    row={'kind':'whole-pgd-unit-metadata-v1','role':role,'control_sha256':plan['control_sha256'],
        'source_manifest_sha256':plan['source_manifest_sha256'],'plan_sha256':plan['plan_sha256'],
        'pid':pid,'baseline':baseline,'unit':unit,'wall_seconds':wall_seconds,
        'peak_allocator_bytes':peak_allocator_bytes,**FALSE}
    validate_work(row,plan);return row
