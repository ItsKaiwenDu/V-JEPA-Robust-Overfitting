"""Saved-only retrospective reader; this module has no execution entry.

The deep reader preserves frozen v6 gates. Only its two numerical-summary
comparisons use the separate validated bookkeeping policy. All old source,
claims, parent rejections and producer completion records remain unchanged.
An assessment cannot authorize a worker, close a cohort or adopt a backend.
"""
import copy
import json
import os
from pathlib import Path
import stat

from models import predictor_sdpa_component_v6 as frozen
from models import predictor_sdpa_component_records_v1 as rec
from models import predictor_sdpa_numerical_receipts_v1 as numerical
from models.predictor_sdpa_continuation_receipts_v3 import load_resource

KIND = 'fixed-SDPA-retrospective-receipt-assessment-v1'
ROOT = Path(__file__).resolve().parents[1]
CELLS, BOUNDS = frozen.CELLS, frozen.BOUNDS
capture, load_reference, load_profile = frozen.capture, frozen.load_reference, frozen.load_profile


def _load_worker(path,p,plan_sha,identity,index,*,reports):
    import numpy as np
    from models.attention_saved_evidence import agreement
    from models import predictor_sdpa_launch_evidence_v2 as events
    w=rec.load(path,kind=rec.WORKER_KIND,plan_sha256=plan_sha);arm,replay=CELLS[index]
    if (w.get('index')!=index or w.get('arm')!=arm or w.get('replay')!=replay or w.get('local_tiny') is not p['local_tiny']
            or w.get('capture_identity')!=identity or w.get('numerical_passed') is not True or w.get('native_baseline_exact') is not (arm=='original')
            or w.get('flags_restored') is not True or w.get('inputs_unchanged') is not True or w.get('profile_replay_exact') is not True
            or type(w.get('pid')) is not int or w['pid']<=0 or type(w.get('oracle_build_calls')) is not int
            or w['oracle_build_calls']!=(1 if index==0 else 0) or type(w.get('wall_seconds')) not in (int,float)
            or not 0<w['wall_seconds']<=120):
        raise ValueError('Passing fixed-input component worker identity changed.')
    from models.predictor_sdpa_component_startup_v2 import validate as validate_startup
    validate_startup(w.get('startup'),pid=w['pid'],device='cpu' if p['local_tiny'] else 'cuda')
    shape=identity['layouts']['q']['shape'];values={}
    for phase in ('untimed','profiled'):
        if set(w['outputs'][phase])!=set(rec.RETURNS):raise ValueError('All actual component returns required.')
        for n,item in w['outputs'][phase].items():
            if item['file']!=phase+'-'+n+'.npy':raise ValueError('Component phase/file identity changed.')
            a=rec.array(path,item,dtype='<f4',shape=shape)
            if not np.any(a):raise ValueError('Finite nonzero component values required.')
            if phase=='untimed':values[n]=a
            if item['logical_sha256']!=w['outputs']['untimed'][n]['logical_sha256']:raise ValueError('Profiled component return did not exactly replay.')
    root=Path(p['reference_dir']).parent;ref,gold=load_reference(p['reference_dir'],p,plan_sha,identity)
    if (w.get('reference_manifest_sha256')!=rec.sha(Path(p['reference_dir'])/'reference.json')
            or w.get('reference_marker_sha256')!=rec.sha(Path(p['reference_dir'])/'complete.json')
            or index==0 and ref.get('builder_pid')!=w['pid']):raise ValueError('Component canonical reference receipt changed.')
    checks={n:agreement(values[n],gold[n],BOUNDS[n]) for n in rec.RETURNS}
    reports.append(numerical.compare_checks(checks,w.get('oracle_checks'),context='canonical-FP64'))
    parent,source,cap,_=capture(p)
    expected_runtime={k:v for k,v in source['baseline']['runtime'].items() if k!='model_execution'}
    expected_flags=cap['selected_call']['actual_call']['entry_flags'] if arm=='original' else {'flash':False,'mem_efficient':False,'math':True,'cudnn':False}
    if w.get('runtime')!=expected_runtime or w.get('effective_flags')!=expected_flags:raise ValueError('Captured runtime or backend selection changed.')
    if set(w.get('actual_inputs',{}))!=set(identity['logical_inputs']):raise ValueError('All actual device input identities required.')
    for n,item in w['actual_inputs'].items():
        layout=identity['layouts'][n]
        expected={'shape':layout['shape'],'stride':layout['stride'],'dtype':'torch.float32','device':'cpu' if p['local_tiny'] else 'cuda:0',
            'sha256':identity['logical_inputs'][n],'storage_offset_elements':layout['storage_offset_elements'],
            'storage_bytes':layout['storage_bytes'],'alias_group':layout['alias_group'],'requires_grad':n!='grad_output',
            'unknown_padding_filled_zero':True,'native_physical_alignment_identity_claimed':False}
        if item!=expected:raise ValueError('Actual fixed-input layout, alias, dtype or logical bytes changed.')
    if arm=='original':
        for n in rec.RETURNS:
            if w['outputs']['untimed'][n]['logical_sha256']!=cap['artifacts'][n]['source']['sha256']:
                raise ValueError('Standalone original saved values differ from actual native values.')
    else:
        original=_load_worker(root/'0',p,plan_sha,identity,0,reports=reports)
        actual_checks={n:agreement(values[n],rec.array(root/'0',original['outputs']['untimed'][n],dtype='<f4',shape=shape),BOUNDS[n]) for n in rec.RETURNS}
        reports.append(numerical.compare_checks(actual_checks,w.get('passing_original_checks'),context='passing-original'))
    prior=[];prior_pids=[]
    for i in range(index):
        old=_load_worker(root/str(i),p,plan_sha,identity,i,reports=reports);prior_pids.append(old['pid'])
        prior.append({'index':i,'worker_sha256':rec.sha(root/str(i)/'worker.json'),'complete_marker_sha256':rec.sha(root/str(i)/'complete.json')})
    if prior!=w.get('previous_receipts') or w['pid'] in prior_pids or w['pid']==source['pid']:
        raise ValueError('Fresh component process or serial predecessor binding changed.')
    if replay:
        old=rec.load(root/str(index-2),kind=rec.WORKER_KIND,plan_sha256=plan_sha)
        if (w.get('fresh_replay_reference_sha256')!=rec.sha(root/str(index-2)/'worker.json')
                or any(w['outputs']['untimed'][n]['logical_sha256']!=old['outputs']['untimed'][n]['logical_sha256'] for n in rec.RETURNS)):
            raise ValueError('Saved fresh-process component replay changed.')
    trace=load_profile(path,w['trace'],w['runtime'])
    observed=events.verify(trace,arm=arm,device='cpu' if p['local_tiny'] else 'cuda')
    if observed!=w['backend_evidence']:raise ValueError('Saved backend association changed.')
    return w


def assess_saved_worker(path, plan_path, index, *, capture_root=None):
    """Validate saved artifacts under the new policy without admitting an old run."""
    if type(index) is not int or index not in range(4):
        raise ValueError('One of the four saved component indices required.')
    path, plan_path = Path(path), Path(plan_path)
    s = plan_path.lstat()
    if (not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid != os.getuid()
            or stat.S_IMODE(s.st_mode) != 0o600 or not 0 < s.st_size <= 128 * 1024):
        raise ValueError('Private bounded saved plan required.')
    plan_sha = rec.sha(plan_path)
    stored = json.loads(plan_path.read_text())
    p = copy.deepcopy(stored)
    p.update(repository=str(ROOT), reference_dir=str((path.parent/'reference').resolve()))
    if capture_root is not None:
        p['capture_root'] = str(Path(capture_root).resolve())
    frozen.validate_plan(p)
    identity = capture(p)[3]
    reports = []
    worker = _load_worker(path, p, plan_sha, identity, index, reports=reports)
    resources = []
    if not p['local_tiny']:
        for i in range(index + 1):
            previous = rec.load(path.parent/str(i), kind=rec.WORKER_KIND, plan_sha256=plan_sha)
            resources.append(load_resource(path.parent/str(i), plan_sha256=plan_sha, index=i, worker=previous))
    if rec.sha(plan_path) != plan_sha:
        raise ValueError('Saved plan changed during assessment.')
    return {'kind': KIND, 'policy': numerical.POLICY, 'assessment_passed': True,
            'index': index, 'arm': worker['arm'], 'replay': worker['replay'], 'local_CPU_fixture': p['local_tiny'],
            'plan_sha256': plan_sha, 'worker_sha256': rec.sha(path/'worker.json'),
            'numerical_receipt_comparisons': reports, 'backend_evidence': worker['backend_evidence'],
            'resource_receipts': resources, 'saved_paths_translated_only_in_memory': True,
            'producer_records_rewritten': False, 'historical_claim_or_parent_status_changed': False,
            'new_worker_or_cohort_admitted': False, 'fresh_execution_or_reference_build': False,
            'full_model_candidate_adopted': False, 'research_admitted': False, 'performance_qualified': False}
