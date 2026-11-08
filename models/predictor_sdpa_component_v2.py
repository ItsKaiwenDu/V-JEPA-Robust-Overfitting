"""Fixed-input attention component protocol; current entry is local CPU only."""
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from models import predictor_sdpa_component_records_v1 as rec
from models import predictor_sdpa_capture_record_v1 as native

PLAN_KIND='native-SDPA-fixed-input-plan-v2'
CELLS=(('original',0),('math',0),('original',1),('math',1))
BOOK={'rtol':1e-12,'atol':1e-14}
BOUNDS={'output':{'rtol':1e-4,'atol':2e-5},**{n:{'rtol':.002,'atol':2e-7} for n in ('dq','dk','dv')}}
SOURCE_NAMES=('models/predictor_sdpa_component_v2.py','models/predictor_sdpa_component_records_v1.py',
    'models/predictor_sdpa_component_events_v1.py','models/predictor_sdpa_capture_record_v1.py',
    'models/attention_saved_evidence.py','models/execution.py','models/training.py',
    'models/predictor_eot8_capture_v1.py','models/predictor_eot8_adapter_v1.py',
    'scripts/check_predictor_sdpa_components_v2.py','tests/test_predictor_sdpa_component_startup_v2.py',
    'models/predictor_sdpa_component_startup_v2.py','models/predictor_sdpa_component_target_v2.py')

def read(p):return json.loads(Path(p).read_text())
def hashed_json(v):return hashlib.sha256(rec.encoded(v)).hexdigest()

def validate_plan(p):
    if (p.get('kind')!=PLAN_KIND or type(p.get('local_tiny')) is not bool or p.get('cells')!=[list(x) for x in CELLS]
            or p.get('bounds')!=BOUNDS or p.get('bookkeeping')!=BOOK or p.get('reference_builder_calls')!=1
            or p.get('worker_seconds')!=120 or p.get('untimed_iterations')!=1 or p.get('profiled_iterations')!=1
            or any(p.get(k) is not False for k in ('research_admitted','performance_qualified','full_model_candidate_adopted',
                'pure_backward_arithmetic_isolated','paid_compute','automatic_retry','private_backward_API'))
            or set(p.get('source_sha256',{}))!=set(SOURCE_NAMES)):
        raise ValueError('Exact diagnostic component contract required; no adoption or expanded work.')
    root=Path(p['repository'])
    if root!=Path(__file__).resolve().parents[1]:raise ValueError('Bound component source root changed.')
    for name,want in p['source_sha256'].items():
        if rec.sha(root/name)!=want:raise ValueError('Component scientific source changed: '+name)
    for key in ('native_worker_sha256','boundary_sha256','capture_control_sha256'):
        if not isinstance(p.get(key),str) or len(p[key])!=64 or any(c not in '0123456789abcdef' for c in p[key]):raise ValueError('Exact native source hashes required.')
    if set(p['full_native_anchor_logical_sha256'])!={'loss','input'}:raise ValueError('Passing original loss/input value anchors required.')
    return p

def make_local_plan(root,reference_dir):
    root=Path(root);review=read(root/'report/checks/2026-11-06-predictor-sdpa-observer-v1-local-review.json')
    source=next(w for w in review['workers'] if w['case']=='creator-0');parent=root/source['worker']['path']
    if rec.sha(parent)!=source['worker']['sha256']:raise ValueError('Reviewed local native parent changed.')
    w=read(parent)
    p={'kind':PLAN_KIND,'local_tiny':True,'repository':str(root),'capture_root':str(parent.parent),
        'native_worker_sha256':source['worker']['sha256'],'boundary_sha256':source['boundary']['sha256'],'capture_control_sha256':'d'*64,
        'full_native_anchor_logical_sha256':{n:w['artifacts'][n]['source']['sha256'] for n in ('loss','input')},
        'reference_dir':str(Path(reference_dir).resolve()),'cells':[list(x) for x in CELLS],
        'bounds':BOUNDS,'bookkeeping':BOOK,'reference_builder_calls':1,'worker_seconds':120,'untimed_iterations':1,'profiled_iterations':1,
        'research_admitted':False,'performance_qualified':False,'full_model_candidate_adopted':False,'pure_backward_arithmetic_isolated':False,
        'paid_compute':False,'automatic_retry':False,'private_backward_API':False,
        'source_sha256':{n:rec.sha(root/n) for n in SOURCE_NAMES}}
    validate_plan(p);return p

def capture(p):
    parent=Path(p['capture_root']);file=parent/'worker.json'
    if file.is_symlink() or file.stat().st_size>256*1024 or rec.sha(file)!=p['native_worker_sha256']:raise ValueError('Native parent binding changed.')
    w=read(file);kind='original-SDPA-observer-local-CPU-rehearsal-v1' if p['local_tiny'] else 'original-SDPA-native-target-capture-v1'
    if (w.get('kind')!=kind or w.get('local_tiny') is not p['local_tiny'] or w.get('complete') is not True
            or any(w.get(k) is not True for k in ('original_value_replay_exact','model_unchanged','route_restored','observer_restored'))
            or any(w.get(k) is not False for k in ('research_admitted','performance_qualified','native_candidate_adopted'))
            or w.get('boundary_header_sha256')!=p['boundary_sha256']):raise ValueError('A completed passing original native value control is required.')
    folder=parent/'boundary'
    if rec.sha(folder/'boundary.json')!=p['boundary_sha256']:raise ValueError('Native boundary changed.')
    cap=native.load(folder,control_sha256=p['capture_control_sha256'])
    if cap['local_tiny'] is not p['local_tiny']:raise ValueError('Native component domains cannot be mixed.')
    for n in ('loss','input'):
        item=w['artifacts'][n];q=parent/item['file']
        if (item['file']!=n+'.pt' or q.is_symlink() or not 0<q.stat().st_size<=16*rec.MiB or rec.sha(q)!=item['sha256']
                or item['source']['sha256']!=p['full_native_anchor_logical_sha256'][n]):raise ValueError('Native full-loss/input anchor changed.')
    identity={'boundary_sha256':p['boundary_sha256'],'native_worker_sha256':p['native_worker_sha256'],
        'logical_inputs':{n:cap['artifacts'][n]['source']['sha256'] for n in ('q','k','v','grad_output')},
        'layouts':{n:{'shape':cap['artifacts'][n]['source']['shape'],'stride':cap['artifacts'][n]['source']['stride'],
            **{k:cap['source_storage'][n][k] for k in ('storage_offset_elements','storage_bytes','alias_group')}} for n in ('q','k','v','grad_output')}}
    return parent,w,cap,identity

def load_reference(path,p,plan_sha,identity):
    w=rec.load(path,kind=rec.REFERENCE_KIND,plan_sha256=plan_sha)
    if (w.get('capture_identity')!=identity or w.get('local_tiny') is not p['local_tiny'] or w.get('reference_is_passing_worker') is not False
            or w.get('builder_index')!=0 or w.get('oracle_calls')!=1 or w.get('query_chunk')!=128
            or w.get('bookkeeping')!=BOOK or w.get('oracle_source_sha256')!=p['source_sha256']['models/attention_saved_evidence.py']
            or set(w.get('arrays',{}))!=set(rec.RETURNS)):raise ValueError('Frozen canonical analytical reference binding changed.')
    shape=identity['layouts']['q']['shape']
    values={n:rec.array(path,item,dtype='<f8',shape=shape) for n,item in w['arrays'].items()}
    if any(w['arrays'][n]['file']!=n+'.npy' for n in rec.RETURNS):raise ValueError('Canonical fixed FP64 filenames required.')
    return w,values

def publish_reference(path,p,plan_sha,identity,inputs):
    from models.attention_saved_evidence import oracle
    path=Path(path);path.mkdir(mode=0o700,exist_ok=False);store=rec.Store(path)
    w={'kind':rec.REFERENCE_KIND,'plan_sha256':plan_sha,'complete':False,'local_tiny':p['local_tiny'],'capture_identity':identity,
        'builder_index':0,'builder_pid':os.getpid(),'oracle_calls':0,'query_chunk':128,'bookkeeping':BOOK,
        'oracle_source_sha256':p['source_sha256']['models/attention_saved_evidence.py'],'arrays':{},'reference_is_passing_worker':False,
        'research_admitted':False,'performance_qualified':False,'full_model_candidate_adopted':False,'pure_backward_arithmetic_isolated':False}
    store.json('reference.json',w)
    primary=None
    try:
        w['oracle_calls']=1;values=oracle(*(inputs[n].detach().cpu().numpy() for n in ('q','k','v','grad_output')),chunk=128)
        for n in rec.RETURNS:
            item=store.array(n+'.npy',values[n],dtype='<f8',shape=identity['layouts']['q']['shape']);w['arrays'][n]=item
            if not item['finite']:raise ValueError('Actual nonfinite FP64 reference retained; no acceptance.')
        w['complete']=True
    except BaseException as e:primary=e;w.update(error_type=type(e).__name__,error=str(e)[:2048]);raise
    finally:
        try:store.finish('reference.json',w,complete=w['complete'])
        except BaseException as e:
            if primary is None:raise
            primary.add_note('Reference publication failure also retained: '+repr(e))
    return w

def load_worker(path,p,plan_sha,identity,index):
    import numpy as np
    from models.attention_saved_evidence import agreement
    from models import predictor_sdpa_component_events_v1 as events
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
    if checks!=w.get('oracle_checks') or not all(c['passed'] for c in checks.values()):raise ValueError('Actual saved component/reference numerical bound rejected.')
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
        original=load_worker(root/'0',p,plan_sha,identity,0)
        actual_checks={n:agreement(values[n],rec.array(root/'0',original['outputs']['untimed'][n],dtype='<f4',shape=shape),BOUNDS[n]) for n in rec.RETURNS}
        if actual_checks!=w.get('passing_original_checks') or not all(c['passed'] for c in actual_checks.values()):raise ValueError('Saved candidate did not pass original component bounds.')
    prior=[];prior_pids=[]
    for i in range(index):
        old=load_worker(root/str(i),p,plan_sha,identity,i);prior_pids.append(old['pid'])
        prior.append({'index':i,'worker_sha256':rec.sha(root/str(i)/'worker.json'),'complete_marker_sha256':rec.sha(root/str(i)/'complete.json')})
    if prior!=w.get('previous_receipts') or w['pid'] in prior_pids or w['pid']==source['pid']:
        raise ValueError('Fresh component process or serial predecessor binding changed.')
    if replay:
        old=rec.load(root/str(index-2),kind=rec.WORKER_KIND,plan_sha256=plan_sha)
        if (w.get('fresh_replay_reference_sha256')!=rec.sha(root/str(index-2)/'worker.json')
                or any(w['outputs']['untimed'][n]['logical_sha256']!=old['outputs']['untimed'][n]['logical_sha256'] for n in rec.RETURNS)):
            raise ValueError('Saved fresh-process component replay changed.')
    if w['trace'].get('file')!='events.jsonl':raise ValueError('Actual component event file required.')
    trace=[json.loads(line) for line in rec.verified(path,w['trace']).splitlines()]
    if w['trace'].get('records')!=len(trace):raise ValueError('Component trace count changed.')
    observed=events.verify(trace,arm=arm,device='cpu' if p['local_tiny'] else 'cuda')
    if observed!=w['backend_evidence']:raise ValueError('Saved backend association changed.')
    return w

def previous(p,plan_sha,identity,index):
    root=Path(p['reference_dir']).parent;records=[]
    for i in range(index):
        folder=root/str(i);w=load_worker(folder,p,plan_sha,identity,i)
        records.append({'index':i,'worker_sha256':rec.sha(folder/'worker.json'),'complete_marker_sha256':rec.sha(folder/'complete.json')})
    return records

def _operands(parent,cap,device):
    import torch
    from models.predictor_eot8_capture_v1 import tensor_info
    cpu=native.rebuild_cpu_inputs(parent/'boundary',cap);groups={};result={};metadata={}
    for n in ('q','k','v','grad_output'):
        storage=cap['source_storage'][n];sig=cap['artifacts'][n]['source'];g=storage['alias_group'];size=storage['storage_bytes']
        if not 0<size<=16*rec.MiB or size%4:raise ValueError('Actual device backing extent invalid.')
        if g not in groups:groups[g]=torch.zeros(size//4,device=device,dtype=torch.float32)
        t=groups[g].as_strided(sig['shape'],sig['stride'],storage['storage_offset_elements']);t.copy_(cpu[n]);result[n]=t
    for n,t in result.items():
        sig=tensor_info(t);expected=cap['artifacts'][n]['source']
        if {k:sig[k] for k in ('shape','stride','dtype','sha256')}!={k:expected[k] for k in ('shape','stride','dtype','sha256')}:
            raise ValueError('Actual device input logical values/strides changed.')
        t.requires_grad_(n!='grad_output')
        actual_group=next((g for g,b in groups.items() if b.untyped_storage()._cdata==t.untyped_storage()._cdata),None)
        metadata[n]={**sig,'storage_offset_elements':t.storage_offset(),'storage_bytes':t.untyped_storage().nbytes(),
            'alias_group':actual_group,'requires_grad':t.requires_grad,
            'unknown_padding_filled_zero':True,'native_physical_alignment_identity_claimed':False}
    return result,metadata

def _iteration(inputs,arm,device):
    import torch
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from models import predictor_sdpa_component_events_v1 as events
    from models.predictor_eot8_adapter_v1 import flags
    before=flags();ctx=sdpa_kernel(SDPBackend.MATH) if arm=='math' else nullcontext()
    try:
        with ctx:
            inside=flags()
            if arm=='math' and inside!={'flash':False,'mem_efficient':False,'math':True,'cudnn':False}:raise ValueError('Exclusive math selection failed.')
            if arm=='original' and inside!=before:raise ValueError('Original component selection changed.')
            with torch.profiler.record_function(events.FORWARD):
                output=torch.nn.functional.scaled_dot_product_attention(*(inputs[n] for n in ('q','k','v')),dropout_p=0.,is_causal=False,attn_mask=None)
            with torch.profiler.record_function(events.BACKWARD):
                gradients=torch.autograd.grad(output,tuple(inputs[n] for n in ('q','k','v')),inputs['grad_output'])
        if device=='cuda':torch.cuda.synchronize()
        return dict(zip(rec.RETURNS,(output.detach(),*(t.detach() for t in gradients)))),inside
    finally:
        if flags()!=before:raise ValueError('Component flags not restored.')

def run(args):
    # The current executable path is deliberately local CPU only. A separately
    # reviewed/authorized target entry must be prepared with its operator.
    if platform.node() in ('linux10603','linux10609') or args.device!='cpu':
        raise ValueError('Only local fixed-input CPU rehearsal is prepared; no university/CUDA execution grant.')
    plan_path=Path(args.plan)
    if plan_path.is_symlink() or rec.sha(plan_path)!=args.plan_sha256:raise ValueError('Bound component plan changed.')
    p=validate_plan(read(plan_path))
    if p['local_tiny'] is not True:raise ValueError('Current entry accepts only the local CPU capture domain.')
    if type(args.index) is not int or args.index not in range(4):raise ValueError('Four fixed serial component cells required.')
    parent,source,cap,identity=capture(p)
    prior=previous(p,args.plan_sha256,identity,args.index)
    reference_path=Path(p['reference_dir'])
    if args.index:
        reference,_=load_reference(reference_path,p,args.plan_sha256,identity)
        first=rec.load(reference_path.parent/'0',kind=rec.WORKER_KIND,plan_sha256=args.plan_sha256)
        if first['reference_manifest_sha256']!=rec.sha(reference_path/'reference.json'):raise ValueError('Canonical reference changed since passing original worker.')
    elif reference_path.exists():raise FileExistsError('The single canonical FP64 reference cannot be rebuilt or replaced.')
    if Path(args.output)!=reference_path.parent/str(args.index):raise ValueError('Bound fixed cell output directory required.')
    return _execute(args,p,parent,source,cap,identity,prior)

def _execute(args,p,parent,source,cap,identity,prior):
    import numpy as np
    import torch
    from models.predictor_sdpa_component_startup_v2 import initialize
    from models.training import runtime_contract
    from models.predictor_eot8_capture_v1 import tensor_info
    from models.predictor_eot8_adapter_v1 import flags
    from models.attention_saved_evidence import agreement
    from models import predictor_sdpa_component_events_v1 as events
    arm,replay=CELLS[args.index];device=args.device;start=time.monotonic()
    if not 0<args.deadline_unix-time.time()<=120:raise ValueError('Original continuing component deadline required.')
    out=Path(args.output);out.mkdir(mode=0o700,exist_ok=False);store=rec.Store(out)
    w={'kind':rec.WORKER_KIND,'plan_sha256':args.plan_sha256,'index':args.index,'arm':arm,'replay':replay,
        'local_tiny':p['local_tiny'],'pid':os.getpid(),'complete':False,'capture_identity':identity,'previous_receipts':prior,
        'oracle_build_calls':0,'outputs':{},'numerical_passed':False,'native_baseline_exact':False,
        'profile_replay_exact':False,'flags_restored':False,'inputs_unchanged':False,
        'research_admitted':False,'performance_qualified':False,'full_model_candidate_adopted':False,'pure_backward_arithmetic_isolated':False}
    store.json('worker.json',w);primary=None
    try:
        w['startup']=initialize(device);torch.set_num_threads(4)
        runtime=runtime_contract(device);expected={k:v for k,v in source['baseline']['runtime'].items() if k!='model_execution'}
        if runtime!=expected:raise ValueError('Captured native runtime/precision changed.')
        w['runtime']=runtime;before_flags=flags()
        if before_flags!=cap['selected_call']['actual_call']['entry_flags']:raise ValueError('Original captured SDPA flags changed.')
        # Check actual full native values, not just a parent completion flag.
        for n in ('loss','input'):
            value=torch.load(parent/source['artifacts'][n]['file'],weights_only=True,map_location='cpu')
            if not isinstance(value,torch.Tensor) or tensor_info(value)['sha256']!=p['full_native_anchor_logical_sha256'][n]:
                raise ValueError('Actual native full-model value anchor changed.')
        inputs,w['actual_inputs']=_operands(parent,cap,device)
        signatures={n:tensor_info(t) for n,t in inputs.items()};versions={n:t._version for n,t in inputs.items()}
        reference_path=Path(p['reference_dir'])
        if args.index==0:
            publish_reference(reference_path,p,args.plan_sha256,identity,inputs);w['oracle_build_calls']=1
        reference,values=load_reference(reference_path,p,args.plan_sha256,identity)
        w['reference_manifest_sha256']=rec.sha(reference_path/'reference.json');w['reference_marker_sha256']=rec.sha(reference_path/'complete.json')
        def unchanged():
            if time.time()>args.deadline_unix or time.monotonic()-start>120:raise TimeoutError('Continuing component allowance exhausted.')
            if previous(p,args.plan_sha256,identity,args.index)!=prior:raise ValueError('Earlier component evidence changed.')
            validate_plan(p);capture(p)
            if rec.sha(reference_path/'reference.json')!=w['reference_manifest_sha256'] or rec.sha(reference_path/'complete.json')!=w['reference_marker_sha256']:
                raise ValueError('Fixed reference changed during component work.')
            for n,t in inputs.items():
                if tensor_info(t)!=signatures[n] or t._version!=versions[n] or t.grad is not None:raise ValueError('Native fixed input/gradient bytes or versions changed.')
            if flags()!=before_flags or runtime_contract(device)!=runtime:raise ValueError('Component runtime/flags changed.')
        unchanged()
        def save_phase(phase,actual):
            w['outputs'][phase]={}
            for n,t in actual.items():
                w['outputs'][phase][n]=store.array(phase+'-'+n+'.npy',t.cpu().numpy(),dtype='<f4',shape=identity['layouts']['q']['shape'])
            store.json('worker.json',w)
        actual,inside=_iteration(inputs,arm,device);save_phase('untimed',actual);w['effective_flags']=inside
        w['oracle_checks']={n:agreement(actual[n].cpu().numpy(),values[n],BOUNDS[n]) for n in rec.RETURNS}
        if not all(c['passed'] for c in w['oracle_checks'].values()):raise ValueError('Original component FP64 bound rejected; no tolerance change.')
        if any(not torch.isfinite(t).all() or not t.abs().max()>0 for t in actual.values()):raise ValueError('Finite nonzero component returns required.')
        if arm=='original':
            for n,t in actual.items():
                expected=native.load_tensor(parent/'boundary',n,cap)
                if tensor_info(t)['sha256']!=tensor_info(expected)['sha256']:raise ValueError('Standalone original did not exactly replay native boundary values.')
            w['native_baseline_exact']=True
        else:
            original=load_worker(reference_path.parent/'0',p,args.plan_sha256,identity,0)
            w['passing_original_checks']={n:agreement(actual[n].cpu().numpy(),rec.array(reference_path.parent/'0',original['outputs']['untimed'][n],dtype='<f4',shape=identity['layouts']['q']['shape']),BOUNDS[n]) for n in rec.RETURNS}
            if not all(x['passed'] for x in w['passing_original_checks'].values()):raise ValueError('Original candidate/reference component bound rejected; no tolerance change.')
        if replay:
            index=args.index-2;old=load_worker(reference_path.parent/str(index),p,args.plan_sha256,identity,index)
            for n,item in w['outputs']['untimed'].items():
                if item['logical_sha256']!=old['outputs']['untimed'][n]['logical_sha256']:raise ValueError('Fresh component replay is not bitwise exact.')
            w['fresh_replay_reference_sha256']=rec.sha(reference_path.parent/str(index)/'worker.json')
        w['numerical_passed']=True;unchanged()
        activities=[torch.profiler.ProfilerActivity.CPU]
        if device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=False,with_stack=False,profile_memory=False) as profile:
            profiled,profile_flags=_iteration(inputs,arm,device)
        save_phase('profiled',profiled)
        rows=events.capture(profile);w['trace']=store.write('events.jsonl',lambda f:[f.write(rec.encoded(row)) for row in rows])
        w['trace']['records']=len(rows)
        if profile_flags!=inside:raise ValueError('Profile selection context changed.')
        if any(w['outputs']['untimed'][n]['logical_sha256']!=w['outputs']['profiled'][n]['logical_sha256'] for n in rec.RETURNS):
            raise ValueError('Profiled component returns did not replay exactly.')
        w['profile_replay_exact']=True;w['backend_evidence']=events.verify(rows,arm=arm,device=device)
        unchanged();w.update(complete=True,flags_restored=True,inputs_unchanged=True)
    except BaseException as e:primary=e;w.update(error_type=type(e).__name__,error=str(e)[:2048]);raise
    finally:
        w['wall_seconds']=time.monotonic()-start
        try:store.finish('worker.json',w,complete=w['complete'])
        except BaseException as e:
            if primary is None:raise
            primary.add_note('Component publication failure also retained: '+repr(e))
    return w
