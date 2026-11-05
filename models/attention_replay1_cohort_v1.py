"""Saved scientific checks and one explicitly bounded original NumPy return.

Scientific imports are deferred until dispatch; this module has no parent loop.
An accepted imported subset is an attestation, never a new complete capture.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from models import attention_replay1_contract_v1 as contract
from models.oracle_boundary_observer_v1 import bind_function,record,read_record,digest

KIND='attention-replay1-cohort-science-v1'
OUTPUTS=('output','dq','dk','dv')
INPUTS=('q','k','v','cotangent')
MIB=2**20
QUALIFICATION={'original_parent_session_qualified':False,'performance_qualified':False,'research_admitted':False,
    'old_refusals_waived':False,'full_model_qualified':False,'optimization_adopted':False}


class Checks:
    def __init__(self):self.coverage={}
    def require(self,condition,message,*,attested=False):
        self.coverage.setdefault(message,[]).append('accepted-reference-attestation' if attested else 'direct')
        if not condition:raise ValueError(message)


def saved_worker(folder,packet,*,local,expected,imported=False,preliminary=False):
    """The scientific parts of the frozen summary; no oracle or SDPA evaluation."""
    import numpy as np
    import torch
    from models.fixed_attention_packet import verify
    from models.fixed_attention_diagnostic import runtime_ok,ORACLE_ID
    from models.attention_component import backend_evidence
    from models.attention_saved_evidence import agreement,backend
    from models.cuda_profile import tensor_signature
    from models.precision_diagnostic import same_report
    from models.attention_return_capture_v1 import load_return
    folder,packet=Path(folder),Path(packet);checks=Checks();require=checks.require
    metadata,_=verify(packet,digest(packet/'FIXED-ATTENTION-HANDOFF.json'))
    fixture=read_record(packet/'fixtures/encoder-1/fixture.json');w=read_record(folder/'worker.json');cap=read_record(folder/'capture.json')
    identity=all(w.get(k)==expected[k] and cap.get(k)==expected[k] for k in ('case','arm','replay'))
    require(identity and w.get('kind')=='fixed-input-fp32-attention-diagnostic-v1' and w.get('complete') is True
        and w.get('passed') is True and w.get('stage')=='complete' and w.get('within_process_exact') is True
        and w.get('research_admitted') is False and w.get('tiny_fixture') is local
        and w.get('device')==('cpu' if local else 'cuda'),'Invalid immutable-input worker receipt.')
    common=w['common'];shape=fixture['files']['q']['tensor']['shape'];tolerances=fixture['tolerances']
    fixed={n:np.load(packet/'fixtures/encoder-1'/(n+'.npy'),allow_pickle=False) for n in (*INPUTS,*('oracle_'+n for n in OUTPUTS))}
    signatures={n:tensor_signature(torch.from_numpy(fixed[n])) for n in INPUTS}
    require(common['kind']==w['kind'] and common['handoff_sha256']==digest(packet/'FIXED-ATTENTION-HANDOFF.json')
        and common['case']=='encoder-1' and common['tiny_fixture'] is local and common['fixture']==metadata['fixtures']['encoder-1']
        and common['shape']==fixture['shape_metadata'] and common['operand_signatures']==signatures
        and common['tolerances']==tolerances and common['oracle']==ORACLE_ID and common['oracle_query_chunk']==128
        and runtime_ok(common['runtime'],w['device']) and common['runtime_generation_allowed'] is False
        and common['generator_replay_qualified'] is False and common['canonical_evaluation_artifact'] is False
        and common['synthetic_operands_only'] is True and common['released_weights_or_pixels_loaded'] is False
        and common['research_admitted'] is False,'Immutable fixture, layout, runtime, oracle or component bounds changed.')
    if local:require(common['hardware'] is None,'Worker hardware changed.')
    else:
        hw=common['hardware']
        require(hw['host']=='linux10603' and hw['uuid']=='GPU-e54fae90-feeb-8106-d8c0-1974fe48555f'
            and hw['gpu']=='NVIDIA GeForce RTX 3070' and hw['driver']=='580.178.04','Worker hardware changed.')
    def artifact(item,filename):
        p=folder/filename
        require(item['file']==filename and not p.is_symlink() and 0<p.stat().st_size<=32*MIB and digest(p)==item['sha256'],'Evidence tensor file changed.')
        data=torch.load(p,map_location='cpu',weights_only=True)
        require({n:tensor_signature(t) for n,t in data.items()}==item['tensor_signatures'],'Evidence tensor signatures changed.')
        return data
    require(set(w['evidence'])=={'inputs','device_inputs'},'Missing input or device-input snapshot.',attested=imported)
    for key,name in (('inputs','inputs.pt'),('device_inputs','device-inputs.pt')):
        if imported:
            require(w['evidence'][key]['file']==name and w['evidence'][key]['tensor_signatures']==signatures,
                'Early input evidence differs from the immutable fixture.',attested=True)
        else:
            data=artifact(w['evidence'][key],name)
            require(set(data)==set(INPUTS) and w['evidence'][key]['tensor_signatures']==signatures,
                'Early input evidence differs from the immutable fixture.')
    # The CPU export may pack V. The original gate is on the recorded LIVE layout.
    require(w['device_input_strides']==fixture['shape_metadata']['strides'],'Device input layout changed.')
    values=artifact(w['artifact'],'values.pt')
    require(set(values)==set(INPUTS)|{'actual_'+n for n in OUTPUTS}|{'oracle_'+n for n in OUTPUTS},'Incomplete untimed value evidence.')
    for n in (*INPUTS,*('oracle_'+n for n in OUTPUTS)):
        require(tensor_signature(values[n])==tensor_signature(torch.from_numpy(fixed[n])),'Saved operands/reference differ from bound bytes.')
    metrics={}
    for n in OUTPUTS:
        result=agreement(values['actual_'+n].numpy(),fixed['oracle_'+n],tolerances[n]);metrics[n]=result
        require(values['actual_'+n].dtype==torch.float32 and result['passed'] and same_report(result,w['oracle_checks'][n]),'Saved component numerical gate failed.')
        book=w['oracle_bookkeeping'][n]
        require(book['passed'] is True and book['rtol']==1e-12 and book['atol']==1e-14
            and math.isfinite(book['max_absolute_error']),'Local oracle bookkeeping receipt failed.',attested=imported)
        require(tensor_signature(values['actual_'+n])['abs_max']>0,'Saved component numerical gate failed.')
    observed=backend_evidence(w['backend_profile_events'],w['arm'],'cpu') if local else backend(w['backend_profile_events'],w['arm'])
    require(observed==w['backend_evidence'] and observed['passed'],'Actual backend evidence failed.')
    sigs={n:tensor_signature(values['actual_'+n]) for n in OUTPUTS}
    require([x['phase'] for x in w['replay_signatures']]==list(contract.PHASES[1:])
        and all(x['signatures']==sigs for x in w['replay_signatures']),'Incomplete or unequal profile/warmup/timing replay.')
    names=('forward_seconds','backward_seconds','total_seconds')
    require(len(w['timings'])==3 and all(set(t)==set(names) and all(math.isfinite(t[n]) and t[n]>0 for n in names)
        and math.isclose(t['forward_seconds']+t['backward_seconds'],t['total_seconds'],rel_tol=1e-10,abs_tol=1e-12) for t in w['timings']),
        'Malformed timing observations.')
    require(w['median_seconds']=={n:statistics.median(t[n] for t in w['timings']) for n in names},'Timing median mismatch.')
    require(0<w['wall_seconds']<=(30 if local else 120) and sum(t['total_seconds'] for t in w['timings'])<=w['wall_seconds'],'Worker clock bound failed.')
    memory=w['peak_cuda_allocated_bytes']
    require(memory is None if local else isinstance(memory,int) and 0<memory<=4*2**30,'Worker memory/domain bound failed.')
    if not preliminary:
        require(cap['outcome']=='original-worker-passed' and cap['capture_coverage_complete'] is True and cap['saved_value_checks_passed'] is True,
            'Saved capture was not accepted.')
    require(cap['CPU_oracle_calls']==cap['CPU_oracle_returns']==1 and cap['attention_calls']==cap['SDPA_calls']==6
        and [x['phase'] for x in cap['attention_returns']]==list(contract.PHASES),'Incomplete typed return capture.')
    require(all(cap.get(k) is True for k in ('profile_restored','trace_restored','profile_intact_until_original_exit','alarm_handler_restored')),
        'Observation/alarm state was not restored.')
    require(cap['local_tiny_rehearsal'] is local and digest(folder/'worker.json')==cap['original_worker_sha256'],'Capture worker identity differs.')
    for item in cap['function_identities'].values():
        require(item['bytecode_matches_bound_source'] and digest(packet/item['source'])==item['sha256'],'Original function identity differs.')
    if not preliminary:require(digest(folder/'saved-audit.json')==cap['saved_audit_sha256'],'Saved audit receipt changed.')
    direct_CPU_checks=0;direct_GPU_checks=0
    if not imported:
        cpu=load_return(folder,cap['CPU_return'],dtype='float64',shape=shape,limit=16*MIB)
        for n in OUTPUTS:
            require(bool(torch.allclose(torch.from_numpy(cpu[n]),torch.from_numpy(fixed['oracle_'+n]),rtol=1e-12,atol=1e-14))
                and float(np.max(np.abs(cpu[n]-fixed['oracle_'+n])))==w['oracle_bookkeeping'][n]['max_absolute_error'],'Actual CPU return gate failed.')
            direct_CPU_checks+=1
        for entry in cap['attention_returns']:
            arrays=load_return(folder,entry['artifact'],dtype='float32',shape=shape,limit=8*MIB)
            require(entry['artifact']['dtype_preserved'] is True,'Return dtype changed.')
            for n in OUTPUTS:
                meta=entry['artifact']['source_tensors'][n]
                require(meta['dtype']=='torch.float32' and meta['device'].split(':')[0]==('cpu' if local else 'cuda')
                    and meta['shape']==shape and len(meta['stride'])==4,'Original return dtype/device/layout metadata changed.')
                require(np.isfinite(arrays[n]).all() and arrays[n].tobytes()==values['actual_'+n].contiguous().numpy().tobytes()
                    and agreement(arrays[n],fixed['oracle_'+n],tolerances[n])['passed'],'Typed FP32 return or replay differs.')
                direct_GPU_checks+=1
    return {'worker':w,'values':values,'capture':cap,'coverage':checks.coverage,'component_checks':metrics,
        'direct_CPU_return_checks':direct_CPU_checks,'direct_GPU_return_checks':direct_GPU_checks,'imported_reference_subset':imported}


def reference_check(folder,previous,*,local):
    import torch
    from models.attention_saved_evidence import agreement
    from models.cuda_profile import tensor_signature
    from models.precision_diagnostic import same_report
    folder,previous=Path(folder),Path(previous);w=read_record(folder/'worker.json');p=read_record(previous/'worker.json')
    auth=read_record(folder/'authorization.json');want={'worker':previous.name,'worker_sha256':digest(previous/'worker.json'),'values_sha256':digest(previous/'values.pt')}
    if auth['reference']!=want or not p['complete'] or not p['passed'] or p['common']!=w['common']:
        raise ValueError('Original reference chain or common contract differs.')
    actual=torch.load(folder/'values.pt',map_location='cpu',weights_only=True);expected=torch.load(previous/'values.pt',map_location='cpu',weights_only=True)
    if not local and w['arm']=='efficient':
        if w['reference_comparison']!={'exact_replay':True} or any(tensor_signature(actual['actual_'+n])!=tensor_signature(expected['actual_'+n]) for n in OUTPUTS):
            raise ValueError('Original efficient fresh-process exact replay failed.')
    else:
        for n in OUTPUTS:
            result=agreement(actual['actual_'+n].numpy(),expected['actual_'+n].numpy(),w['common']['tolerances'][n])
            if not result['passed'] or not same_report(result,w['reference_comparison'][n]):
                raise ValueError('Fixed-input same-case candidate/reference comparison failed.')
    return True


def audit_new_worker(folder,packet,*,local):
    folder=Path(folder);index=next((i for i in (0,1) if contract.task(i,local)['worker']==folder.name),None)
    if index is None:raise ValueError('Worker outside the new replay-1 plan.')
    checked=saved_worker(folder,packet,local=local,expected=contract.task(index,local),preliminary=True)
    prior=folder.parent/(contract.imported(local)[0] if index==0 else contract.task(0,local)['worker'])
    reference_check(folder,prior,local=local)
    return {'kind':KIND,'saved_value_checks_passed':True,'local_tiny_analogue':local,'original_worker_sha256':digest(folder/'worker.json'),
        'CPU_return_checks':checked['direct_CPU_return_checks'],'GPU_return_checks':checked['direct_GPU_return_checks'],
        'attention_returns':6,'original_reference_check_passed':True,**QUALIFICATION}


def save_numpy_return(path,values):
    import numpy as np
    from models.attention_return_capture_v1 import signature
    path=Path(path);partial=path.with_name(path.name+'.partial')
    if path.exists() or partial.exists():raise FileExistsError('Preserve existing NumPy return.')
    if set(values)!=set(OUTPUTS):raise ValueError('NumPy oracle return coverage differs.')
    if any(not isinstance(a,np.ndarray) or a.dtype!=np.dtype('<f8') for a in values.values()):
        raise ValueError('Require actual NumPy FP64 returns; never silently cast.')
    arrays={n:np.array(a,copy=True,order='C') for n,a in values.items()}
    if sum(a.nbytes for a in arrays.values())+65536>16*MIB:raise ValueError('NumPy return exceeds 16 MiB.')
    with partial.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    if partial.stat().st_size>16*MIB:raise ValueError('NumPy NPZ exceeds bound; partial retained.')
    os.link(partial,path);partial.unlink()
    return {'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path),'source_dtype':'float64',
        'arrays':{n:signature(a) for n,a in arrays.items()},'dtype_preserved':True}


def audit_cohort(results,packet,*,local,numpy_values,full_old_roots=None):
    """Full old data is re-read only in the final Mac review, not invented remotely."""
    import numpy as np
    from models.precision_diagnostic import same_report
    from models.attention_saved_evidence import agreement
    results,packet=Path(results),Path(packet);checks=Checks();roots=[];all_checked=[]
    old_names=contract.imported(local);new=[contract.task(i,local) for i in (0,1)]
    for index,name in enumerate(old_names):
        root=Path(full_old_roots[index]) if full_old_roots else results/'workers'/name
        expected={'case':'encoder-1','arm':'math' if local else ('efficient','math')[index],'replay':index if local else 0}
        roots.append(root);all_checked.append(saved_worker(root,packet,local=local,expected=expected,imported=full_old_roots is None))
    for selected in new:
        root=results/'workers'/selected['worker'];roots.append(root)
        all_checked.append(saved_worker(root,packet,local=local,expected=selected))
    for checked in all_checked:
        for name,modes in checked['coverage'].items():checks.coverage.setdefault(name,[]).extend(modes)
    common=all_checked[0]['worker']['common']
    checks.require(all(x['worker']['common']==common for x in all_checked),'Worker hardware changed.')
    for role in (0,1):
        a,b=all_checked[role]['worker'],all_checked[role+2]['worker']
        checks.require(a['common']==common and b['common']==common and a['artifact']['tensor_signatures']==b['artifact']['tensor_signatures'],
            'Fresh-process fixture/runtime or bitwise result replay differs.')
    if read_record(roots[0]/'authorization.json')['reference'] is not None:raise ValueError('Original baseline reference was substituted.')
    for index,previous in ((1,0),(2,0),(3,2)):reference_check(roots[index],roots[previous],local=local)
    # Check both per-replay candidate decisions, using the original comparison bounds.
    for a,b in ((0,1),(2,3)):
        for n in OUTPUTS:
            actual=all_checked[b]['values']['actual_'+n].numpy();expected=all_checked[a]['values']['actual_'+n].numpy()
            result=agreement(actual,expected,common['tolerances'][n])
            checks.require(result['passed'] and same_report(result,all_checked[b]['worker']['reference_comparison'][n]),
                'Fixed-input same-case candidate/reference comparison failed.')
    reconstruct={}
    for checked in all_checked:
        for n in OUTPUTS:
            actual=checked['values']['oracle_'+n].numpy();expected=numpy_values[n]
            # Original summarizer argument order: saved canonical first, fresh NumPy second.
            passed=bool(np.allclose(actual,expected,rtol=1e-12,atol=1e-14))
            checks.require(passed,'Fixed FP64 reference failed independent NumPy reconstruction.')
            reconstruct[n]={'passed':passed,'max_absolute_error':float(np.abs(actual-expected).max()),'rtol':1e-12,'atol':1e-14}
    return {'kind':KIND,'cross_attempt_component_cohort_checks_passed':True,'local_tiny_analogue':local,
        'imported_workers':list(old_names),'new_workers':[x['worker'] for x in new],'scientific_check_coverage':checks.coverage,
        'NumPy_reconstruction_checks':reconstruct,'old_captures_directly_reread':full_old_roots is not None,
        'old_subset_attestation_only':full_old_roots is None,**QUALIFICATION}


def verify_once(results,packet,*,local,plan_sha=None,control_sha=None):
    """One fresh CPU-only verification phase, including exactly one original oracle."""
    import numpy as np
    import torch
    from models.attention_saved_evidence import oracle
    from models.attention_return_capture_v1 import load_return
    results,packet=Path(results),Path(packet);out=results/'cohort-verifier';out.mkdir(mode=0o700)
    state={'kind':KIND,'local_tiny_analogue':local,'complete':False,'passed':False,'NumPy_oracle_calls':0,'NumPy_oracle_returns':0,
        'NumPy_return':None,'phase_plan_sha256':plan_sha,'control_sha256':control_sha,'process_id':os.getpid(),'CUDA_initialized_before':torch.cuda.is_initialized(),**QUALIFICATION}
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('The cohort verifier must have hidden, uninitialized CUDA.')
    if sys.getprofile() is not None or sys.gettrace() is not None:raise ValueError('Verifier refuses an existing observer hook.')
    manifest=json.loads((packet/'FIXED-ATTENTION-HANDOFF.json').read_text())
    state['oracle_identity']=bind_function(oracle,packet,manifest,'oracle');started=time.monotonic();failure=None
    def hook(frame,event,arg):
        if event=='c_call' and getattr(arg,'__name__','')=='scaled_dot_product_attention':raise ValueError('CPU verifier cannot evaluate SDPA.')
        if frame.f_code is oracle.__code__:
            if event=='call':
                state['NumPy_oracle_calls']+=1
                if state['NumPy_oracle_calls']!=1:raise ValueError('Only one original NumPy oracle call is allowed.')
            elif event=='return' and arg is not None:
                state['NumPy_oracle_returns']+=1
                if state['NumPy_oracle_returns']!=1:raise ValueError('Only one original NumPy return is allowed.')
                state['NumPy_return']=save_numpy_return(out/'NumPy-return.npz',arg)
                state['return_saved_before_gate']=True
    try:
        from models.attention_fixture import load
        item=manifest['fixtures']['encoder-1']
        frozen,_=load(packet/item['folder'],item['sha256'],case='encoder-1',tiny=local)
        operands=[frozen[n].numpy() for n in INPUTS]
        state['operand_source']='Original fixed loader with original post-RoPE views; no regeneration or layout substitution.'
        sys.setprofile(hook)
        try:values=oracle(*operands,chunk=128)
        finally:sys.setprofile(None)
        if state['NumPy_oracle_calls']!=state['NumPy_oracle_returns'] or state['NumPy_oracle_returns']!=1 or not state.get('return_saved_before_gate'):
            raise ValueError('NumPy return coverage incomplete.')
        shape=list(operands[0].shape);saved=load_return(out,state['NumPy_return'],dtype='float64',shape=shape,limit=16*MIB)
        if any(saved[n].tobytes()!=values[n].tobytes() for n in OUTPUTS):raise ValueError('Actual NumPy return changed during capture.')
        audit=audit_cohort(results,packet,local=local,numpy_values=saved)
        record(out/'saved-cohort-audit.json',audit)
        state.update(complete=True,passed=True,saved_audit_sha256=digest(out/'saved-cohort-audit.json'))
    except BaseException as exc:
        failure=exc;state.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        sys.setprofile(None);state.update(wall_seconds=time.monotonic()-started,CUDA_initialized_after=torch.cuda.is_initialized(),
            profile_restored=sys.getprofile() is None,trace_restored=sys.gettrace() is None)
        if state['CUDA_initialized_after']:state.update(complete=False,passed=False,error='Verifier initialized CUDA.')
        record(out/'verifier.json',state)
    return 0 if state['complete'] and state['passed'] else 1
