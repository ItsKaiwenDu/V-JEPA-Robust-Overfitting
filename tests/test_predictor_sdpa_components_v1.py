"""Actual saved-native CPU components, fresh replay, refusal and publication tests."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from models import predictor_sdpa_component_v1 as component
from models import predictor_sdpa_component_records_v1 as rec
from models import predictor_sdpa_component_events_v1 as events

ROOT=Path(__file__).resolve().parents[1]
CLI=ROOT/'scripts/check_predictor_sdpa_components_v1.py'


def write(path,value):
    path.write_bytes(rec.encoded(value));path.chmod(0o600)


def resign(folder,name,value):
    write(folder/name,value)
    write(folder/'complete.json',{'kind':value['kind'],'plan_sha256':value['plan_sha256'],
        'header':{'file':name,'bytes':(folder/name).stat().st_size,'sha256':rec.sha(folder/name)},'complete':True})


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    root=tmp_path_factory.mktemp('fixed-SDPA')/'cohort'
    p=subprocess.run([sys.executable,'-B',str(CLI),'--output',str(root)],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (root.parent/'cohort.log').write_bytes(p.stdout)
    assert p.returncode==0,p.stdout.decode()[-9000:]+''.join(x.read_text()[-4000:] for x in sorted(root.glob('*.log')))
    return root


def context(root):
    p=component.read(root/'component-plan.json');ps=rec.sha(root/'component-plan.json')
    return p,ps,component.capture(p)[3]


def rebase(cohort,tmp_path):
    root=tmp_path/'copy';shutil.copytree(cohort,root)
    p,_,identity=context(cohort);p['reference_dir']=str(root/'reference');write(root/'component-plan.json',p);ps=rec.sha(root/'component-plan.json')
    ref=component.read(root/'reference/reference.json');ref['plan_sha256']=ps;resign(root/'reference','reference.json',ref)
    for i in range(4):
        folder=root/str(i);w=component.read(folder/'worker.json');w['plan_sha256']=ps
        w['reference_manifest_sha256']=rec.sha(root/'reference/reference.json');w['reference_marker_sha256']=rec.sha(root/'reference/complete.json')
        w['previous_receipts']=[{'index':j,'worker_sha256':rec.sha(root/str(j)/'worker.json'),
            'complete_marker_sha256':rec.sha(root/str(j)/'complete.json')} for j in range(i)]
        if i>=2:w['fresh_replay_reference_sha256']=rec.sha(root/str(i-2)/'worker.json')
        resign(folder,'worker.json',w)
    return root,p,ps,identity


def test_four_fresh_workers_share_one_native_input_and_canonical_reference(cohort):
    p,ps,identity=context(cohort);ref,gold=component.load_reference(cohort/'reference',p,ps,identity)
    workers=[component.load_worker(cohort/str(i),p,ps,identity,i) for i in range(4)]
    assert len({w['pid'] for w in workers})==4
    assert [w['oracle_build_calls'] for w in workers]==[1,0,0,0]
    assert ref['oracle_calls']==1 and ref['builder_pid']==workers[0]['pid'] and not ref['reference_is_passing_worker']
    assert all(w['reference_manifest_sha256']==rec.sha(cohort/'reference/reference.json') for w in workers)
    for i,w in enumerate(workers):
        assert w['local_tiny'] and not w['performance_qualified'] and not w['research_admitted'] and not w['pure_backward_arithmetic_isolated']
        assert w['backend_evidence']['CPU_only'] and not w['backend_evidence']['CUDA_forward_and_backward_qualified']
        assert w['backend_evidence']['observed_family'] in ('math','cpu-flash')
        if i%2:assert w['backend_evidence']['observed_family']=='math'
        for name in ('q','k','v','grad_output'):
            a=w['actual_inputs'][name];layout=identity['layouts'][name]
            assert a['sha256']==identity['logical_inputs'][name]
            assert a['stride']==layout['stride'] and a['storage_offset_elements']==layout['storage_offset_elements']
            assert a['storage_bytes']==layout['storage_bytes'] and a['alias_group']==layout['alias_group']
            assert not a['native_physical_alignment_identity_claimed'] and a['unknown_padding_filled_zero']
        assert w['actual_inputs']['v']['storage_offset_elements']>0
        for name in rec.RETURNS:
            assert w['outputs']['untimed'][name]['logical_sha256']==w['outputs']['profiled'][name]['logical_sha256']
            if i>=2:assert w['outputs']['untimed'][name]['logical_sha256']==workers[i-2]['outputs']['untimed'][name]['logical_sha256']
    with pytest.raises((ValueError,FileNotFoundError)):rec.load(cohort/'reference',kind=rec.WORKER_KIND,plan_sha256=ps)


def test_canonical_FP64_values_independently_reconstruct_from_actual_native_inputs(cohort):
    from models.attention_saved_evidence import oracle,agreement
    p,ps,identity=context(cohort);_,saved=component.load_reference(cohort/'reference',p,ps,identity)
    parent,_,cap,_=component.capture(p)
    inputs=component.native.rebuild_cpu_inputs(parent/'boundary',cap)
    reconstructed=oracle(*(inputs[n].numpy() for n in ('q','k','v','grad_output')),chunk=128)
    checks={n:agreement(saved[n],reconstructed[n],component.BOOK) for n in rec.RETURNS}
    assert all(c['passed'] for c in checks.values())
    write(cohort.parent/'independent-reference-reconstruction.json',{'CPU_only':True,'SDPA_rerun':False,'bookkeeping':component.BOOK,'checks':checks})


def test_later_fresh_worker_cannot_rebuild_oracle_and_keeps_reference_bytes(cohort,tmp_path):
    root,p,ps,identity=rebase(cohort,tmp_path);(root/'1').rename(root/'preserved-original-math-worker')
    before={f.name:rec.sha(f) for f in (root/'reference').iterdir()}
    code='''import sys,time
from pathlib import Path
from types import SimpleNamespace
from models import attention_saved_evidence as a
def forbidden(*args,**kwargs):raise AssertionError('A later worker must never rebuild the canonical oracle')
a.oracle=forbidden
from models.predictor_sdpa_component_v1 import run
run(SimpleNamespace(plan=Path(sys.argv[1]),plan_sha256=sys.argv[2],output=Path(sys.argv[3]),index=1,device='cpu',deadline_unix=time.time()+120))'''
    proc=subprocess.run([sys.executable,'-B','-c',code,str(root/'component-plan.json'),ps,str(root/'1')],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (root/'fresh-math-with-oracle-disabled.log').write_bytes(proc.stdout)
    assert proc.returncode==0,proc.stdout.decode()[-8000:]
    w=component.load_worker(root/'1',p,ps,identity,1);assert w['oracle_build_calls']==0
    assert before=={f.name:rec.sha(f) for f in (root/'reference').iterdir()}


@pytest.mark.parametrize('fault',('admission','pure-backward','runtime','flags','input-hash','input-offset','alias','nonfresh-PID','native-replay','serial-receipt','oracle-count','profile-exact','replay-binding'))
def test_rehashed_worker_metadata_cannot_relabel_or_qualify_a_component(cohort,tmp_path,fault):
    root,p,ps,identity=rebase(cohort,tmp_path);folder=root/'3';w=component.read(folder/'worker.json')
    if fault=='admission':w['research_admitted']=True
    elif fault=='pure-backward':w['pure_backward_arithmetic_isolated']=True
    elif fault=='runtime':w['runtime']['threads']=8
    elif fault=='flags':w['effective_flags']['flash']=True
    elif fault=='input-hash':w['actual_inputs']['grad_output']['sha256']='e'*64
    elif fault=='input-offset':w['actual_inputs']['v']['storage_offset_elements']=0
    elif fault=='alias':w['actual_inputs']['v']['alias_group']=999
    elif fault=='nonfresh-PID':w['pid']=component.read(root/'0/worker.json')['pid']
    elif fault=='native-replay':w['native_baseline_exact']=True
    elif fault=='serial-receipt':w['previous_receipts'][1]['worker_sha256']='e'*64
    elif fault=='oracle-count':w['oracle_build_calls']=1
    elif fault=='profile-exact':w['profile_replay_exact']=False
    else:w['fresh_replay_reference_sha256']='e'*64
    resign(folder,'worker.json',w)
    with pytest.raises(ValueError):component.load_worker(folder,p,ps,identity,3)


def test_rehashed_actual_bad_values_are_rejected_despite_passing_booleans(cohort,tmp_path):
    root,p,ps,identity=rebase(cohort,tmp_path);folder=root/'1';w=component.read(folder/'worker.json')
    for phase in ('untimed','profiled'):
        item=w['outputs'][phase]['dq'];path=folder/item['file'];a=np.load(path,allow_pickle=False);a.flat[0]+=1
        with path.open('wb') as f:np.save(f,a,allow_pickle=False)
        item.update(bytes=path.stat().st_size,sha256=rec.sha(path),logical_sha256=hashlib.sha256(a.tobytes()).hexdigest())
    resign(folder,'worker.json',w)
    with pytest.raises(ValueError,match='numerical bound'):component.load_worker(folder,p,ps,identity,1)


@pytest.mark.parametrize('fault',('changed-reference','rehashed-reference','reference-marker','missing-original','failed-original','earlier-bytes'))
def test_serial_barriers_refuse_changed_or_missing_reference_and_prior_workers(cohort,tmp_path,fault):
    root,p,ps,identity=rebase(cohort,tmp_path)
    if fault in ('changed-reference','rehashed-reference'):
        folder=root/'reference';r=component.read(folder/'reference.json');item=r['arrays']['dq'];path=folder/item['file']
        a=np.load(path,allow_pickle=False);a.flat[0]+=1
        with path.open('wb') as f:np.save(f,a,allow_pickle=False)
        if fault=='rehashed-reference':
            item.update(bytes=path.stat().st_size,sha256=rec.sha(path),logical_sha256=hashlib.sha256(a.tobytes()).hexdigest());resign(folder,'reference.json',r)
    elif fault=='reference-marker':(root/'reference/complete.json').unlink()
    elif fault=='missing-original':(root/'0/complete.json').unlink()
    elif fault=='failed-original':write(root/'0/failure.json',{})
    else:(root/'0/untimed-output.npy').write_bytes(b'changed earlier actual bytes')
    with pytest.raises((ValueError,FileNotFoundError)):component.previous(p,ps,identity,3)


FAULTS={
    'numerical':'''old=c._iteration
def bad(inputs,arm,device):
 out,flags=old(inputs,arm,device);out['dq']=out['dq']+1.;return out,flags
c._iteration=bad''',
    'profile':'''old=c._iteration;calls=0
def bad(inputs,arm,device):
 global calls
 out,flags=old(inputs,arm,device);calls+=1
 if calls==2:out['dv']=out['dv']+1.
 return out,flags
c._iteration=bad''',
    'backend':'''from models import predictor_sdpa_component_events_v1 as e
def bad(*a,**k):raise ValueError('cpu-backend-association-refusal')
e.verify=bad''',
    'write':'''old=c.rec.Store.write
def bad(self,name,writer):
 if name=='untimed-dq.npy':
  def fail(f):writer(f);raise OSError('primary-component-write-error')
  return old(self,name,fail)
 return old(self,name,writer)
c.rec.Store.write=bad''',
    'nonfinite-reference':'''from models import attention_saved_evidence as a
old=a.oracle
def bad(*args,**kw):
 values=old(*args,**kw);values['dq'].flat[0]=float('nan');return values
a.oracle=bad'''}


@pytest.mark.parametrize('fault',tuple(FAULTS))
def test_real_child_failures_preserve_values_and_stop_before_next_phase(tmp_path,fault):
    root=tmp_path/'fault';root.mkdir(mode=0o700);p=component.make_local_plan(ROOT,root/'reference');write(root/'plan.json',p);ps=rec.sha(root/'plan.json')
    code='''import sys,time
from pathlib import Path
from types import SimpleNamespace
from models import predictor_sdpa_component_v1 as c
'''+FAULTS[fault]+'''
c.run(SimpleNamespace(plan=Path(sys.argv[1]),plan_sha256=sys.argv[2],output=Path(sys.argv[3]),index=0,device='cpu',deadline_unix=time.time()+120))'''
    proc=subprocess.run([sys.executable,'-B','-c',code,str(root/'plan.json'),ps,str(root/'0')],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (root/'child.log').write_bytes(proc.stdout);assert proc.returncode==1,proc.stdout.decode()
    assert not (root/'0/complete.json').exists() and not (root/'1').exists()
    if fault=='write':
        assert (root/'0/untimed-output.npy').exists() and (root/'0/untimed-dq.npy.partial').exists()
        assert b'primary-component-write-error' in proc.stdout and b'publication failure also retained' in proc.stdout
    elif fault=='nonfinite-reference':
        r=rec.load(root/'reference',kind=rec.REFERENCE_KIND,plan_sha256=ps,require_complete=False)
        assert not r['complete'] and not r['arrays']['dq']['finite']
        a=rec.array(root/'reference',r['arrays']['dq'],dtype='<f8',shape=[1,2,32,32],finite=False);assert np.isnan(a.flat[0])
        assert not (root/'0/untimed-output.npy').exists()
    else:
        w=rec.load(root/'0',kind=rec.WORKER_KIND,plan_sha256=ps,require_complete=False)
        assert set(w['outputs']['untimed'])==set(rec.RETURNS)
        if fault=='numerical':assert 'profiled' not in w['outputs'] and not w['oracle_checks']['dq']['passed']
        else:assert set(w['outputs']['profiled'])==set(rec.RETURNS) and (root/'0/events.jsonl').exists()


def args_for(tmp_path):
    p=component.make_local_plan(ROOT,tmp_path/'reference');write(tmp_path/'plan.json',p)
    return p,SimpleNamespace(plan=tmp_path/'plan.json',plan_sha256=rec.sha(tmp_path/'plan.json'),output=tmp_path/'0',index=0,device='cpu',deadline_unix=time.time()+120)


@pytest.mark.parametrize('fault',('university','cuda','expired','plan-source','captured-parent','domain','expanded-bounds','second-reference'))
def test_local_entry_refuses_unprepared_domains_changed_sources_and_reset(tmp_path,monkeypatch,fault):
    p,args=args_for(tmp_path)
    if fault=='university':monkeypatch.setattr(component.platform,'node',lambda:'linux10603')
    elif fault=='cuda':args.device='cuda'
    elif fault=='expired':args.deadline_unix=time.time()-1
    elif fault=='plan-source':p['source_sha256']['models/execution.py']='e'*64
    elif fault=='captured-parent':p['native_worker_sha256']='e'*64
    elif fault=='domain':p['local_tiny']=False
    elif fault=='expanded-bounds':p=copy.deepcopy(p);p['bounds']['dq']['atol']=1
    else:(tmp_path/'reference').mkdir(mode=0o700)
    if fault not in ('university','cuda','expired','second-reference'):write(args.plan,p);args.plan_sha256=rec.sha(args.plan)
    with pytest.raises((ValueError,FileExistsError)):component.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize('fault',('symlink','hardlink','mode','oversize','partial','mixed-markers'))
def test_typed_private_storage_and_completion_refuse_unsafe_files(cohort,tmp_path,fault):
    folder=tmp_path/'reference';shutil.copytree(cohort/'reference',folder);_,ps,_=context(cohort)
    if fault=='symlink':
        (folder/'reference.json').unlink();(folder/'reference.json').symlink_to(cohort/'reference/reference.json')
    elif fault=='hardlink':(folder/'other.json').hardlink_to(folder/'reference.json')
    elif fault=='mode':(folder/'reference.json').chmod(0o644)
    elif fault=='oversize':(folder/'reference.json').write_bytes(b'x'*(rec.LIMITS['reference.json']+1))
    elif fault=='partial':(folder/'output.npy.partial').write_bytes(b'partial-preserved')
    else:write(folder/'failure.json',{})
    with pytest.raises(ValueError):rec.load(folder,kind=rec.REFERENCE_KIND,plan_sha256=ps)


@pytest.mark.parametrize('fault',('large-shape','object-pickle','wrong-dtype','extra-bytes'))
def test_NPY_header_is_validated_before_load_or_allocation(tmp_path,monkeypatch,fault):
    folder=tmp_path/'array';folder.mkdir(mode=0o700);path=folder/'output.npy';buf=io.BytesIO();shape=[1,2,32,32]
    if fault=='large-shape':np.lib.format.write_array_header_1_0(buf,{'descr':'<f4','fortran_order':False,'shape':(2**28,4)})
    elif fault=='object-pickle':np.save(buf,np.array([{'unsafe':'pickle'}],dtype=object),allow_pickle=True)
    elif fault=='wrong-dtype':np.save(buf,np.zeros(shape,dtype=np.float64),allow_pickle=False)
    else:np.save(buf,np.zeros(shape,dtype=np.float32),allow_pickle=False);buf.write(b'extra')
    path.write_bytes(buf.getvalue());path.chmod(0o600)
    item={'file':path.name,'bytes':path.stat().st_size,'sha256':rec.sha(path),'shape':shape,'dtype':'<f4',
        'logical_sha256':'e'*64,'finite':True,'saved_before_numerical_gate':True}
    def forbidden(*a,**k):raise AssertionError('np.load must not run for an unsafe NPY header')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError):rec.array(folder,item,dtype='<f4',shape=shape)


def test_partial_writer_and_publication_collision_keep_original_bytes(tmp_path):
    folder=tmp_path/'files';folder.mkdir(mode=0o700);store=rec.Store(folder)
    (folder/'worker.json').write_bytes(b'foreign-appeared')
    with pytest.raises(FileExistsError):store.json('worker.json',{})
    assert (folder/'worker.json').read_bytes()==b'foreign-appeared'
    folder2=tmp_path/'partial';folder2.mkdir(mode=0o700);store=rec.Store(folder2)
    def fail(f):f.write(b'saved-prefix');raise OSError('write interruption')
    with pytest.raises(OSError):store.write('events.jsonl',fail)
    assert (folder2/'events.jsonl.partial').read_bytes()==b'saved-prefix' and store.poisoned
    with pytest.raises(ValueError):store.json('worker.json',{})


def trace_fixture(family,device='cuda'):
    # Handwritten parser fixtures ONLY; never actual CUDA execution evidence.
    rows=[]
    def row(name,start,end,parent=None,kernels=(),device_type='cpu',annotation=False):
        i=len(rows);rows.append({'index':i,'name':name,'device':device_type,'device_index':0,'thread':7,'fwd_thread':0,'sequence_nr':-1,
            'scope':0,'user_annotation':annotation,'parent':parent,'start_us':float(start),'end_us':float(end),'legacy':False,
            'linked_kernels':[{'name':n,'device':0,'duration_us':2.} for n in kernels]});return i
    f=row(events.FORWARD,0,40,annotation=True)
    if family=='efficient':
        names=('fmha_cutlassF_f32_parser_fixture','fmha_cutlassB_f32_parser_fixture')
        row(events.FAMILIES[family][0],1,35,f,(names[0],))
    elif family=='math':
        names=('sgemm_parser_fixture','softmax_warp_forward<parser_fixture>','softmax_warp_backward<parser_fixture>')
        a=row(events.FAMILIES[family][0],1,35,f)
        row('aten::bmm',2,10,a,(names[0],));row('aten::_softmax',12,25,a,(names[1],))
    else:
        names=();row(events.FAMILIES[family][0],1,35,f)
    b=row(events.BACKWARD,50,99,annotation=True)
    if family=='efficient':row(events.FAMILIES[family][1],51,95,b,(names[1],))
    elif family=='math':
        row(events.FAMILIES[family][1],51,70,b,(names[2],));row('aten::bmm',72,95,b,(names[0],))
    else:row(events.FAMILIES[family][1],51,95,b)
    if device=='cuda':
        for i,n in enumerate(names):row(n,100+i*4,102+i*4,device_type='cuda')
    else:
        for r in rows:r['linked_kernels']=[]
    return rows


@pytest.mark.parametrize('family',('efficient','math'))
def test_parser_requires_associated_CUDA_forward_and_backward_families(family):
    rows=trace_fixture(family);proof=events.verify(rows,arm='original',device='cuda')
    assert proof['observed_family']==family and proof['CUDA_forward_and_backward_qualified']
    assert all(proof['phase_correlated_CUDA_kernel_names'].values()) and not proof['default_family_was_assumed']
    assert not proof['timings_usable_for_speed_or_cost']


@pytest.mark.parametrize('fault',('no-device','annotation-only','no-link','backward-no-link','wrong-device','wrong-kernel','mixed','wrong-phase','bad-parent','candidate-not-math','CPU-label-as-CUDA'))
def test_flags_names_or_unassociated_CUDA_rows_cannot_qualify_backend(fault):
    rows=trace_fixture('efficient');arm='original'
    if fault=='no-device':rows=[r for r in rows if r['device']=='cpu']
    elif fault=='annotation-only':
        for r in rows:
            if r['device']=='cuda':r['user_annotation']=True
    elif fault=='no-link':
        for r in rows:r['linked_kernels']=[]
    elif fault=='backward-no-link':rows[3]['linked_kernels']=[]
    elif fault=='wrong-device':rows[-1]['device_index']=1
    elif fault=='wrong-kernel':
        rows[1]['linked_kernels'][0]['name']='unrelated_fill';rows[-2]['name']='unrelated_fill'
    elif fault=='mixed':rows[3]['name']=events.FAMILIES['math'][1]
    elif fault=='wrong-phase':rows[3]['name']=events.FAMILIES['efficient'][0]
    elif fault=='bad-parent':rows[3]['parent']=1
    elif fault=='candidate-not-math':arm='math'
    else:rows=trace_fixture('cpu-flash','cpu')
    with pytest.raises(ValueError):events.verify(rows,arm=arm,device='cuda')


def test_unrecognized_default_backend_refuses_without_forcing_fallback():
    rows=trace_fixture('efficient');rows[1]['name']='aten::_scaled_dot_product_cudnn_attention'
    with pytest.raises(ValueError):events.verify(rows,arm='original',device='cuda')
