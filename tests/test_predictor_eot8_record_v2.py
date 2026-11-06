"""Full-volume metadata and real CPU integration; no target/GPU execution."""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from models import predictor_eot8_record_v2 as record
from models import predictor_eot8_owned_operator_v1 as frozen_op
from models import predictor_eot8_owned_client_v1 as frozen_client
from models.predictor_eot8_owned_receipt_v1 import verify_ownership
from models.predictor_eot8_capture_v1 import tensor_info

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'external/vjepa2'
OLD=ROOT/'runs/pretrained/diagnostic/predictor-eot8-10603-v1-20261105/cpu-four-worker'
INITIAL_SHA='989665a5a4f8ea90600214be5536b4600c37fc448108a5bd177b3416e5ae0338'
PHASES=['EOT8-before','draw-before','draw-profiled','draw-after','EOT8-after']


def capacity_fixture(n=2736,*,padding=False,index=0):
    """Explicit synthetic capacity data, never recovered or live CUDA evidence."""
    state={'kind':'predictor-clean-eot8-owned-worker-v1','execution_host':'synthetic-capacity-only',
           'local_tiny_analogue':False,'index':index,'replay':index//2,'mode':'original' if index%2==0 else 'predictor-math',
           'pid':12345+index,'control_sha256':'synthetic-capacity-not-a-grant','complete':False,'passed':False,
           'research_admitted':False,'performance_qualified':False,'synthetic_capacity_fixture':True,'phases':PHASES,
           'baseline':{'scope':'synthetic metadata only; no model, GPU, oracle or target receipts'},'artifacts':{}}
    scopes=[];rows=[]
    flags={'flash':False,'math':True,'mem_efficient':True,'cudnn':True}
    for phase_index,phase in enumerate(PHASES):
        scope={'index':phase_index,'phase':phase,'state':'complete','pid':state['pid'],'creator_thread':10000,
               'runner_thread':10000,'root_hook_thread':20000,'graph_id':100+phase_index,'call_threads':[20000],
               'loss_id':30000+phase_index,'input_id':40000+phase_index,'hook_removed':True}
        scopes.append(scope)
        for backward in (False,True):
            for family in range(16 if phase.startswith('EOT8') else 2):
                for role,count in [('encoder',24),('predictor',12)]:
                    for block in range(count):
                        heads,length,width=(16,656 if family%2==0 else 400,64) if role=='encoder' else (12,2048,32)
                        layout={'shape':[1,heads,length,width],'stride':[heads*length*width,width,heads*width,1],
                                'dtype':'torch.float32','device':'cuda:0','requires_grad':True}
                        rows.append({'index':len(rows),'owner':role+'.'+str(block),'role':role,'block':block,
                            'q':copy.deepcopy(layout),'k':copy.deepcopy(layout),'v':copy.deepcopy(layout),'entry_flags':flags,
                            'effective_flags':flags,'exit_flags':flags,'outcome':'returned','phase':phase,'during_backward':backward,
                            'thread':20000 if backward else 10000,'creator_thread':10000,'pid':state['pid'],
                            'graph_id':scope['graph_id'] if backward else -1,'backward_scope':phase_index if backward else None})
    assert len(rows)==2736
    while len(rows)<n:
        row=copy.deepcopy(rows[-1]);row['index']=len(rows);rows.append(row)
    rows=rows[:n]
    if padding:
        for row in rows:
            row['synthetic_capacity_padding']='';remaining=record.MAX_ROW_BYTES-len(record._json(row));assert remaining>=0
            row['synthetic_capacity_padding']='x'*remaining;assert len(record._json(row))==record.MAX_ROW_BYTES
    return state,SimpleNamespace(rows=rows,scopes=scopes,refusals=[],active=False)


def private(tmp_path,name='worker'):
    p=tmp_path/name;p.mkdir(mode=0o700);return p


def test_pretrained_volume_reproduces_old_bound_and_new_writer_preserves_every_row(tmp_path):
    state,route=capacity_fixture();folder=private(tmp_path)
    logical={**state,'call_rows':route.rows,'backward_scopes':route.scopes,'ownership_refusals':route.refusals}
    assert len(record._json(logical))>2**20
    with pytest.raises(ValueError,match='Operational JSON exceeds bound'):
        frozen_op.old.write_json(folder/'legacy.json',logical)
    assert not (folder/'legacy.json').exists()
    store=record.WorkerRecord(folder);compact=store.write(state,route=route,final=True)
    actual=record.load_worker(folder,require_passed=False)
    assert actual['call_rows']==route.rows and actual['backward_scopes']==route.scopes
    assert verify_ownership(actual)['owners']==36
    assert not any(k in compact for k in record.INLINE)
    assert (folder/record.HEADER).stat().st_size<record.MAX_HEADER_BYTES<2**20
    assert (folder/record.CALLS).stat().st_size>2**20
    assert frozen_op.old.POLICY['max_header_bytes']==2**20
    with pytest.raises(ValueError,match='Complete passing actual worker'):record.load_worker(folder)
    (folder/'capacity-evidence.json').write_text(json.dumps({'synthetic_only':True,'rows':2736,'old_JSON_bytes':len(record._json(logical)),
        'split_call_bytes':(folder/record.CALLS).stat().st_size,'small_worker_bytes':(folder/record.HEADER).stat().st_size}))


def test_4096_maximum_size_rows_four_serial_workers_and_one_atomic_copy_fit_metadata_budget(tmp_path):
    scratch=private(tmp_path,'scratch');(scratch/'results').mkdir(mode=0o700)
    for i in range(4):
        folder=private(scratch/'results',str(i));state,route=capacity_fixture(4096,padding=True,index=i)
        store=record.WorkerRecord(folder);store.write(state,route=route,final=True)
        view=record.load_worker(folder,require_passed=False)
        assert view['call_rows']==route.rows and (folder/record.CALLS).stat().st_size==8*2**20
    # Simulate the only active atomic snapshot beside four existing final files.
    current=scratch/'results/3';shutil.copyfile(current/record.CALLS,current/(record.CALLS+'.partial'))
    parent=tmp_path.stat();sample=frozen_op.storage(scratch,{'device':parent.st_dev,'inode':parent.st_ino},parent=tmp_path)
    bound=record.storage_projection()
    assert sample['categories']['metadata']<bound['existing_metadata_category_bytes']==64*2**20
    assert sample['allocated_bytes']<9*2**28 and bound['four_workers_plus_one_atomic_copy_bytes']<64*2**20
    (scratch/'capacity-budget.json').write_text(json.dumps({'synthetic_only':True,'workers':4,'rows_each':4096,
        'row_bytes_each':2048,'measured_storage':sample,'static_projection':bound}))


@pytest.mark.parametrize('fault',['rows','row_bytes','scopes','refusals','scope_bytes','header_bytes','inline'])
def test_write_limits_preserve_partial_and_never_publish_success(tmp_path,fault):
    state,route=capacity_fixture();folder=private(tmp_path);store=record.WorkerRecord(folder)
    if fault=='rows':route.rows=capacity_fixture(4097)[1].rows
    elif fault=='row_bytes':route.rows[0]['too_large']='x'*record.MAX_ROW_BYTES
    elif fault=='scopes':route.scopes.append(copy.deepcopy(route.scopes[-1]))
    elif fault=='refusals':route.refusals=[{'predicate':'fixture'}]*33
    elif fault=='scope_bytes':route.scopes[0]['too_large']='x'*record.MAX_SCOPE_BYTES
    elif fault=='header_bytes':state['too_large']='x'*record.MAX_HEADER_BYTES
    elif fault=='inline':state['call_rows']=route.rows
    with pytest.raises(ValueError) as error:store.write(state,route=route,final=True)
    store.failure(state,error.value)
    assert not (folder/record.COMPLETE).exists() and (folder/record.FAILURE).is_file()
    with pytest.raises((ValueError,FileNotFoundError)):record.load_worker(folder)
    before={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='no retry'):store.write(state,route=route,final=True)
    assert before=={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}


def test_previously_saved_row_and_file_tampering_refuse_without_repair(tmp_path):
    state,route=capacity_fixture();route.rows=route.rows[:10];folder=private(tmp_path);store=record.WorkerRecord(folder)
    store.write(state,route=route);saved=(folder/record.CALLS).read_bytes()
    route.rows[0]['thread']+=1
    with pytest.raises(ValueError,match='Previously saved call row'):store.write(state,route=route)
    assert (folder/record.CALLS).read_bytes()==saved and (folder/(record.CALLS+'.partial')).exists()
    assert not (folder/record.COMPLETE).exists()


@pytest.mark.parametrize('target',['directory','calls','dangling_marker'])
def test_symlinks_or_old_record_cannot_be_reused(tmp_path,target):
    folder=private(tmp_path);elsewhere=private(tmp_path,'elsewhere')
    if target=='directory':
        link=tmp_path/'link';link.symlink_to(folder)
        with pytest.raises(ValueError):record.WorkerRecord(link)
    else:
        (folder/(record.CALLS if target=='calls' else record.COMPLETE)).symlink_to(elsewhere/'missing')
        with pytest.raises(FileExistsError):record.WorkerRecord(folder)
    assert list(elsewhere.iterdir())==[]


@pytest.fixture(scope='module')
def four_workers(tmp_path_factory):
    folder=tmp_path_factory.mktemp('split-record-real-cpu')
    code='''import sys,os
os.umask(0o077)
from pathlib import Path
from types import SimpleNamespace
from models.predictor_eot8_record_cpu_v2 import run
b=Path(sys.argv[1]);i=int(sys.argv[2]);run(SimpleNamespace(output=b/str(i),index=i,tiny=True,device='cpu',upstream=Path(sys.argv[3]),initial=Path(sys.argv[4]),initial_sha=sys.argv[5],control_sha256='local-split-record-CPU-only',reference=b/str(i-1) if i in (1,3) else None,replay_reference=b/str(i-2) if i in (2,3) else None))'''
    assert frozen_op.old.digest(OLD/'initial.pt')==INITIAL_SHA
    for i in range(4):
        p=subprocess.run([sys.executable,'-B','-c',code,str(folder),str(i),str(UPSTREAM),str(OLD/'initial.pt'),INITIAL_SHA],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (folder/('worker-'+str(i)+'.log')).write_bytes(p.stdout)
        assert p.returncode==0,p.stdout.decode()[-5000:]
    return folder


def test_four_fresh_workers_use_external_evidence_preserve_40_actual_returns(four_workers):
    pids=[];returns=0
    for i in range(4):
        folder=four_workers/str(i);view=record.load_worker(folder,expected_control='local-split-record-CPU-only')
        small=json.loads((folder/record.HEADER).read_text());old=json.loads((OLD/str(i)/'worker.json').read_text());pids.append(view['pid'])
        assert view['complete'] and view['passed'] and view['adapter_restored'] and view['local_tiny_analogue']
        assert len(view['phases'])==5 and len(view['backward_scopes'])==5 and len(view['artifacts'])==10
        assert view['ownership_check']==verify_ownership(view) and not view['backend']['CUDA_qualified']
        assert not any(k in small for k in record.INLINE) and (folder/record.HEADER).stat().st_size<record.MAX_HEADER_BYTES
        assert (folder/record.COMPLETE).exists() and not (folder/record.FAILURE).exists()
        assert view['baseline']==old['baseline']
        for n,item in view['artifacts'].items():
            actual=torch.load(folder/item['file'],weights_only=True,map_location='cpu')
            expected=torch.load(OLD/str(i)/old['artifacts'][n]['file'],weights_only=True,map_location='cpu')
            assert torch.equal(actual,expected) and tensor_info(actual)==item['source'] and item['saved_before_numerical_gate'];returns+=1
    assert len(set(pids))==4 and returns==40


@pytest.mark.parametrize('fault',['calls','scopes','header','missing_marker','marker','dangling_failure','missing_calls','path','hardlink'])
def test_corrupt_or_missing_evidence_never_loads_as_reference(four_workers,tmp_path,fault):
    folder=tmp_path/'copy';shutil.copytree(four_workers/'0',folder)
    if fault in ('calls','scopes','header','marker'):
        name={'calls':record.CALLS,'scopes':record.SCOPES,'header':record.HEADER,'marker':record.COMPLETE}[fault]
        p=folder/name;data=p.read_bytes();p.write_bytes(data[:-1]+b'x')
    elif fault=='missing_marker':(folder/record.COMPLETE).unlink()
    elif fault=='dangling_failure':(folder/record.FAILURE).symlink_to(tmp_path/'missing')
    elif fault=='missing_calls':(folder/record.CALLS).unlink()
    elif fault=='path':
        p=folder/record.HEADER;data=json.loads(p.read_text());data['evidence']['calls']['file']='../outside';p.write_text(json.dumps(data))
    elif fault=='hardlink':os.link(folder/record.CALLS,tmp_path/'external-link')
    with pytest.raises((ValueError,FileNotFoundError)):record.load_worker(folder)


def test_same_connection_collection_preserves_external_files_hash_mode_and_exact_mtime(four_workers,tmp_path):
    source=four_workers/'0';response=io.BytesIO();frozen_op.stream(source,'local-copy-test',response)
    payload=tmp_path/'response.bin';payload.write_bytes(response.getvalue());copy=tmp_path/'saved'
    frozen_client.unpack(payload,copy,{},'local-copy-test',collect=True)
    actual=record.load_worker(copy);original=record.load_worker(source)
    assert actual==original
    for name in (record.CALLS,record.SCOPES,record.HEADER,record.COMPLETE):
        a=source/name;b=copy/name
        assert a.read_bytes()==b.read_bytes() and a.stat().st_mtime_ns==b.stat().st_mtime_ns
        assert a.stat().st_mode&0o777==b.stat().st_mode&0o777==0o600


def run_args(folder,index=0,reference=None):
    return SimpleNamespace(output=folder,index=index,tiny=True,device='cpu',upstream=UPSTREAM,initial=OLD/'initial.pt',
        initial_sha=INITIAL_SHA,reference=reference,replay_reference=None,control_sha256='local-split-record-CPU-only')


def test_real_backward_failure_retains_actual_loss_failed_scope_and_removed_hook(tmp_path,monkeypatch):
    from models.predictor_eot8_record_cpu_v2 import run
    from models.predictor_eot8_owned_v1 import PredictorRoute
    original=PredictorRoute._admit
    def refuse(self):
        if self._task is not None and self._task.state=='running':self._fail('injected-split-record-refusal')
        return original(self)
    monkeypatch.setattr(PredictorRoute,'_admit',refuse);folder=tmp_path/'failure'
    with pytest.raises(ValueError,match='injected-split-record-refusal'):run(run_args(folder))
    view=record.load_worker(folder,require_passed=False)
    assert not view['complete'] and not view['passed'] and view['adapter_restored']
    assert set(view['artifacts'])=={'EOT8-before-loss'} and not (folder/'EOT8-before-input.pt').exists()
    scope=view['backward_scopes'][0];assert scope['state']=='failed' and scope['hook_removed'] and scope['graph_id']>=0
    assert view['ownership_refusals'][0]['current_graph_id']==scope['graph_id']
    assert (folder/record.FAILURE).exists() and not (folder/record.COMPLETE).exists()


def test_writer_failure_cannot_mask_original_scientific_refusal(tmp_path,monkeypatch):
    from models.predictor_eot8_record_cpu_v2 import run
    from models.predictor_eot8_owned_v1 import PredictorRoute
    admit=PredictorRoute._admit;write=record.WorkerRecord.write
    def refuse(self):
        if self._task is not None and self._task.state=='running':self._fail('primary-scientific-refusal')
        return admit(self)
    def fail_final(self,state,*,final=False,**kwargs):
        if final:raise OSError('secondary-record-write-failure')
        return write(self,state,final=final,**kwargs)
    monkeypatch.setattr(PredictorRoute,'_admit',refuse);monkeypatch.setattr(record.WorkerRecord,'write',fail_final)
    folder=tmp_path/'failure'
    with pytest.raises(ValueError,match='primary-scientific-refusal'):run(run_args(folder))
    marker=json.loads((folder/record.FAILURE).read_text())
    assert marker['original_error_type']=='ValueError' and 'primary-scientific-refusal' in marker['original_error']
    assert marker['error_type']=='OSError' and not marker['passed']
    with pytest.raises(ValueError):record.load_worker(folder)


def test_final_commit_failure_stops_before_any_passing_reference(tmp_path,monkeypatch):
    from models.predictor_eot8_record_cpu_v2 import run
    replace=record.os.replace
    def refuse_marker(src,dst):
        if Path(dst).name==record.COMPLETE:raise OSError('completion-publication-failed')
        return replace(src,dst)
    monkeypatch.setattr(record.os,'replace',refuse_marker);folder=tmp_path/'failure'
    with pytest.raises(OSError,match='completion-publication-failed'):run(run_args(folder))
    # Even a success-looking header is inadmissible until the final marker exists.
    header=json.loads((folder/record.HEADER).read_text());assert header['passed'] and header['complete']
    assert (folder/record.FAILURE).exists() and not (folder/record.COMPLETE).exists()
    with pytest.raises(ValueError,match='Complete passing actual worker'):record.load_worker(folder)


def test_candidate_refuses_missing_completion_before_model_setup(four_workers,tmp_path,monkeypatch):
    import models.execution as execution
    from models.predictor_eot8_record_cpu_v2 import run
    reference=tmp_path/'reference';shutil.copytree(four_workers/'0',reference);(reference/record.COMPLETE).unlink()
    monkeypatch.setattr(execution,'configure_execution',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Model setup reached')))
    with pytest.raises(FileNotFoundError):run(run_args(tmp_path/'candidate',index=1,reference=reference))
    view=record.load_worker(tmp_path/'candidate',require_passed=False);assert not view['passed'] and not view['artifacts']


def test_control_domain_or_synthetic_evidence_cannot_be_a_passing_reference(four_workers,tmp_path):
    with pytest.raises(ValueError,match='control binding'):record.load_worker(four_workers/'0',expected_control='different')
    state,route=capacity_fixture();state.update(complete=True,passed=True,adapter_restored=True)
    state['ownership_check']=verify_ownership({**state,'call_rows':route.rows,'backward_scopes':route.scopes,'ownership_refusals':[]})
    store=record.WorkerRecord(private(tmp_path))
    with pytest.raises(ValueError,match='actual worker evidence'):store.write(state,route=route,final=True)


def test_CPU_driver_refuses_live_or_CUDA_before_scientific_import(tmp_path):
    program='''import sys
from types import SimpleNamespace
from models.predictor_eot8_record_cpu_v2 import run
def deny(event,args):
 if event=='import' and args[0].split('.')[0] in ('torch','numpy'):raise AssertionError('Scientific import before refusal')
sys.addaudithook(deny)
try:run(SimpleNamespace(tiny=False,device='cuda',output=sys.argv[1]))
except ValueError as e:assert 'Only local CPU analogue' in str(e)
else:raise AssertionError('Live execution reached')
'''
    p=subprocess.run([sys.executable,'-B','-c',program,str(tmp_path/'none')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    assert p.returncode==0,p.stderr.decode();assert not (tmp_path/'none').exists()


def test_long_unicode_errors_fit_failure_marker_without_success(tmp_path):
    state,route=capacity_fixture();state.update(error_type='ValueError',error='😀'*10000)
    folder=private(tmp_path);store=record.WorkerRecord(folder);store.write(state,route=route)
    store.failure(state,ValueError('😀'*10000))
    assert (folder/record.FAILURE).stat().st_size<record.MAX_MARKER_BYTES
    marker=json.loads((folder/record.FAILURE).read_text())
    assert marker['error_type']=='ValueError' and marker['error'] and not marker['passed']
    with pytest.raises(ValueError):record.load_worker(folder)


def test_full_8MiB_call_file_collection_roundtrip_and_reference_refusal(tmp_path):
    state,route=capacity_fixture(4096,padding=True);source=private(tmp_path)
    record.WorkerRecord(source).write(state,route=route,final=True)
    response=io.BytesIO();frozen_op.stream(source,'full-capacity-only',response)
    payload=tmp_path/'full-response.bin';payload.write_bytes(response.getvalue());copy=tmp_path/'collected'
    frozen_client.unpack(payload,copy,{},'full-capacity-only',collect=True)
    assert (copy/record.CALLS).stat().st_size==8*2**20
    assert record.load_worker(copy,require_passed=False)['call_rows']==route.rows
    with pytest.raises(ValueError,match='Complete passing actual worker'):record.load_worker(copy)


def test_legacy_complete_reference_cannot_substitute_for_split_receipts(tmp_path):
    folder=private(tmp_path)
    old=json.loads((OLD/'0/worker.json').read_text());assert old['complete'] and old['passed']
    (folder/record.HEADER).write_text(json.dumps(old))
    with pytest.raises(ValueError,match='legacy/inline'):record.load_worker(folder)
