"""New stdlib callback/source simulations only: no tensors, models or profilers."""
import ast
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import pytest
from models import post_gradient_progress_v1 as p
from models import predictor_pgd_progress_adapter_v1 as adapter

ROOT=Path(__file__).resolve().parents[1]
DESCRIPTOR={'file':'input.npy','bytes':12583040,'sha256':'1'*64,'shape':[1,3,16,256,256],
            'dtype':'float32','source_device':'cuda:0','saved_before_gate':True}


def journal(folder,profiled=False,index=0):
    folder.chmod(0o700)
    return p.Journal(folder,control_sha256='2'*64,objective_index=index,profiled=profiled)


def loaded(j):
    return p.read_progress(j.folder/p.FILE)


def succeed(j,omit=None):
    o=p.Observer(j);o.note_gradient(DESCRIPTOR)
    for phase in ['gradient_checks','partial_snapshot','hook_removal','native_exit','profiler_stop',
                  'restoration_checks','input_restore_checks','native_proof_validation',
                  'events_materialize','events_aggregate','backend_publish','backend_validate']:
        if phase==omit or not j.identity['profiled'] and phase in p.PROFILE_REQUIRED:continue
        o.call(phase,lambda:None)
    return o


def test_begin_is_fsynced_before_callback_and_return_object_is_opaque(tmp_path,monkeypatch):
    j=journal(tmp_path);o=p.Observer(j);seen=[];real=p.os.fsync
    def sync(fd):seen.append(fd);return real(fd)
    monkeypatch.setattr(p.os,'fsync',sync)
    class Opaque:
        def __iter__(self):raise AssertionError('Do not inspect a callback return')
        def __str__(self):raise AssertionError('Do not stringify a callback return')
    value=Opaque()
    def callback():
        assert seen[-1]==j.fd
        read=loaded(j)
        assert read['unreturned_observed_phase']=='gradient_checks'
        assert read['rows'][-1]['event']=='begin'
        return value
    assert o.call('gradient_checks',callback) is value
    assert loaded(j)['unreturned_observed_phase'] is None
    j.close()


def test_abrupt_simulated_process_exit_leaves_unreturned_interval(tmp_path):
    tmp_path.chmod(0o700)
    code="""import os,sys
from models.post_gradient_progress_v1 import Journal
j=Journal(sys.argv[1],control_sha256='2'*64,objective_index=0,profiled=True)
j.begin('profiler_stop')
os._exit(23)
"""
    r=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path)],cwd=ROOT,
                     capture_output=True,timeout=10,check=False)
    assert r.returncode==23,r.stderr
    result=p.read_progress(tmp_path/p.FILE)
    assert result['unreturned_observed_phase']=='profiler_stop'
    assert result['scientific_qualified'] is False and result['exact_stall_cause_established'] is False


def test_cleanup_matches_exitstack_LIFO_suppression_and_exception_identity(tmp_path):
    class BodyError(Exception):pass
    class ExitError(Exception):pass
    original=BodyError('SIMULATED');cleanup_error=ExitError('SIMULATED')
    def run(stack,observed):
        seen=[]
        class CM:
            def __init__(self,name,suppress=False):self.name=name;self.suppress=suppress
            def __enter__(self):return self
            def __exit__(self,*exc):seen.append((self.name,exc[1]));return self.suppress
        def remove():seen.append(('hook_removal',original));raise cleanup_error
        enter=stack.named_enter_context if observed else lambda _,cm:stack.enter_context(cm)
        callback=stack.named_callback if observed else lambda _,cb:stack.callback(cb)
        enter('profiler_stop',CM('profiler_stop',True));enter('native_exit',CM('native_exit'))
        callback('hook_removal',remove)
        suppressed=stack.__exit__(BodyError,original,None)
        return seen,suppressed
    expected=run(ExitStack(),False)
    j=journal(tmp_path,True)
    actual=run(p.ObservedExitStack(p.Observer(j)),True)
    assert actual==expected
    assert [a[0] for a in actual[0]]==['hook_removal','native_exit','profiler_stop']
    assert actual[0][1][1] is cleanup_error
    assert loaded(j)['rows'][2]['event']=='raised'
    j.close()


@pytest.mark.parametrize('failure',['begin','end'])
def test_cleanup_runs_once_and_remaining_unwind_survives_log_failure(tmp_path,monkeypatch,failure):
    j=journal(tmp_path,True);o=p.Observer(j);s=p.ObservedExitStack(o);seen=[]
    class CM:
        def __init__(self,name):self.name=name
        def __enter__(self):return self
        def __exit__(self,*exc):seen.append(self.name)
    s.named_enter_context('profiler_stop',CM('profiler_stop'))
    s.named_enter_context('native_exit',CM('native_exit'))
    s.named_callback('hook_removal',lambda:seen.append('hook_removal'))
    real=j._write
    def broken(phase,event,detail):
        if phase=='hook_removal' and event==('begin' if failure=='begin' else 'returned'):
            j.poisoned=True;raise OSError('SIMULATED disk failure')
        return real(phase,event,detail)
    monkeypatch.setattr(j,'_write',broken)
    with pytest.raises((OSError,p.ProgressError)):s.__exit__(None,None,None)
    assert seen==['hook_removal','native_exit','profiler_stop']
    with pytest.raises(p.ProgressError):o.cleanup('hook_removal',lambda:seen.append('RETRY'))
    assert 'RETRY' not in seen
    with pytest.raises(p.ProgressError):s.__exit__(None,None,None)
    j.close()


def test_failure_publication_runs_once_after_journal_poison(tmp_path):
    j=journal(tmp_path);o=p.Observer(j);j.poisoned=True;calls=[]
    with pytest.raises(p.ProgressError):o.publish(lambda:calls.append('FAILED'),success=False)
    with pytest.raises(p.ProgressError):o.publish(lambda:calls.append('RETRY'),success=False)
    assert calls==['FAILED'];j.close()


@pytest.mark.parametrize('missing',sorted(p.REQUIRED|p.PROFILE_REQUIRED))
def test_passing_publication_cannot_skip_any_required_gate(tmp_path,missing):
    j=journal(tmp_path,True);o=succeed(j,missing);seen=[]
    with pytest.raises(p.ProgressError):o.publish(lambda:seen.append('PASS'),success=True)
    assert not seen
    j.close()


def test_complete_diagnostic_journal_never_becomes_a_scientific_receipt(tmp_path):
    j=journal(tmp_path,True);o=succeed(j);calls=[]
    o.publish(lambda:calls.append(True),success=True)
    assert calls==[True]
    assert all(r['scientific_qualified'] is False for r in loaded(j)['rows'])
    with pytest.raises(p.ProgressError):o.publish(lambda:calls.append('RETRY'),success=True)
    assert not (tmp_path/'success.json').exists()
    j.close()
    with pytest.raises(FileExistsError):journal(tmp_path)


def test_creator_thread_and_foreign_process_refuse_before_callback(tmp_path,monkeypatch):
    j=journal(tmp_path);o=p.Observer(j);seen=[];errors=[]
    def foreign():
        try:o.cleanup('native_exit',lambda:seen.append('BAD'))
        except p.ProgressError as error:errors.append(error)
    t=threading.Thread(target=foreign);t.start();t.join(timeout=2)
    assert not t.is_alive() and len(errors)==1 and not seen
    realpid=p.os.getpid;monkeypatch.setattr(p.os,'getpid',lambda:realpid()+1)
    with pytest.raises(p.ProgressError):o.cleanup('native_exit',lambda:seen.append('BAD'))
    assert not seen;j.close()


@pytest.mark.parametrize('change',['rewrite','mode','link','replace','symlink','parent_mode','append'])
def test_inode_permission_link_and_committed_prefix_changes_refuse(tmp_path,change):
    j=journal(tmp_path);f=tmp_path/p.FILE;seen=[]
    if change=='rewrite':
        data=f.read_bytes();f.write_bytes(data.replace(b'observer',b'Xbserver'))
    elif change=='mode':f.chmod(0o644)
    elif change=='link':os.link(f,tmp_path/'other')
    elif change=='replace':f.unlink();f.write_bytes(b'');f.chmod(0o600)
    elif change=='symlink':f.rename(tmp_path/'old');f.symlink_to(tmp_path/'old')
    elif change=='parent_mode':tmp_path.chmod(0o755)
    elif change=='append':
        with f.open('ab') as stream:stream.write(b'foreign')
    with pytest.raises(p.ProgressError):p.Observer(j).call('gradient_checks',lambda:seen.append('BAD'))
    assert not seen;j.close()


def test_partial_write_preserved_and_no_callback_started(tmp_path,monkeypatch):
    j=journal(tmp_path);original=(tmp_path/p.FILE).read_bytes();real=p.os.write;seen=[];n=[0]
    def short(fd,data):
        n[0]+=1
        if n[0]==1:return real(fd,data[:17])
        raise OSError('SIMULATED short write')
    monkeypatch.setattr(p.os,'write',short)
    with pytest.raises(OSError):p.Observer(j).call('gradient_checks',lambda:seen.append('BAD'))
    assert not seen and j.poisoned
    after=(tmp_path/p.FILE).read_bytes();assert after.startswith(original) and len(after)==len(original)+17
    read=loaded(j);assert read['incomplete_tail_bytes']==17 and read['scientific_qualified'] is False
    assert (tmp_path/p.FILE).read_bytes()==after
    j.close()


def test_fsync_failure_preserves_bytes_and_blocks_callback(tmp_path,monkeypatch):
    j=journal(tmp_path);seen=[]
    def failure(fd):raise OSError('SIMULATED fsync failure')
    monkeypatch.setattr(p.os,'fsync',failure)
    with pytest.raises(OSError):p.Observer(j).call('gradient_checks',lambda:seen.append('BAD'))
    assert not seen and j.poisoned and (tmp_path/p.FILE).stat().st_size>len(j.prefix)
    j.close()


@pytest.mark.parametrize('case',['nonplain','row','clock','count','bytes'])
def test_metadata_and_append_ceiling_refusals_before_callback(tmp_path,monkeypatch,case):
    j=journal(tmp_path);seen=[]
    if case=='nonplain':
        class NoConvert:
            def __str__(self):raise AssertionError('Do not convert arbitrary metadata')
        with pytest.raises(p.ProgressError):j.observed_gradient(dict(DESCRIPTOR,extra=NoConvert()))
    elif case=='row':
        with pytest.raises(p.ProgressError):j.observed_gradient(dict(DESCRIPTOR,extra='x'*1000))
    else:
        if case=='clock':j.clock=lambda:j.last_clock-1
        elif case=='count':monkeypatch.setattr(p,'MAX_ROWS',1)
        else:monkeypatch.setattr(p,'MAX_BYTES',len(j.prefix)+1)
        with pytest.raises(p.ProgressError):p.Observer(j).call('gradient_checks',lambda:seen.append('BAD'))
    assert not seen and j.poisoned;j.close()


@pytest.mark.parametrize('malform',['duplicate_key','chain','identity','elapsed','duplicate_gradient','phase','success'])
def test_cold_reader_rejects_malformed_or_promoted_records(tmp_path,malform):
    j=journal(tmp_path);j.observed_gradient(DESCRIPTOR);j.close();f=tmp_path/p.FILE
    rows=[json.loads(b) for b in f.read_bytes().splitlines()]
    if malform=='duplicate_key':
        f.write_bytes(f.read_bytes().replace(b'"sequence":0',b'"sequence":0,"sequence":0'));bad=f.read_bytes()
    else:
        if malform=='chain':rows[-1]['previous_sha256']='9'*64
        elif malform=='identity':rows[0]['identity']['objective_index']=True
        elif malform=='elapsed':rows[-1]['elapsed_ns']=-1
        elif malform=='duplicate_gradient':rows.append(dict(rows[-1]))
        elif malform=='phase':rows[-1].update(phase='native_exit',event='returned',detail={})
        elif malform=='success':rows[-1]['scientific_qualified']=True
        head=p.ZERO
        for i,row in enumerate(rows):
            if malform!='chain':row['previous_sha256']=head
            row['sequence']=i;row.pop('row_sha256',None)
            row['row_sha256']=hashlib.sha256(p.encoded(row)).hexdigest();head=row['row_sha256']
        f.write_bytes(b''.join(p.encoded(r) for r in rows));bad=f.read_bytes()
    with pytest.raises((p.ProgressError,ValueError)):p.read_progress(f)
    assert f.read_bytes()==bad


def test_snapshot_is_exact_provisional_metadata_before_cleanup(tmp_path):
    j=journal(tmp_path,True);o=p.Observer(j);o.note_gradient(DESCRIPTOR);o.call('gradient_checks',lambda:None)
    calls=[{'index':0,'owner':'predictor.0','SIMULATION':True}]
    header=o.call('partial_snapshot',p.provisional_snapshot,j,calls=calls,scopes=[],refusals=[],vjps=[],borrowed=None)
    assert header['completion_or_restoration_admitted'] is False and header['scientific_qualified'] is False
    assert header['restoration_state']=='not_yet_checked'
    assert (tmp_path/p.SNAPSHOT_NAMES[0]).read_bytes()==p.encoded(calls[0])
    for name,d in header['files'].items():
        f=tmp_path/name;assert f.stat().st_mode&0o777==0o600
        assert len(f.read_bytes())==d['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256']
    assert not list(tmp_path.glob('*.npy')) and not (tmp_path/'success.json').exists()
    with pytest.raises(p.ProgressError):p.provisional_snapshot(j,calls=[],scopes=[],refusals=[],vjps=[],borrowed=None)
    j.close()


@pytest.mark.parametrize('case',['calls','VJP_count','row','scope','index','existing','object'])
def test_provisional_snapshot_caps_and_no_overwrite(tmp_path,case):
    j=journal(tmp_path,True,index=2 if case=='index' else 0);o=p.Observer(j)
    o.note_gradient(DESCRIPTOR);o.call('gradient_checks',lambda:None)
    kwargs=dict(calls=[],scopes=[],refusals=[],vjps=[],borrowed=None)
    if case=='calls':kwargs['calls']=[{}]*4097
    elif case=='VJP_count':kwargs['vjps']=[{}]*385
    elif case=='row':kwargs['calls']=[{'x':'x'*2048}]
    elif case=='scope':kwargs['scopes']=[{},{}]
    elif case=='existing':(tmp_path/p.SNAPSHOT_NAMES[2]).write_bytes(b'keep')
    elif case=='object':kwargs['borrowed']=object()
    with pytest.raises((p.ProgressError,FileExistsError)):
        o.call('partial_snapshot',p.provisional_snapshot,j,**kwargs)
    assert not (tmp_path/p.SNAPSHOT_NAMES[-1]).exists()
    if case=='existing':assert (tmp_path/p.SNAPSHOT_NAMES[2]).read_bytes()==b'keep'
    j.close()


def prototype():
    return adapter.prototype((ROOT/'models/predictor_pgd_target_worker_v1.py').read_bytes(),
                             (ROOT/'models/predictor_pgd_target_records_v1.py').read_bytes())


def test_actual_source_edits_are_reversible_and_profiler_predicates_exact():
    module,audit=prototype()
    assert audit['inverse_ast_exact'] and audit['profiler_body_reassembled_exact']
    assert audit['target_runner_available'] is False
    assert {n.name for n in module.body}=={'UnboundCapturePrototype','_materialize','_aggregate'}
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(module))
    with pytest.raises(ValueError):adapter.prototype(b'changed',b'changed')


def simulated_capture(folder,*,candidate=False,profiled=True,failure=None,monkeypatch=None):
    """Actual transformed gradient/exit AST with opaque returns and fake callbacks."""
    module,audit=prototype();seen=[];active=[None]
    def action(name,result=None):
        seen.append(name)
        if failure==name:raise RuntimeError('SIMULATED '+name)
        return result
    class Profile:
        def __enter__(self):return self
        def __exit__(self,*exc):action('profiler_stop')
        def events(self):
            action('events_materialize')
            return [NS(name='aten::_scaled_dot_product_efficient_attention',device_type='cpu',cpu_parent=None,kernels=[])]
    class Gradient:
        def any(self):return action('gradient_checks',True)
    gradient=Gradient()
    class Native:
        rows=[{'SIMULATION':True}];scopes=[];refusals=[]
        def __enter__(self):return self
        def __exit__(self,*exc):action('native_exit')
        def prepare_backward(self,*a):return NS(run=lambda:(action('native_gradient',gradient),))
    class Route(Native):
        native=Native();prototype=NS(calls=[{'SIMULATION':True}],scope=None)
        def gradient(self,*a):return action('native_gradient',gradient)
    def save_json(where,name,value,**kwargs):
        action('backend_publish');f=where/name;f.write_bytes(p.encoded(value));f.chmod(0o600)
    def final(where,record,**proof):
        action('record_publish')
        (where/'SIMULATED-final.json').write_bytes(p.encoded({'SIMULATION':True,'passed':record['passed']}))
        return record['passed']
    rec=NS(compare=lambda *a:action('comparison',{'SIMULATION':True}),
        check_scope=lambda *a:action('native_proof_validation',{'SIMULATION':True}),
        save_json=save_json,write_final=final,
        validate_backend=lambda *a:action('backend_validate'),
        frozen=NS(sha_file=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()))
    def idle():
        action('restoration_checks')
        if failure=='input_restore_checks':capture.value._version+=1
    ns={'progress':p,'records':rec,'active':active,'idle':idle,'tensor_signature':lambda v:{'SIMULATION':True},
        'frozen':NS(atomic=NS(_json=lambda x:p.encoded(x))),'math_is_finite':lambda x:True}
    exec(compile(module,'<UNBOUND-ONLY-CALLBACK-SIMULATION>','exec'),ns)
    aggregate=ns['_aggregate']
    def observed_aggregate(*a):action('events_aggregate');return aggregate(*a)
    ns['_aggregate']=observed_aggregate
    if failure=='partial_snapshot':
        def failed_snapshot(*a,**k):return action('partial_snapshot')
        monkeypatch.setattr(p,'provisional_snapshot',failed_snapshot)
    capture=ns['UnboundCapturePrototype'].__new__(ns['UnboundCapturePrototype'])
    capture.folder=folder;capture.step=0 if profiled else 1;capture.candidate=candidate
    capture.index=int(candidate) if profiled else 2+int(candidate)
    capture.progress_journal=journal(folder,profiled,index=capture.index)
    capture.observer=p.Observer(capture.progress_journal);capture.stack=p.ObservedExitStack(capture.observer)
    capture.profile=capture.stack.named_enter_context('profiler_stop',Profile()) if profiled else None
    capture.route=capture.stack.named_enter_context('native_exit',Route() if candidate else Native())
    capture.stack.named_callback('hook_removal',lambda:action('hook_removal'))
    capture.value=NS(_version=1,grad=None);capture.input_version=1;capture.input_hash={'SIMULATION':True}
    capture.used=capture.closed=False;capture.considered=candidate;capture.root=object();capture.reference=NS(arrays={}) if candidate else None
    capture.arrays={'fake-'+str(i):None for i in range(36)}
    capture.record={'artifacts':{},'events':['forward','consider'] if candidate else ['forward'],
                    'mode':'candidate' if candidate else 'original','complete':False,'passed':False}
    def save(name,value):
        assert name=='input' and value is gradient
        seen.append('saved_input_descriptor');capture.arrays[name]=value
        capture.record['artifacts'][name]=dict(DESCRIPTOR)
    capture.save=save;capture.admitted=lambda:None;active[0]=capture
    return capture,gradient,seen,ns


@pytest.mark.parametrize('candidate,profiled',[(False,True),(True,True),(False,False),(True,False)])
def test_actual_transformed_methods_keep_order_gates_and_opaque_returns(tmp_path,candidate,profiled):
    c,value,seen,ns=simulated_capture(tmp_path,candidate=candidate,profiled=profiled)
    assert c.gradient() is value
    if profiled:
        assert (tmp_path/p.SNAPSHOT_NAMES[-1]).exists()
        assert 'profiler_stop' not in seen
    c.__exit__(None,None,None)
    assert c.record['complete'] and c.record['passed'] and c.closed and ns['active']==[None]
    assert seen.index('hook_removal')<seen.index('native_exit')<seen.index('restoration_checks')
    if profiled:assert seen.index('native_exit')<seen.index('profiler_stop')<seen.index('restoration_checks')
    assert seen.index('native_proof_validation')<seen.index('record_publish')
    assert not (tmp_path/'success.json').exists()  # Only a labeled fake publisher ran.
    read=p.read_progress(tmp_path/p.FILE)
    assert read['scientific_qualified'] is False and read['rows'][-1]['phase']=='record_publish'


@pytest.mark.parametrize('failure',sorted(p.PHASES))
def test_each_actual_transformed_boundary_failure_prevents_passing_publication(tmp_path,monkeypatch,failure):
    c,value,seen,ns=simulated_capture(tmp_path,failure=failure,monkeypatch=monkeypatch)
    problem=None
    try:c.gradient()
    except BaseException as error:problem=error
    try:c.__exit__(None if problem is None else type(problem),problem,None)
    except BaseException as error:problem=error
    assert problem is not None
    if failure!='record_publish':
        assert c.record['passed'] is False and c.record['complete'] is False
        assert json.loads((tmp_path/'SIMULATED-final.json').read_bytes())['passed'] is False
    else:assert not (tmp_path/'SIMULATED-final.json').exists()
    assert {'hook_removal','native_exit','profiler_stop'}<=set(seen)
    read=p.read_progress(tmp_path/p.FILE)
    assert any(r['phase']==failure and r['event']=='raised' for r in read['rows'])
    assert read['scientific_qualified'] is False and ns['active']==[None]
    if failure in {'gradient_checks','partial_snapshot','hook_removal','native_exit','profiler_stop','restoration_checks','input_restore_checks','native_proof_validation'}:
        assert 'events_materialize' not in seen


def test_materialization_boundary_still_enforces_original_post_parse_event_cap():
    module,audit=prototype();functions=ast.Module(body=module.body[:2],type_ignores=[]);ns={}
    exec(compile(functions,'<SIMULATED-EVENTS-ONLY>','exec'),ns)
    for events in ([],[None]*1_000_001):
        with pytest.raises(ValueError,match='Bounded nonempty'):ns['_materialize'](NS(events=lambda:events))


def test_production_modules_import_only_standard_library_and_expose_no_runner():
    for name in ['post_gradient_progress_v1','predictor_pgd_progress_adapter_v1']:
        root=ast.parse((ROOT/'models'/(name+'.py')).read_text())
        modules={n.module for n in ast.walk(root) if isinstance(n,ast.ImportFrom)}
        modules|={a.name for n in ast.walk(root) if isinstance(n,ast.Import) for a in n.names}
        assert modules<=set(sys.stdlib_module_names)
        assert not any(isinstance(n,ast.FunctionDef) and n.name in {'main','run','worker'} for n in root.body)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
