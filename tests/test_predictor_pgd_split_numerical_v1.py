"""New source/metadata/opaque-callback integration checks; NO scientific code."""
import ast
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import CodeType, SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_numerical_v1 as source
from models import predictor_pgd_split_store_v1 as typed

ROOT=Path(__file__).resolve().parents[1]
FILES={'worker':'predictor_pgd_target_worker_v1','records':'predictor_pgd_target_records_v1',
       'progress':'post_gradient_progress_v1','writer':'predictor_forward_preserving_records_v1'}
INPUTS={k:(ROOT/'models'/(name+'.py')).read_bytes() for k,name in FILES.items()}
BASE=(ROOT/'configs/predictor_pgd_target_v1.json').read_bytes()
DESIGN=(ROOT/'configs/predictor_pgd_split_profile_v1.json').read_bytes()
SHA='4'*64


@pytest.fixture(scope='module')
def built():return source.build(INPUTS)


def progress_namespace(built):
    ns={'__name__':'EXPLICIT_LOCAL_PROGRESS_V2_METADATA_ONLY'}
    exec(compile(built[0]['progress'],'<NEW-METADATA-JOURNAL-ONLY>','exec'),ns)
    return NS(**ns)


def syntax(tree):return ast.dump(tree,include_attributes=False)


def functions(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}


def test_frozen_scientific_guards_and_math_are_unchanged_by_source_integration(built):
    modules,audit=built
    assert audit['worker_AST_inverse_exact_before_explicit_callback_filename_and_entry_renames']
    assert audit['worker_reversible_edits']==12 and audit['save_progress_calls_renamed']==6
    original_records=functions(ast.parse(INPUTS['records']));adapted_records=functions(modules['records'])
    for name in ('identity','check_scope','arrays','compare'):
        assert syntax(original_records[name])==syntax(adapted_records[name])
    old_run=functions(ast.parse(INPUTS['worker']))['run'];new_run=functions(modules['numerical'])['unbound_numerical_worker']
    for name in ('common','idle','no_grad','context'):
        old=next(n for n in ast.walk(old_run) if isinstance(n,ast.FunctionDef) and n.name==name)
        new=copy.deepcopy(next(n for n in ast.walk(new_run) if isinstance(n,ast.FunctionDef) and n.name==name))
        for n in ast.walk(new):
            if isinstance(n,ast.Name) and n.id=='save_progress':n.id='progress'
        assert syntax(old)==syntax(new),name
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('profiler','profile_summary','validate_backend') for n in ast.walk(new_run))
    assert not {'profile_summary','validate_backend'}&set(adapted_records)
    # Entire sources compile only; the model, decoder, verifier math and run body
    # are never imported, bound, called or executed by this cold check.
    for role,t in modules.items():compile(t,'<UNBOUND-COLD-'+role+'>','exec')
    assert not audit['scientific_qualified'] and not audit['target_runner_available']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('which',list(FILES))
def test_changed_frozen_source_refuses_before_any_namespace_is_bound(which):
    data=dict(INPUTS);data[which]+=b'\n'
    with pytest.raises(ValueError,match='Frozen input changed'):source.build(data)


class Opaque:
    """Tiny text token only; never a tensor, array or mathematical payload."""
    def __init__(self,engine,label):self.engine=engine;self.label=label;self._version=0;self.grad=None
    def view(self,*shape):return Opaque(self.engine,self.label+'-VIEW')
    def sum(self):return Opaque(self.engine,self.label+'-SUM-STUB')
    def register_hook(self,callback):return self.engine.handle('prediction',self.engine.incoming,callback)


class Handle:
    def __init__(self,engine,label,holder,callback):
        self.engine,self.label,self.holder,self.callback=engine,label,holder,callback;holder.append(callback)
    def remove(self):
        self.engine.events.append('remove-'+self.label)
        if self.engine.fault=='hook_removal' and self.label=='prediction':raise ValueError('SIMULATED removal refusal')
        self.holder.remove(self.callback)


class Route:
    def __init__(self,engine,candidate):
        self.engine=engine;self.candidate=candidate;self.mode='candidate' if candidate else 'original';self.phase=None
        self.rows=[{'origin':'OPAQUE-CALLBACK-SIMULATION','row':i} for i in range(3)]
        self.scopes=[{'origin':'OPAQUE-CALLBACK-SIMULATION'}];self.refusals=[]
        self.native=self;self.prototype=NS(calls=[],scope=None);self.closed=False
    def __enter__(self):self.engine.events.append('native-enter');return self
    def __exit__(self,*exc):
        self.engine.events.append('native_exit');self.closed=True
        if self.engine.fault=='native_exit':raise ValueError('SIMULATED native restoration refusal')
        return False
    def loss(self,value,*unused):return self.engine.emit(value)
    def gradient(self,*unused):
        self.engine.events.append('GRADIENT-STUB');self.engine.gradient_calls+=1
        for callback in list(self.engine.incoming):callback(Opaque(self.engine,'COTANGENT-STUB'))
        if self.engine.fault=='input_version':self.engine.value._version+=1
        return Opaque(self.engine,'INPUT-GRADIENT-STUB')
    def prepare_backward(self,*unused):return NS(run=lambda:(self.gradient(),))


class Harness:
    def __init__(self,built,root,index,*,fault=None):
        root.chmod(0o700);self.root=root;self.index=index;self.fault=fault;self.events=[]
        self.forward=[];self.incoming=[];self.gradient_calls=0;self.boundaries=0
        (root/'results').mkdir(mode=0o700);(root/'results/numerical').mkdir(mode=0o700)
        self.folder=root/'results/numerical';self.value=Opaque(self,'PIXELS-STUB')
        self.store=typed.Store(typed.Inventory(root,base_bytes=BASE,design_bytes=DESIGN))
        self.progress=progress_namespace(built);self.state={'gradient_objectives_started':index,'baseline_sha256':'5'*64,'mask_context_lengths':[]}
        engine=self
        class Predict:
            def register_forward_hook(self,callback):return engine.handle('forward',engine.forward,callback)
        model=NS(predictor=Predict())
        ns={'__name__':'EXPLICIT_OPAQUE_CAPTURE_SIMULATION','ExitStack':ExitStack,'sys':sys,'os':os,
            'folder':self.folder,'args':NS(control_sha256=SHA),'state':self.state,'progress':self.progress,
            'tensor_signature':lambda t:{'opaque':t.label},'model':model,'targets':Opaque(self,'TARGETS-STUB'),
            'packed':Opaque(self,'MASKS-STUB'),'banks':object(),'active':[None],'save_progress':lambda:None,
            'common':lambda:None,'torch':NS(is_grad_enabled=lambda:True,_C=NS(_current_graph_task_id=lambda:-1))}
        def idle():
            assert ns['active'][0] is None
            if self.fault=='restoration' and self.gradient_calls:raise ValueError('SIMULATED model restoration refusal')
        ns['idle']=idle
        ns['native']=NS(PredictorRoute=lambda *a,**kw:Route(self,False))
        ns['target']=NS(TargetRoute=lambda *a,**kw:Route(self,True))
        ns['native_objective']=lambda *a,**kw:self.emit
        class Boundary:
            def __enter__(self):engine.events.append('boundary-enter');return self
            def __exit__(self,*exc):
                engine.events.append('boundary_exit')
                if engine.fault=='boundary_exit':raise ValueError('SIMULATED observer restoration refusal')
                return False
        def attach(route,**kw):
            assert index==0 and not route.candidate and route.phase=='EOT8-before'
            assert kw=={'source_value':self.value,'source_objective_index':0,'control_sha256':SHA,'folder':root/'results/boundary'}
            self.boundaries+=1;return Boundary()
        ns['boundary']=NS(attach=attach)
        self.ns=ns
        self.records=self.record_namespace(built);ns['records']=self.records
        if fault=='journal-open':
            def refuse(*args,**kw):raise OSError('SIMULATED journal open refusal')
            self.progress.Journal=refuse
        if fault=='journal-gate':
            original=self.progress.Journal.begin
            def refuse(journal,phase):
                if phase=='gradient_checks':raise OSError('SIMULATED fsync refusal before gate')
                return original(journal,phase)
            self.progress.Journal.begin=refuse
        if fault=='journal-publication-end':
            original=self.progress.Journal.end
            def refuse(journal,phase,error=None):
                if phase=='record_publish' and error is None:raise OSError('SIMULATED fsync refusal after marker')
                return original(journal,phase,error)
            self.progress.Journal.end=refuse
        exec(compile(built[0]['capture'],'<NEW-CAPTURE-OPAQUE-CALLBACKS-ONLY>','exec'),ns)
        self.capture=ns['Capture'](self.value,index//2,bool(index%2),NS(arrays={}) if index%2 else None)

    def handle(self,label,holder,callback):return Handle(self,label,holder,callback)
    def emit(self,value):
        for i in range(15 if self.fault=='missing-return' else 16):
            for callback in list(self.forward):callback(None,(),Opaque(self,'PREDICTION-STUB-'+str(i)))
        return Opaque(self,'LOSS-STUB')
    def relative(self,folder,name):return (Path(folder)/name).relative_to(self.root).as_posix()
    def record_namespace(self,built):
        engine=self
        def atomic(folder,name,chunks,cap):
            if engine.fault=='marker-publish' and name=='numerical-complete.json':raise OSError('SIMULATED marker write refusal')
            data=b''.join(chunks);assert len(data)<=cap
            return engine.store.write_bytes(engine.relative(folder,name),data)
        atomic_ns=NS(_json=lambda v:(json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode(),_atomic=atomic)
        ns={'__name__':'EXPLICIT_NUMERICAL_RECORD_METADATA_SIMULATION','Path':Path,'json':json,
            'atomic':atomic_ns,'sha_file':lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            'frozen':NS(atomic=atomic_ns),'progress':self.progress}
        trees=built[0]['records'].body
        chosen=[n for n in trees if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KIND','OBJECTIVE','ALGORITHM','BOUNDS','LIMITS')
                or isinstance(n,ast.FunctionDef) and n.name in ('identity','save_json','write_final')]
        exec(compile(ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[])),'<METADATA-WRITER-SIMULATION>','exec'),ns)
        def save_tensor(folder,state,name,token):
            state['evidence_origin']='OPAQUE-CALLBACK-SIMULATION'
            raw=('OPAQUE SIMULATION: '+token.label+'\n').encode()
            item=engine.store.write_bytes(engine.relative(folder,name+'.npy'),raw)
            state['artifacts'][name]=item
            return NS(any=lambda:engine.fault!='zero-gradient')
        def compare(actual,reference):
            engine.events.append('COMPARE-STUB')
            if engine.fault=='compare':raise ValueError('SIMULATED unchanged numerical gate rejection')
            return {'origin':'OPAQUE-CALLBACK-SIMULATION'}
        def scope(*args):
            engine.events.append('NATIVE-PROOF-STUB')
            if engine.fault=='proof':raise ValueError('SIMULATED native proof rejection')
            return {'origin':'OPAQUE-CALLBACK-SIMULATION'}
        ns.update(save_tensor=save_tensor,compare=compare,check_scope=scope)
        return NS(**ns)
    def evaluate(self):
        with self.capture:
            if self.index%2:self.capture.note_consider(self.capture.value,self.capture.losses)
            self.capture.gradient()
    def journal(self):return self.capture.folder/'post-gradient-progress-v2.jsonl'


@pytest.mark.parametrize('index',range(8))
def test_all_eight_objective_positions_integrate_new_writer_journal_and_typed_inventory(built,tmp_path,index):
    h=Harness(built,tmp_path,index);h.evaluate()
    assert h.gradient_calls==1 and h.boundaries==(index==0)
    assert not h.forward and not h.incoming and h.capture.closed and h.capture.progress_journal.closed
    assert h.events.index('native_exit')>h.events.index('GRADIENT-STUB')
    if index==0:assert h.events.index('boundary_exit')<h.events.index('native_exit')
    read=h.progress.read_progress(h.journal())
    assert h.progress.verify_completion(read,control_sha256=SHA,objective_index=index)=={'progress_gates_returned':True,'scientific_qualified':False}
    assert read['identity']['provisional'] is (index<2) and 'profiled' not in read['identity']
    assert len(read['rows'])<=64 and all(r['scientific_qualified'] is False for r in read['rows'])
    f=h.capture.folder
    assert (f/'numerical-complete.json').is_file() and not (f/'numerical-failed.json').exists()
    state=json.loads((f/'numerical.json').read_bytes())
    assert state['kind']==source.OBJECTIVE_KIND and state['evidence_origin']=='OPAQUE-CALLBACK-SIMULATION'
    assert len(state['artifacts'])==37 and state['complete'] and not state['research_admitted']
    for n in h.progress.SNAPSHOT_NAMES:assert (f/n).exists() is (index<2)
    report=h.store.check();assert report['scientific_qualified'] is False
    assert all('worker.json' not in k and not k.endswith(('success.json','backend.json')) for k in report['files'])
    with pytest.raises(ValueError):h.capture.__enter__()
    assert h.gradient_calls==1


@pytest.mark.parametrize('fault',['journal-open','journal-gate','missing-return','zero-gradient','compare',
 'hook_removal','boundary_exit','native_exit','restoration','input_version','proof','marker-publish','journal-publication-end'])
def test_numerical_failure_stops_success_or_receipt_admission_and_preserves_evidence(built,tmp_path,fault):
    index=1 if fault=='compare' else 0
    h=Harness(built,tmp_path,index,fault=fault)
    with pytest.raises((ValueError,OSError)) as caught:h.evaluate()
    parts=[];error=caught.value;seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error));parts.append(str(error));error=error.__context__
    expected={'journal-open':'SIMULATED journal open','journal-gate':'SIMULATED fsync refusal before gate',
        'missing-return':'All actual returns','zero-gradient':'All actual returns','compare':'SIMULATED unchanged numerical',
        'hook_removal':'SIMULATED removal','boundary_exit':'SIMULATED observer restoration','native_exit':'SIMULATED native restoration',
        'restoration':'SIMULATED model restoration','input_version':'Input/version changed','proof':'SIMULATED native proof',
        'marker-publish':'SIMULATED marker write','journal-publication-end':'SIMULATED fsync refusal after marker'}[fault]
    assert expected in '\n'.join(parts),parts
    assert h.gradient_calls==(0 if fault=='journal-open' else 1)
    assert h.capture.closed and (h.capture.progress_journal is None or h.capture.progress_journal.closed)
    assert h.gradient_calls<2
    f=h.capture.folder
    if h.journal().exists():
        read=h.progress.read_progress(h.journal())
        with pytest.raises(ValueError):h.progress.verify_completion(read,control_sha256=SHA,objective_index=index)
    else:assert fault=='journal-open'
    if fault!='journal-publication-end':assert not (f/'numerical-complete.json').exists()
    if fault=='journal-publication-end':
        assert (f/'numerical-complete.json').exists()  # Marker alone is deliberately insufficient.
        adapted=functions(built[0]['records'])['load_objective']
        calls=[n.func.attr for n in ast.walk(adapted) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        assert 'verify_completion' in calls and 'read_progress' in calls
    if fault in ('journal-gate','hook_removal','boundary_exit'):
        assert 'native_exit' in h.events
    assert any(f.iterdir())


@pytest.mark.parametrize('provisional,index',[(False,0),(True,2),(True,8),(True,True),(True,'0')])
def test_new_progress_schema_rejects_wrong_provisional_or_index_binding(built,tmp_path,provisional,index):
    p=progress_namespace(built);tmp_path.chmod(0o700)
    with pytest.raises(ValueError):p.Journal(tmp_path,control_sha256=SHA,objective_index=index,provisional=provisional)
    assert not list(tmp_path.iterdir())


def test_progress_reader_refuses_other_control_and_retains_incomplete_tail(built,tmp_path):
    h=Harness(built,tmp_path,2);h.evaluate();p=h.progress
    read=p.read_progress(h.journal())
    with pytest.raises(ValueError):p.verify_completion(read,control_sha256='0'*64,objective_index=2)
    with h.journal().open('ab') as f:f.write(b'{')
    partial=p.read_progress(h.journal());assert partial['incomplete_tail_bytes']==1
    with pytest.raises(ValueError):p.verify_completion(partial,control_sha256=SHA,objective_index=2)


@pytest.mark.parametrize('bad',['results/numerical/worker.json','results/numerical/objective-0/worker.json',
 'results/numerical/objective-0/backend.json','results/components/native/warmup.npy','results/boundary/reference.pt',
 '../results/plan.json','results//plan.json','/results/plan.json','results/plan.json.partial.partial'])
def test_inventory_has_no_legacy_unknown_or_noncanonical_path_escape(tmp_path,bad):
    tmp_path.chmod(0o700);writer=typed.Store(typed.Inventory(tmp_path,base_bytes=BASE,design_bytes=DESIGN))
    with pytest.raises(ValueError):writer.write_bytes(bad,b'{}\n')
    assert not list(tmp_path.iterdir())


def test_store_preserves_immutable_files_and_existing_partials(tmp_path):
    tmp_path.chmod(0o700);inv=typed.Inventory(tmp_path,base_bytes=BASE,design_bytes=DESIGN);s=typed.Store(inv)
    name='results/plan.json';s.write_bytes(name,b'{}\n')
    assert (tmp_path/name).read_bytes()==b'{}\n'
    with pytest.raises(FileExistsError):s.write_bytes(name,b'changed')
    assert (tmp_path/name).read_bytes()==b'{}\n' and s.poisoned
    with pytest.raises(ValueError):s.write_bytes('results/run-status.json',b'{}')


@pytest.mark.parametrize('fault',['symlink','hardlink','mode','unknown-directory','unknown-file','oversize','stale-partial'])
def test_result_inventory_refuses_unsafe_or_overbound_namespace(tmp_path,fault):
    tmp_path.chmod(0o700);inv=typed.Inventory(tmp_path,base_bytes=BASE,design_bytes=DESIGN);writer=typed.Store(inv)
    writer.write_bytes('results/run-status.json',b'{}\n');p=tmp_path/'results/run-status.json'
    if fault=='symlink':p.unlink();p.symlink_to('missing')
    elif fault=='hardlink':os.link(p,tmp_path/'results/plan.json')
    elif fault=='mode':p.chmod(0o644)
    elif fault=='unknown-directory':(tmp_path/'unknown').mkdir(mode=0o700)
    elif fault=='unknown-file':q=tmp_path/'results/unknown.json';q.write_bytes(b'{}');q.chmod(0o600)
    elif fault=='oversize':
        with p.open('wb') as f:f.truncate(256*1024+1)
    else:
        q=p.with_name(p.name+'.partial');q.write_bytes(b'PARTIAL');q.chmod(0o600)
        with pytest.raises(FileExistsError):writer.write_bytes('results/run-status.json',b'{}')
        assert q.read_bytes()==b'PARTIAL' and p.read_bytes()==b'{}\n';return
    with pytest.raises(ValueError):inv.scan()


def test_writer_requires_plain_bytes_and_design_pin_without_scientific_imports(tmp_path):
    tmp_path.chmod(0o700)
    with pytest.raises(ValueError):typed.Inventory(tmp_path,base_bytes=BASE,design_bytes=DESIGN+b'\n')
    writer=typed.Store(typed.Inventory(tmp_path,base_bytes=BASE,design_bytes=DESIGN))
    with pytest.raises(ValueError):writer.write_bytes('results/plan.json',object())
    assert not list(tmp_path.iterdir()) and 'torch' not in sys.modules and 'numpy' not in sys.modules


@pytest.mark.parametrize('mutation',['mode','replacement'])
def test_directory_identity_guards_remain_strict_after_legitimate_file_link_changes(tmp_path,monkeypatch,mutation):
    tmp_path.chmod(0o700);writer=typed.Store(typed.Inventory(tmp_path,base_bytes=BASE,design_bytes=DESIGN))
    writer.write_bytes('results/run-status.json',b'BEFORE')
    writer.write_bytes('results/run-status.json',b'AFTER')
    assert (tmp_path/'results/run-status.json').read_bytes()==b'AFTER'
    original=typed.os.write;changed=False
    def intercepted(fd,data):
        nonlocal changed
        n=original(fd,data)
        if not changed:
            changed=True
            parent=tmp_path/'results'
            if mutation=='mode':parent.chmod(0o755)
            else:
                parent.rename(tmp_path/'replaced-results');parent.mkdir(mode=0o700)
        return n
    monkeypatch.setattr(typed.os,'write',intercepted)
    with pytest.raises(ValueError):writer.write_bytes('results/plan.json',b'PRESERVE-PARTIAL')
    assert changed and writer.poisoned and not (tmp_path/'results/plan.json').exists()
