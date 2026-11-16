"""Run only stdlib outer function definitions with entirely inert server bindings."""
import ast
import builtins
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_preclaim_refusal_v4 as fix

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=(ROOT/'models/predictor_pgd_split_outer_v3.py').read_bytes()
REFUSAL=ROOT/'runs/pretrained/diagnostic/predictor-pgd-startup-v3-20261116/handoff-parent/linux10603-unarmed/attended-session/REFUSAL.json'


def scenario(source,*,claimed=False,has_guard=False,error=None,idle=True):
    events=[];frames=[];ctx=NS(events=events,frames=frames,owner=None)
    query=json.loads(REFUSAL.read_bytes())['last_guard_refusal']
    class Alarm:
        def arm(self,soft,*,hard_deadline,phase):events.append(('alarm',phase,soft,hard_deadline))
        def check(self):pass
        def close(self):events.append('alarm-closed')
    class Root:
        def mkdir(self,mode):assert mode==0o700;events.append('EXPLICIT-FAKE-ROOT-CLAIM')
        def __str__(self):return '/var/tmp/EXPLICIT-INERT-NO-FILES'
    class Journal:
        def __init__(self):self.last_refusal=query if error is not None else None;self.closed=False
        def checkpoint(self,role):events.append(('journal',role))
        def close(self):self.closed=True;events.append('journal-closed')
    class Driver:
        guard=NS(layout=NS()) if has_guard else None
        outer_journal=Journal()
        def assert_idle(self):
            events.append('owned-idle-barrier')
            if not idle:raise RuntimeError('EXPLICIT FAKE owned child exit unconfirmed')
        def save(self,*a,**kw):events.append('fake-parent-save')
    class Owner:
        def __init__(self,*a,**kw):self.claimed=claimed;self.driver=Driver();self.state={};ctx.owner=self
        def run_work(self):
            events.append('FAKE-PARENT-PREPARATION')
            if error is not None:raise error
    class Collector:
        def __init__(self,guard,*a,**kw):
            events.append('collector-constructor')
            # Exact live secondary failure: original Collector.__init__ uses guard.layout.
            value=guard.layout
            self.window=NS(started=1001.0,deadline=1601.0)
        def prepare(self):events.append('collector-prepare')
        def stream(self,out):events.append('collector-stream')
    def importing(name,globals=None,locals=None,fromlist=(),level=0):
        assert name=='models' and fromlist==('predictor_pgd_split_collection_v1',)
        events.append('FAKE-COLLECTOR-IMPORT');return NS(predictor_pgd_split_collection_v1=NS(Collector=Collector))
    def send(out,frame,check,**kw):check();frames.append(copy.deepcopy(frame))
    message={'nonce':(b'0'*32).hex(),'client_elapsed':1.0,'control_text':'INERT-CONTROL','authorization_text':'INERT-GRANT','sources':{}}
    ctx.ns={'__builtins__':dict(vars(builtins),__import__=importing),'Alarms':Alarm,'OutputBudget':lambda out,check:out,
        'time':NS(monotonic=lambda:1000.0),'os':NS(urandom=lambda n:b'0'*n),'send_frame':send,'frame':lambda *a,**kw:message,
        'finite':lambda v:type(v) in (int,float),'validate_control':lambda *a:({'supervised_control_text':'INERT'},None,None,'f'*64),
        'validate_authorization':lambda *a:{},'decode_sources':lambda *a:{},'boot_namespace':lambda *a:NS(control=lambda *a:None),
        'preflight_parent':lambda:events.append('FAKE-HOST-CHECK'),'remote_root':lambda *a:Root(),
        'prepare_tree':lambda *a:(NS(OuterCoordinator=Owner),None,NS(),{},b'INERT'),
        'MiB':2**20,'FLAGS':{'scientific_qualified':False,'training_admitted':False}}
    exec(compile(source,'<INERT-OUTER-FUNCTION-ONLY>','exec'),ctx.ns)
    try:ctx.result=ctx.ns['remote_session'](NS(),NS(),code_sha256='f'*64)
    except BaseException as raised:ctx.raised=raised
    return ctx


def original_function():
    tree=ast.parse(ORIGINAL);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='remote_session')
    return ast.unparse(fn).encode()


def test_observed_GPU_refusal_is_reproduced_then_preserved_by_the_local_correction(tmp_path):
    error=ValueError('Another CUDA compute process is active; leave it alone and refuse.')
    old=scenario(original_function(),error=error)
    assert old.frames[-1]['message']=="'NoneType' object has no attribute 'layout'"
    assert 'collector-constructor' in old.events
    fixed,metadata=fix.build(ORIGINAL);new=scenario(fixed,error=error)
    assert new.frames[-1]['error_type']=='ValueError' and new.frames[-1]['message']==str(error)
    assert new.frames[-1]['last_guard_refusal']==old.frames[-1]['last_guard_refusal']
    assert 'collector-constructor' not in new.events and 'FAKE-COLLECTOR-IMPORT' not in new.events
    assert new.owner.driver.outer_journal.closed and new.result==1
    assert not metadata['authorized'] and not metadata['target_helper_ready']
    (tmp_path/'OBSERVED-REFUSAL-REGRESSION.json').write_text(json.dumps({'actual_saved_refusal_join':True,'old_message':old.frames[-1]['message'],
        'new_message':new.frames[-1]['message'],'events':new.events,'no_actual_server_or_process_or_proof':True}))


@pytest.mark.parametrize('claimed,has_guard',[(False,False),(False,True),(True,False)])
def test_every_incomplete_parent_failure_refuses_before_collection(claimed,has_guard):
    source,_=fix.build(ORIGINAL);error=OSError('INERT ORIGINAL PREPARATION ERROR')
    x=scenario(source,claimed=claimed,has_guard=has_guard,error=error)
    assert x.result==1 and x.frames[-1]['error_type']=='OSError' and x.frames[-1]['message']==str(error)
    assert 'collector-constructor' not in x.events and 'collector-stream' not in x.events
    assert x.events.index('owned-idle-barrier')<x.events.index('journal-closed')
    assert x.frames[-1]['new_work_started_after_error'] is False
    alarms=[n for n in x.events if isinstance(n,tuple) and n[0]=='alarm']
    assert [n[1] for n in alarms]==['handshake','setup','work','refusal']
    assert alarms[-1][2]==1002.0 and alarms[-1][3]==3999.0


def test_inconsistent_success_without_session_is_never_promoted_to_collection():
    source,_=fix.build(ORIGINAL);x=scenario(source)
    assert x.result==1 and x.frames[-1]['message']=='Parent session was not initialized; no collection permitted.'
    assert 'collector-constructor' not in x.events


@pytest.mark.parametrize('failed',[False,True])
def test_original_claimed_session_success_or_failure_keeps_existing_collection_route(failed):
    source,_=fix.build(ORIGINAL);error=RuntimeError('INERT CLAIMED-WORK FAILURE') if failed else None
    old=scenario(original_function(),claimed=True,has_guard=True,error=error)
    new=scenario(source,claimed=True,has_guard=True,error=error)
    assert new.result==old.result==(1 if failed else 0)
    assert new.events==old.events and new.frames==old.frames
    assert 'collector-stream' in new.events


def test_unconfirmed_owned_cleanup_cannot_enter_refusal_or_collection():
    source,_=fix.build(ORIGINAL);x=scenario(source,error=ValueError('INERT GPU'),idle=False)
    assert isinstance(x.raised,RuntimeError) and 'unconfirmed' in str(x.raised)
    assert not any(n in x.events for n in ('collector-constructor','collector-stream'))
    assert not any(f['kind']=='pgd-split-outer-refusal-v1' for f in x.frames)


def test_exact_inverse_retains_all_original_transport_clocks_cleanup_and_worker_logic():
    source,meta=fix.build(ORIGINAL);tree=ast.parse(source);fn=tree.body[0];block=next(n for n in fn.body if isinstance(n,ast.Try))
    matches=[n for n in block.body if isinstance(n,ast.If) and ast.dump(n,include_attributes=False)==ast.dump(ast.parse(fix.BRANCH).body[0],include_attributes=False)]
    assert len(matches)==1;block.body.remove(matches[0])
    original=ast.parse(original_function()).body[0]
    assert ast.dump(fn,include_attributes=False)==ast.dump(original,include_attributes=False)
    assert meta['uninitialized_parent_collection_barrier']==1
    with pytest.raises(ValueError):fix.build(ORIGINAL+b'UNREVIEWED')
