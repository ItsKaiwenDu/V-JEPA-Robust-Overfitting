"""New phase accounting controls; all clock/OS/scientific work ports are fake."""
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
from types import SimpleNamespace as NS

import pytest
from models import predictor_pgd_phase_accounting_v1 as phases
from models import predictor_pgd_phase_accounting_build_v1 as build
from models import predictor_pgd_workload_budget_v1 as budget
from models import predictor_pgd_workload_handoff_client_v4 as closed_client
from models import predictor_pgd_workload_runtime_records_v1  # Metadata definitions.

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/pgd-workload-phase-accounting-v1-20261123'
EVIDENCE=Path(os.environ['VJEPA_PHASE_EVIDENCE'])
spec=importlib.util.spec_from_file_location('_new_phase_engine',STAGE/'fake-phase-engine-v1.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
put=engine.put
ORIGIN=engine.ORIGIN


@pytest.fixture(autouse=True)
def forbid_actual_work(monkeypatch):
    engine.support.no_actual_work.__wrapped__(monkeypatch)  # Normal fixture, not a generator.
    def forbidden(*a,**kw):raise AssertionError('Real network/PIDFD/pipe/signal work forbidden.')
    for module,name in [(socket,'socket'),(os,'pidfd_open'),(os,'pipe2'),(signal,'pidfd_send_signal')]:
        if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)


@pytest.fixture
def area(request):
    path=EVIDENCE/request.node.name;path.mkdir(mode=0o700)
    put(path/'FAKE-ORIGIN.json',{'origin':ORIGIN,'actual_science_or_target_work':False,'actual_user_grant':False})
    return path


class Clock:
    def __init__(self):self.now=100.
    def __call__(self):return self.now
    def advance(self,seconds):self.now+=seconds


class Guard:
    def __init__(self,root):
        self.root=root
        self.binding={'root_identity':[root.lstat().st_dev,root.lstat().st_ino,os.getuid(),0o700]}
        self.layout=NS(classify=lambda name:{'category':'metadata','mutable':True,'max_bytes':phases.MAX_BYTES})
    def check_root(self):
        info=self.root.lstat()
        assert [info.st_dev,info.st_ino,info.st_uid,info.st_mode&0o777]==self.binding['root_identity']
    def open_root(self):return os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)


class DelaySink:
    def __init__(self,guard,clock):
        self.inner=phases.FileSink(guard,'phase-events.jsonl');self.clock=clock
    def append(self,raw):self.inner.append(raw);self.clock.advance(.002)
    def receipt(self):return self.inner.receipt()
    def close(self):return self.inner.close()


def ledger(area,*,clock=None):
    clock=clock or Clock()
    sink=DelaySink(Guard(area),clock)
    value=phases.PhaseLedger(sink,role='original-unit',control_sha256='a'*64,plan_sha256='b'*64,
        deadline=820.,scope_started=clock.now,clock=clock)
    return value,clock,sink


def publish_to(area):
    return lambda name,row:put(area/name,row)


def metadata_validate(raw,summary):
    return phases.validate_saved(raw,summary,role='original-unit',pid=os.getpid(),
        control_sha256='a'*64,plan_sha256='b'*64,deadline=820.)


@pytest.fixture(scope='module')
def boundary_golden():
    area=EVIDENCE/'boundary-golden';area.mkdir(mode=0o700)
    value,clock,sink=ledger(area)
    source=(STAGE/'cold-build-v2/workers/models/predictor_pgd_workload_target_v1.py').read_bytes()
    old_exit=engine.owned_exit(source,clock);calls=[]
    binding=NS(active=None,poisoned=False,steps=[],idle=lambda:None)
    class Journal:
        def put(self,name,row):
            calls.append('original-completion-journal');clock.advance(.4);put(area/(name+'.json'),row)
    binding.journal=Journal()
    class Timer:
        def __exit__(self,*error):calls.append('original-timer-exit');clock.advance(.5);return False
    class Owned:
        __exit__=old_exit
        def __init__(self):
            clock.advance(.1);self.binding=binding;self.exited=False;self.complete=True
            self.row={'index':0};self.index=0;self.timer=Timer();self.timer_entered=True
            self.route=NS(__exit__=lambda *a:calls.append('original-route-exit'))
            self.root=self.losses=self.value=object()
        def __enter__(self):
            self.started_at=1000000.+clock.now;clock.advance(.2);binding.active=self;return self
    made=[]
    def create():made.append(Owned());return made[0]
    with value.context(create,index=0) as original:
        assert original is made[0];clock.advance(.3)
    value.finalize(publish_to(area),'phase-summary.json')
    summary=json.loads((area/'phase-summary.json').read_bytes())
    raw=(area/'phase-events.jsonl').read_bytes()
    result={'origin':ORIGIN,'calls':calls,'legacy_context_seconds':binding.steps[0]['context_seconds_including_observations'],
        'outer_context_seconds':json.loads(raw.splitlines()[1])['seconds'],
        'new_scope_seconds':summary['scope_seconds'],'legacy_perf_epoch_offset':1000000.,
        'actual_model_gradient_or_timer':False}
    put(area/'BOUNDARY-RESULT.json',result)
    return area,raw,summary,result


def test_full_context_contains_original_completion_journal_and_timer_exit(boundary_golden):
    area,raw,summary,result=boundary_golden
    assert result['calls']==['original-route-exit','original-completion-journal','original-timer-exit']
    assert result['outer_context_seconds']==pytest.approx(1.502)
    assert result['legacy_context_seconds']==pytest.approx(.5)
    assert result['new_scope_seconds']==pytest.approx(1.504)
    assert metadata_validate(raw,summary)['performance_or_execution_authenticated'] is False


@pytest.mark.parametrize('pending',[False,True])
def test_accounting_failure_after_context_never_skips_original_exit(area,pending):
    value,clock,sink=ledger(area);calls=[]
    class Context:
        def __enter__(self):return self
        def __exit__(self,*error):calls.append('original-exit');return False
    original=RuntimeError('FAKE primary gradient error')
    def run():
        with value.context(Context,index=0):
            value.clock=lambda:float('nan')
            if pending:raise original
    if pending:
        with pytest.raises(RuntimeError) as caught:run()
        assert caught.value is original
        assert any('Phase accounting' in text for text in original.__notes__)
    else:
        with pytest.raises(ValueError):run()
    assert calls==['original-exit'] and value.poisoned
    value.finalize(publish_to(area),'failed-summary.json',primary=original)
    assert not json.loads((area/'failed-summary.json').read_bytes())['measurements_complete'] if (area/'failed-summary.json').exists() else True
    put(area/'RESULT.json',{'origin':ORIGIN,'original_exit_ran':True,'primary_preserved':pending,
        'incomplete_journal_retained':True})


def test_emitted_mode_rng_restoration_runs_when_its_start_clock_refuses(area):
    value,clock,sink=ledger(area);modes=[(NS(training=True),False),(NS(training=False),True)]
    restored=[]
    source=(STAGE/'cold-build-v2/workers/models/predictor_pgd_workload_target_evaluation_v1.py').read_bytes()
    restore=engine.restoration_bridge(source,lambda state,device:restored.append((state,device)))
    primary=RuntimeError('FAKE in-flight work error')
    try:
        raise primary
    except RuntimeError:
        value.clock=lambda:float('nan')
        restore(modes,'FAKE-RNG','cuda',NS(phases=value))
        assert [model.training for model,_ in modes]==[False,True]
        assert restored==[('FAKE-RNG','cuda')]
        assert any('Phase accounting' in text for text in primary.__notes__)
    value.finalize(publish_to(area),'failed-summary.json',primary=primary)
    put(area/'RESULT.json',{'origin':ORIGIN,'exact_emitted_restore_bridge':True,'all_original_restoration_ran':True})


@pytest.mark.parametrize('fault',['backward','nan','deadline'])
def test_start_observation_refuses_before_scientific_constructor(area,fault):
    value,clock,sink=ledger(area);calls=[]
    value.clock={'backward':lambda:99.,'nan':lambda:float('nan'),'deadline':lambda:820.}[fault]
    with pytest.raises((ValueError,TimeoutError)):
        value.context(lambda:calls.append('SCIENCE'),index=0)
    assert not calls
    value.finalize(publish_to(area),'failed-summary.json',primary=RuntimeError('FAKE stop'))
    put(area/'RESULT.json',{'origin':ORIGIN,'constructor_not_called':True,'fault':fault})


@pytest.mark.parametrize('fault',['pid','thread'])
def test_foreign_phase_owner_refuses_without_scientific_work(area,monkeypatch,fault):
    value,clock,sink=ledger(area)
    if fault=='pid':monkeypatch.setattr(phases,'os',NS(**{**vars(os),'getpid':lambda:value.owner[0]+1}))
    else:monkeypatch.setattr(phases,'threading',NS(get_ident=lambda:value.owner[1]+1))
    with pytest.raises(ValueError,match='owned process/thread'):
        value.begin('model_preparation')
    sink.close()
    put(area/'RESULT.json',{'origin':ORIGIN,'foreign_owner_refused':fault})


@pytest.mark.parametrize('fault',['rows','spans','file-bytes'])
def test_bounded_phase_capacity_and_journal_tamper_stop_before_body(area,fault):
    value,clock,sink=ledger(area);calls=[]
    if fault=='rows':value.rows=phases.MAX_ROWS
    elif fault=='spans':value.spans=phases.MAX_SPANS
    else:
        with (area/'phase-events.jsonl').open('ab') as out:out.write(b'UNOWNED METADATA')
    with pytest.raises(ValueError):
        value.context(lambda:calls.append('SCIENCE'),index=0)
    assert not calls
    value.finalize(publish_to(area),'failed-summary.json',primary=RuntimeError('FAKE observer stop'))


@pytest.mark.parametrize('fault',['kind','domain','owner','deadline','scope-epoch','excluded-boundary'])
def test_saved_timing_mutations_cannot_change_clock_or_boundary_claims(area,boundary_golden,fault):
    _,raw,summary,_=boundary_golden
    summary=copy.deepcopy(summary)
    rows=[json.loads(line) for line in raw.splitlines()]
    if fault=='scope-epoch':summary['scope_started']+=1000000.
    elif fault=='excluded-boundary':summary['intervals_exclude_own_end_row_publication_and_return']=False
    else:
        for row in rows:
            if fault=='kind':row['kind']='unreviewed-phase-kind'
            elif fault=='domain':row['clock_domain']='perf-counter/other-epoch'
            elif fault=='owner':row['pid']+=1
            else:row['deadline']+=1
        previous='0'*64
        for row in rows:
            row['previous_sha256']=previous;row.pop('row_sha256')
            row['row_sha256']=build.digest(build.encoded(row));previous=row['row_sha256']
        raw=b''.join(build.encoded(row) for row in rows)
        summary['journal']={'bytes':len(raw),'sha256':build.digest(raw)}
        summary['final_row_sha256']=previous
    with pytest.raises(ValueError):metadata_validate(raw,summary)
    put(area/'MUTATED-JOURNAL.jsonl',raw);put(area/'MUTATED-SUMMARY.json',summary)


def test_unbound_entry_and_dictionary_timing_binding_refuse_before_scientific_imports(area):
    source=(STAGE/'cold-build-v2/workers/models/_pgd_workload_entry_v1.py').read_bytes()
    original=__import__;seen=[]
    def importing(name,*args,**kw):
        if name=='platform':return NS(system=lambda:'Linux',node=lambda:'linux10614')
        if name=='models':
            seen.extend(kw.get('fromlist',args[2] if len(args)>2 else ()))
            return NS(predictor_pgd_phase_accounting_v1=phases)
        assert name.split('.')[0] not in ('torch','numpy')
        return original(name,*args,**kw)
    def namespace():
        ns={'__name__':'_FAKE_ENTRY','__builtins__':{**vars(__import__('builtins')),'__import__':importing}}
        exec(compile(source,'<FAKE-UNBOUND-PHASE-ENTRY>','exec'),ns);return ns
    ns=namespace()
    with pytest.raises(ValueError,match='unbound'):ns['entry'](NS(),objective_timer=lambda:None)
    assert ns['_USED'] and not seen
    ns=namespace()
    args=NS(host='linux10614',control_sha256='a'*64,manifest_sha256='b'*64,
        output=str(area/'unit'),deadline=820.,role='original-unit',plan_sha256='c'*64)
    ns['EXECUTION_BINDING']={'kind':'whole-pgd-reviewed-child-binding-v1','host':args.host,
        'control_sha256':args.control_sha256,'manifest_sha256':args.manifest_sha256,
        'result_root':args.output,'deadline':args.deadline,'role':args.role}
    ns['PHASE_ACCOUNTING']={'complete':True,'approved':True}
    with pytest.raises(ValueError,match='live owned'):ns['entry'](args,objective_timer=lambda:None)
    assert not any(name in seen for name in ('predictor_pgd_workload_target_v1','predictor_forward_preserving_workload_v1','_pgd_workload_inputs_v1'))


def test_cold_full_byte_inverses_caps_and_original_owned_exit_body(area):
    built=build.assemble(ROOT);cfg,files=build.load(ROOT)
    original={name:__import__('base64').b64decode(value) for name,value in json.loads(files[cfg['pins']['payload']]).items()}
    for name,edits in built['derivation']['changes'].items():
        raw=built['generated'][build.OUTER] if name=='outer' else built['uploads'][name]
        expected=files[cfg['pins']['outer']] if name=='outer' else original[name]
        assert build.inverse(raw,edits)==expected
    def node(raw,name):
        return next(n for n in ast.parse(raw).body if getattr(n,'name',None)==name)
    a=node(original['overlay/models/predictor_pgd_workload_target_v1.py'],'OwnedStep')
    b=node(built['workers']['models/predictor_pgd_workload_target_v1.py'],'OwnedStep')
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    assert built['sources']['workload_watch_v1.py']==original['supervisor/workload_watch_v1.py']
    assert built['sources']['predictor_pgd_split_watch_v1.py']==original['supervisor/predictor_pgd_split_watch_v1.py']
    assert built['derivation']['maximum_metadata_bytes_including_partials']==27705344
    assert built['derivation']['uploaded_sources']==56 and built['derivation']['cached_modules']==47
    with pytest.raises(ValueError,match='unqualified'):budget.require_execution()
    with pytest.raises(ValueError,match='unqualified'):closed_client.run(area/'not-a-package')
    put(area/'RESULT.json',{'origin':ORIGIN,'source_inverses':len(built['derivation']['changes']),
        'original_owned_exit_AST_exact':True,'old_budget_gate_closed':True})


def collect(value,area):
    value.owner.driver.assert_idle();value.owner.driver.outer_journal.close()
    codec=value.v.load('models.predictor_pgd_split_collection_v1')
    collector=codec.Collector(value.guard,value.built['control'],value.auth,connection_started=100.,clock=value.clock)
    collector.prepare();output=io.BytesIO();collector.stream(output)
    raw=output.getvalue();put(area/'WIRE.bin',raw)
    receipt=codec.receive(io.BytesIO(raw),area/'received',value.built['control'],value.auth,
        window=codec.Window(100.,collector.window.started,clock=value.clock))
    assert receipt['transport_verified'] and not receipt['scientific_qualified'] and not receipt['training_admitted']
    put(area/'TRANSPORT.json',receipt)
    return receipt


def test_integrated_primary_phase_counts_context_boundaries_parent_and_wire(area):
    value=engine.Engine(area)
    start=value.clock();value.clock.now+=.02
    value.coordinator.record_outer_setup(value.guard,value.csha,start,value.clock())
    value.owner.run_work()
    assert value.owner.state['completed']==['proof','original-unit','candidate-unit','candidate-replay']
    assert value.owner.state['complete'] is False
    assert len(value.finder.files)==56
    receipt=collect(value,area)
    for role in value.owner.state['completed'][1:]:
        p=value.root/'capture/results/workload'/role
        summary=json.loads((p/'phase-summary.json').read_bytes())
        assert summary['measurements_complete'] and summary['completed_spans']['owned_context']==40
        assert summary['completed_spans']['selection']==42 and summary['completed_spans']['score']==6
        rows=[json.loads(line) for line in (p/'phase-events.jsonl').read_bytes().splitlines()]
        measured=[row['seconds'] for row in rows if row['event']=='end' and row['name']=='owned_context']
        legacy=[json.loads(path.read_bytes())['context_seconds_including_observations'] for path in sorted((p/'unit').glob('step-*-complete.json'))]
        assert all(new>old+.006 for new,old in zip(measured,legacy))
        restored=json.loads((area/('RESTORATION-'+role+'.json')).read_bytes())
        assert restored['restored'] and not restored['model_training']
    parent=json.loads((value.root/'phase-parent-summary.json').read_bytes())
    assert parent['measurements_complete'] and parent['completed_spans']['proof']==1
    assert parent['pid']==value.parent
    put(area/'RESULT.json',{'origin':ORIGIN,'new_fake_whole_unit_lifecycle':True,'files':len(receipt['files']),
        'actual_Torch_model_gradient_or_target':False,'all_phase_sources_used':True})
    put(area/'EVENTS.json',value.events)


@pytest.mark.parametrize('fault,kind',[('gradient','RuntimeError'),('timer-exit','RuntimeError'),('phase-publish','OSError')])
def test_integrated_phase_failure_preserves_primary_and_restoration_stops_later_roles(area,fault,kind):
    value=engine.Engine(area,fault=fault)
    with pytest.raises(RuntimeError):value.owner.run_work()
    assert value.owner.state['phase']=='failed' and value.owner.state['completed']==['proof']
    assert len(value.children)==2
    errors=[json.loads(path.read_bytes()) for path in area.glob('child-error-*.json')]
    assert len(errors)==1 and errors[0]['type']==kind
    restored=json.loads((area/'RESTORATION-original-unit.json').read_bytes())
    assert restored['restored'] and not restored['model_training']
    prefix=value.root/'capture/results/workload/original-unit'
    if fault=='phase-publish':
        assert not (prefix/'phase-summary.json').exists()
        assert json.loads((prefix/'complete.json').read_bytes())['complete'] is True
    else:
        summary=json.loads((prefix/'phase-summary.json').read_bytes())
        assert not summary['measurements_complete'] and summary['problem']['type']==kind
        assert not (prefix/'complete.json').exists()
    collect(value,area)
    put(area/'RESULT.json',{'origin':ORIGIN,'original_worker_primary':kind,'mode_rng_cleanup_preserved':True,
        'later_roles_started':False,'old_complete_marker_not_promoted':True})
    put(area/'EVENTS.json',value.events)
