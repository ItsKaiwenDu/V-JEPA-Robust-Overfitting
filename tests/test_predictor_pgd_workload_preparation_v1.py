"""New preparation controls; every CUDA/compute/OS/resource port is explicit FAKE."""
import ast
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
from types import SimpleNamespace as NS

import pytest
from models import predictor_pgd_workload_preparation_v1 as op

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/pgd-workload-preparation-v1-20261123'
PROVIDER='models/_pgd_workload_inputs_v1.py'
ORIGIN='EXPLICIT-FAKE-EARLY-DEVICE-PREPARATION; NOT A GRANT OR EXECUTION CLAIM'
spec=importlib.util.spec_from_file_location('_new_preparation_fake_support',STAGE/'fake-operator-support-v1.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)


@pytest.fixture
def case(request,tmp_path):
    base=os.environ.get('VJEPA_PREPARATION_EVIDENCE')
    path=Path(base)/request.node.name if base else tmp_path/'case'
    path.mkdir(mode=0o700);return path


@pytest.fixture(autouse=True)
def forbid_actual_work(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Physical process/timer/signal/network work forbidden.')
    for module,name in ((subprocess,'Popen'),(subprocess,'check_output'),(signal,'signal'),(signal,'setitimer'),
                        (threading.Thread,'start'),(os,'kill'),(os,'killpg'),(os,'system'),(os,'fork'),(os,'posix_spawn')):
        if hasattr(module,name):monkeypatch.setattr(module,name,refuse)
    original=builtins.__import__
    def importing(name,*a,**kw):
        if name.split('.')[0] in ('torch','numpy','socket'):raise AssertionError('No science or network imports in fake controls.')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',importing)
    monkeypatch.setattr(os,'fstatvfs',lambda fd:NS(f_bavail=20*2**30,f_frsize=1,f_favail=100000))


class Ports:
    """Synthetic memory accounting is not a target memory prediction."""
    def __init__(self,source,*,clock=None,pid=970001,fault=None,engine=None,plan=None,scope=None,timers=None):
        self.clock=clock or support.Clock();self.fault=fault;self.engine=engine;self.pid=pid
        self.plan=plan;self.scope=scope;self.timers=timers;self.events=[];self.deadline=self.clock.now+720
        self.synthetic={'origin':ORIGIN,'host_baseline':256*2**20,'CPU_destination':3*2**30,'mapped_checkpoint':2*2**30,
                        'RSS_cap':4*2**30,'GPU_cap':7*2**30,'memory_fit_or_allocation_cause_established':False}
        owner=self
        class Model:
            def __init__(self):self.device='cpu';self.training=True
            def to(self,device):
                assert device=='cuda';owner.events.append('transfer');self.device='cuda'
                if owner.fault=='transfer':raise RuntimeError('EXPLICIT-FAKE device transfer failed')
                if owner.fault=='transfer_deadline':owner.clock.now=owner.deadline
                return object() if owner.fault=='identity' else self
            def eval(self):self.training=False;owner.events.append('eval');return self
        self.model=Model()
        clean=NS(to=lambda device:self.event('clean-to',NS(device=device)))
        self.parent={'training_row':{'video_id':'explicit-fake-training-row'}}
        self.unit_return=None
        class Settings:
            @staticmethod
            def for_method(method,*,step_size_255):
                owner.events.append('settings');return NS(method=method,step_size_255=step_size_255)
        class Guard:
            def __init__(self,model,device,*,target_versions):
                assert model is owner.model and device=='cuda' and target_versions
            def check(self):
                assert owner.model.device=='cuda' and not owner.model.training;owner.events.append('guard')
        class Unit:
            def __init__(self,*,factory):self.factory=factory
            def run(self,model,clean,masks,settings,**kw):
                assert model is owner.model and not model.training;owner.events.append('unit')
                if owner.engine:
                    row=support.Engine.fake_compute(owner.engine,owner.engine.root,owner.plan,owner.scope,owner.timers)
                    owner.unit_return=row['unit']
                else:owner.unit_return={'origin':ORIGIN,'complete':False}
                return owner.unit_return,None
        def wrap(result,**kw):
            owner.events.append('wrap')
            if engine:
                from models import predictor_pgd_workload_runtime_records_v1 as records
                return records.wrap_work(result,**kw)
            return {'origin':ORIGIN,'result':result,'plan':kw['plan']}
        self.unit=NS(TargetUnit=Unit,TargetFactory=lambda role:NS(role=role),tensor_hash=lambda x:op.digest(b'FAKE-CLEAN'))
        cuda=NS(is_available=lambda:True,get_device_properties=lambda i:NS(total_memory=8*2**30),
            set_per_process_memory_fraction=self.fraction,max_memory_allocated=lambda:3*2**30)
        self.modules={
            'os':NS(getpid=lambda:pid),'platform':NS(python_version=lambda:'3.11.17'),'time':NS(monotonic=self.clock),
            'numpy':NS(__version__='2.4.6'),'torch':NS(__version__='bad' if fault=='version' else '2.8.0+cu128',
                version=NS(cuda='12.8'),cuda=cuda,set_num_threads=lambda n:self.event('threads'),manual_seed=lambda n:self.event('seed')),
            'models.candidate_evaluation':NS(EvaluationSettings=Settings),
            'models.execution':NS(ExecutionGuard=Guard,configure_execution=lambda *a,**k:self.event('configure'),
                apply_model_policy=lambda model,policy:self.event('policy',model)),
            'models.eot_profile_packet':NS(verify=lambda *a:(None,self.parent)),
            'models.reuse_diagnostic':NS(upstream_gate=lambda *a:self.event('upstream')),
            'models.reuse_handoff':NS(training_input=lambda *a:self.event('clean',clean)),
            'models.vjepa2':NS(build_vitl=self.build,load_checkpoint=self.load,sample_masks=lambda *a,**k:[]),
            'models.diagnostics':NS(state_digest=lambda model:op.digest(b'FAKE-WEIGHTS')),
            'models':NS(predictor_pgd_workload_runtime_records_v1=NS(wrap_work=wrap))}
        def importing(name,globals=None,locals=None,fromlist=(),level=0):
            assert not level and name in self.modules,('Unexpected provider import',name)
            return self.modules[name]
        self.namespace={'__builtins__':{**vars(builtins),'__import__':importing}}
        exec(compile(source,'<EXPLICIT-FAKE-PREPARATION-PORTS>','exec'),self.namespace)
    def event(self,name,value=None):self.events.append(name);return value
    def fraction(self,fraction):assert fraction==7/8;self.event('allocator-cap')
    def build(self,*a,**kw):assert kw=={'activation_checkpointing':True};return self.event('build',self.model)
    def load(self,model,path,sha):
        self.events.append('checkpoint');assert model is self.model and sha=='5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'
        rss=self.synthetic['host_baseline']+self.synthetic['mapped_checkpoint']
        if model.device=='cpu':rss+=self.synthetic['CPU_destination']
        self.synthetic['RSS_at_checkpoint']=rss
        if rss>4*2**30 or self.fault=='checkpoint':raise MemoryError('EXPLICIT-FAKE checkpoint RSS refusal')
        if self.fault=='checkpoint_deadline':self.clock.now=self.deadline
        return {'origin':ORIGIN}
    def run(self,role='original-unit'):
        args=NS(deadline=self.deadline,upstream='EXPLICIT-FAKE-UPSTREAM',checkpoint='EXPLICIT-FAKE-CHECKPOINT',
            packet='EXPLICIT-FAKE-PACKET',method='primary',step_size_255=1.,sensitivity_banks=0,role=role,
            output='/EXPLICIT-FAKE-RESULT',control_sha256='a'*64,manifest_sha256='b'*64,plan_sha256='c'*64)
        if self.plan:
            args.deadline=self.plan['deadline'];self.deadline=args.deadline
            args.method=self.scope['method'];args.step_size_255=self.scope['step_size_255'];args.sensitivity_banks=self.scope['sensitivity_banks']
            args.control_sha256=self.plan['control_sha256'];args.manifest_sha256=self.scope['runtime_manifest_sha256']
            args.plan_sha256=op.digest(op.encoded(self.plan))
            args.output=str(self.engine.root/'capture/results/workload'/role/'unit')
        if self.fault=='before_deadline':self.clock.now=self.deadline
        return self.namespace['run_prepared'](args,self.unit)
    def save(self,path):support.put(path,{'origin':ORIGIN,'events':self.events,'synthetic_memory':self.synthetic,'fake_OS_compute_GPU_ports':True})


class Engine(support.Engine):
    def fake_compute(self,root,plan,scope,timers):
        source=self.built['workers'][PROVIDER]
        assert (Path(root)/'overlay'/PROVIDER).read_bytes()==source
        fault=self.fault if plan['role']=='original-unit' else None
        ports=Ports(source,clock=self.clock,pid=self.pid,fault=fault,engine=self,plan=plan,scope=scope,timers=timers)
        try:return ports.run(plan['role'])[0]
        finally:ports.save(self.folder/('preparation-'+plan['role']+'.json'))
    def sample(self,pid,birth):
        child=self.children[pid]
        if self.fault=='rss-guard' and child.plan['role']=='original-unit':
            self.events.append(['EXPLICIT-FAKE-PREPARATION-RSS-SAMPLE',4*2**30+1])
            row=dict(self.identity(pid),rss_bytes=4*2**30+1)
            return {'processes':{pid:row},'rss_bytes':row['rss_bytes']}
        return super().sample(pid,birth)


def provider():return (STAGE/'cold-runtime-v1'/PROVIDER).read_bytes()


def collect(e,case):
    e.owner.driver.assert_idle();e.owner.driver.outer_journal.close()
    codec=e.v.load('models.predictor_pgd_split_collection_v1')
    collector=codec.Collector(e.guard,e.built['control'],e.auth,connection_started=100.,clock=e.clock)
    collector.prepare();wire=io.BytesIO();collector.stream(wire);raw=wire.getvalue();support.put(case/'wire.bin',raw)
    receipt=codec.receive(io.BytesIO(raw),case/'received',e.built['control'],e.auth,window=codec.Window(100.,e.clock(),clock=e.clock))
    support.put(case/'transport.json',receipt);support.put(case/'events.json',e.events)
    assert receipt['transport_verified'] and not receipt['scientific_qualified'] and not receipt['training_admitted']
    with pytest.raises(ValueError):e.v.original_proof(e.guard,op.digest(e.guard.read('allocation-result-v2.json',16384)))
    return receipt


def test_complete_inverses_and_frozen_scientific_source(case):
    derivation=json.loads((STAGE/'DERIVATION-v1.json').read_bytes());manifest=json.loads((STAGE/'SOURCE-MANIFEST-v1.json').read_bytes())
    original=json.loads((ROOT/manifest['original_manifest']).read_bytes())
    for name,want in original['files'].items():
        raw=(STAGE/'cold-runtime-v1'/name).read_bytes()
        if name==PROVIDER:raw=op.inverse(raw,derivation['provider']['edits'])
        assert {'bytes':len(raw),'sha256':op.digest(raw)}==want
    assert op.inverse((ROOT/'models/predictor_pgd_workload_preparation_v1.py').read_bytes(),derivation['operator']['edits'])==(ROOT/'models/predictor_pgd_workload_operator_v1.py').read_bytes()
    assert len(manifest['files'])==47 and sum(x['bytes'] for x in manifest['files'].values())==450548
    support.put(case/'RESULT.json',{'origin':ORIGIN,'byte_exact_files':46,'provider_inverse':True,'operator_inverse':True})


def test_new_pins_cache_and_catalogue_keep_existing_limits(case):
    cfg,f=op.load(ROOT);assert len(cfg['files'])==57
    for method in ('primary','convergence'):
        b=op.assemble(ROOT,method=method,sensitivity_banks=2 if method=='convergence' else 0)
        saved=json.loads((STAGE/('cold-'+method+'-v1')/'SOURCE-PAYLOAD.json').read_bytes())
        assert set(saved)==set(b['uploads']) and len(saved)==55
        # Compare full catalogue caps/result declarations with the accepted original control.
        old_control=json.loads((ROOT/'runs/pretrained/diagnostic/pgd-workload-handoff-v2-20261123/handoff-parent/linux10614-primary-v2-final-unarmed/CONTROL-WORKLOAD.json').read_bytes())
        old_policy=json.loads(json.loads(old_control['supervised_control_text'])['policy_text'])
        assert b['catalogue']['caps']==json.loads(old_policy['scratch_contract_text'])['caps']
        assert b['sources']['workload_watch_v1.py']==(STAGE/'cold-runtime-v1/supervisor/workload_watch_v1.py').read_bytes()
    support.put(case/'RESULT.json',{'origin':ORIGIN,'pins':57,'source_files':55,'caps_unchanged':True})


def test_unbound_entry_still_refuses_before_scientific_import(case):
    raw=(STAGE/'cold-runtime-v1/models/_pgd_workload_entry_v1.py').read_bytes();ns={}
    exec(compile(raw,'<UNBOUND-NEW-PREPARATION>','exec'),ns)
    with pytest.raises(ValueError,match='unbound'):ns['entry'](NS())
    assert ns['EXECUTION_BINDING'] is None
    support.put(case/'RESULT.json',{'origin':ORIGIN,'refused_before_science':True})


def test_synthetic_allocation_order_distinguishes_old_and_new_paths(case):
    manifest=json.loads((STAGE/'SOURCE-MANIFEST-v1.json').read_bytes())
    before=op.inverse(provider(),manifest['changes'][PROVIDER]['edits']);old=Ports(before)
    with pytest.raises(MemoryError):old.run()
    old.save(case/'old-order.json');new=Ports(provider());new.run();new.save(case/'new-order.json')
    assert old.synthetic['RSS_at_checkpoint']>4*2**30 and new.synthetic['RSS_at_checkpoint']<4*2**30
    assert new.events.index('build')<new.events.index('transfer')<new.events.index('checkpoint')<new.events.index('policy')<new.events.index('unit')
    assert new.events.count('transfer')==2


@pytest.mark.parametrize('fault',['transfer','identity','checkpoint','transfer_deadline','checkpoint_deadline','version','before_deadline'])
def test_preparation_failures_stop_before_unit(case,fault):
    ports=Ports(provider(),fault=fault)
    with pytest.raises((ValueError,RuntimeError,MemoryError,TimeoutError)):ports.run()
    ports.save(case/'RESULT.json');assert 'unit' not in ports.events and 'wrap' not in ports.events
    if fault in ('transfer','identity','transfer_deadline','version','before_deadline'):assert 'checkpoint' not in ports.events


@pytest.mark.parametrize('method',['primary','convergence'])
def test_new_preparation_inside_owned_fake_lifecycle_and_transport(case,method):
    e=Engine(case,method=method);result=e.owner.run_work()
    assert result['completed']==list(op.ROLES) and result['complete'] is False
    receipt=collect(e,case)
    for role in op.ROLES[1:]:
        row=json.loads((case/('preparation-'+role+'.json')).read_bytes())
        assert row['events'].index('transfer')<row['events'].index('checkpoint')<row['events'].index('unit')
    assert len(receipt['files'])==(432 if method=='primary' else 792)


@pytest.mark.parametrize('fault',['transfer','checkpoint','transfer_deadline','rss-guard'])
def test_preparation_failure_stops_later_roles_and_collects_evidence(case,fault):
    e=Engine(case,fault=fault)
    with pytest.raises((ValueError,RuntimeError,TimeoutError,MemoryError)):e.owner.run_work()
    assert e.owner.state['phase']=='failed' and e.owner.state['completed']==['proof']
    assert [x[2] for x in e.events if x[0]=='FAKE-Popen']==['proof','original-unit']
    receipt=collect(e,case);assert receipt['transport_verified']
    assert not (e.root/'capture/results/workload/candidate-unit').exists()
    if fault=='rss-guard':assert any(x[0]=='FAKE-OWNED-SIGNAL' for x in e.events)
