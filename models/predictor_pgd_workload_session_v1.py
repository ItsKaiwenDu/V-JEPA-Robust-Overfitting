"""Whole-scratch/cache/owned-monitor/collection integration with explicit fake OS ports.

This local core has no launcher, timer installation, GPU/proof or SSH API. Every
simulation receipt is marked; no dictionary changes scientific admission.
"""
import ast
import builtins
import copy
import importlib.util
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
import threading
from types import ModuleType, SimpleNamespace as NS

from models import predictor_pgd_workload_runtime_v1 as cold
from models import predictor_pgd_workload_runtime_records_v1 as records
from models import predictor_pgd_candidate_scratch_v1 as scratch
from models import predictor_pgd_candidate_bootstrap_v1 as frozen_boot

ORIGIN='EXPLICIT-FAKE-WHOLE-PGD-OS-COMPUTE-PORTS-V1'
ROLES=('proof',*records.ROLES)
SECONDS={'proof':60,**{r:720 for r in records.ROLES}}
PURE=frozenset(('models','attacks','models._pgd_workload_entry_v1'))
encoded=cold.encoded;digest=cold.digest;require=cold.require


def publish(guard,name,row,*,replace=False,expected_sha=None):
    """Bounded anchored metadata append; replacement requires the owned old hash."""
    d=guard.layout.classify(name);raw=encoded(row)
    require(d['category']=='metadata' and len(raw)<=d['max_bytes'] and (not replace or d['mutable']),
            'Only bounded declared metadata may be published.')
    if replace:
        require(expected_sha is not None and digest(guard.read(name,d['max_bytes']))==expected_sha,
                'Only the owned current metadata may be replaced.')
    fd=guard.open_root();held=[]
    try:
        parts=PurePosixPath(name).parts
        for part in parts[:-1]:
            before=os.stat(part,dir_fd=fd,follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode) and stat.S_IMODE(before.st_mode)==0o700
                and before.st_uid==os.getuid() and before.st_dev==guard.binding['root_identity'][0],
                'Private same-device publication ancestors required.')
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            require(scratch.directory_id(os.fstat(child))==scratch.directory_id(before),'Publication ancestor changed.')
            held.append((fd,part,before));fd=child
        leaf=parts[-1]
        with os.fdopen(os.open(leaf+'.partial',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd),'wb') as out:
            out.write(raw);out.flush();os.fsync(out.fileno())
        if replace:os.replace(leaf+'.partial',leaf,src_dir_fd=fd,dst_dir_fd=fd)
        else:
            os.link(leaf+'.partial',leaf,src_dir_fd=fd,dst_dir_fd=fd,follow_symlinks=False)
            os.unlink(leaf+'.partial',dir_fd=fd)
        os.fsync(fd)
        for parent,part,before in held:
            require(scratch.directory_id(os.stat(part,dir_fd=parent,follow_symlinks=False))==scratch.directory_id(before),
                    'Named publication ancestry changed.')
        guard.check_root()
    finally:
        os.close(fd)
        for parent,_,_ in reversed(held):os.close(parent)
    return digest(raw)


def definitions(raw, provided):
    ns={'__name__':'UNBOUND_WHOLE_PGD_DEFINITIONS',**provided}
    exec(compile(raw,'<UNINVOKED-WHOLE-PGD-DEFINITIONS>','exec'),ns)
    return ns


def result_specs(method):
    n={'primary':40,'convergence':100}[method]
    result={}
    def add(name,cap,mutable=False):
        result[name]={'category':'metadata','max_bytes':cap,'mutable':mutable,'partial_allowed':True}
    for role in records.ROLES:
        p='results/workload/'+role+'/'
        for name,cap in (('unit/unit-started.json',16384),('unit/unit-complete.json',512*1024),
                         ('unit/unit-failed.json',16384),('unit-work.json',768*1024),('complete.json',16384),('resource.json',16384)):
            add(p+name,cap)
        for i in range(n):
            add(p+f'unit/step-{i:03}-started.json',4096)
            add(p+f'unit/step-{i:03}-complete.json',16384)
        add('logs/'+role+'.log',2**20)
    for name,cap in (('results/replay.json',16384),('results/final-receipt.json',65536),('results/collection-status.json',65536),('logs/proof.log',2**20)):
        add(name,cap)
    require(sum(d['max_bytes'] for d in result.values())<32*2**20,'No shared metadata-cap increase.')
    return result


class Blueprint:
    def __init__(self,repository,*,method,step_size_255=1.,sensitivity_banks=0):
        require(method in ('primary','convergence') and type(step_size_255) in (int,float)
                and step_size_255 in (.5,1,2) and type(sensitivity_banks) is int
                and sensitivity_banks in (0,2),'Exact declared whole-unit control settings required.')
        self.method,self.step,self.audits=method,step_size_255,sensitivity_banks
        root=Path(repository);self.repository=root
        files,self.manifest=cold.assemble(root)
        self.manifest_bytes=encoded(self.manifest)
        self.runtime={n:d for n,d in self.manifest['files'].items() if not n.startswith('supervisor/')}
        self.payload={'overlay/'+n:files[n] for n in self.runtime}
        self.payload['supervisor/workload_watch_v1.py']=files['supervisor/workload_watch_v1.py']
        self.payload['supervisor/proof_watch_v1.py']=(root/'models/predictor_pgd_split_watch_v1.py').read_bytes()
        self.payload['runtime-source-manifest.json']=self.manifest_bytes
        self.assets={n:{'bytes':len(v),'sha256':digest(v)} for n,v in self.payload.items()}
        self.result=result_specs(method)
        self.root_meta={'control.json':(65536,False,True),'authorization.json':(65536,False,True),
            'proposal.json':(65536,False,True),'session.json':(128*1024,True,True),
            'allocation-result-v2.json':(16384,False,True),'allocation-events-v2.jsonl':(2**20,True,False)}
        for role in ROLES:
            self.root_meta['parent-'+role+'.json']=(128*1024,False,True)
            self.root_meta['worker-plan-'+role+'.json']=(16384,False,True)
        self.catalogue_bytes=encoded({'kind':'whole-pgd-local-catalogue-v1','assets':self.assets,
            'runtime':self.runtime,'result':self.result,'root_metadata':self.root_meta,
            'category_caps':scratch.CAPS,'execution_binding':None})
        self.control_bytes=encoded({'kind':'whole-pgd-REJECTED-local-control-v1','method':method,
            'step_size_255':step_size_255,'sensitivity_banks':sensitivity_banks,
            'catalogue_sha256':digest(self.catalogue_bytes),'manifest_sha256':digest(self.manifest_bytes),
            'execution_binding':None,'scientific_qualified':False,'training_admitted':False})
        self.control_sha=digest(self.control_bytes)
        self.local_binding_bytes=encoded({'kind':ORIGIN,'approved':False,'maximum_new_attempts':0,
            'control_sha256':self.control_sha,'live_execution_binding':None})
        self.installation_bytes=encoded({'kind':'whole-pgd-local-installation-v1',
            'control_sha256':self.control_sha,'files':{n:d for n,d in self.assets.items() if n.startswith('supervisor/')}})
        self.assets['installation.json']={'bytes':len(self.installation_bytes),'sha256':digest(self.installation_bytes)}
        # Inert compatibility fields are byte identities; they grant no authority.
        self.base=b'{}\n';self.design=b'{}\n'
        self.policy={'scratch_contract_text':self.catalogue_bytes.decode(),'base_text':self.base.decode(),'design_text':self.design.decode()}
        boot=NS(encoded=encoded,digest=digest,parsed=scratch.parse_json,sha=scratch.sha,
            finite=records.finite,positive=lambda v:type(v) is int and v>0,ROLES=ROLES,SECONDS=SECONDS,
            control=self.control,authorization=self.local_binding,installation=self.installation)
        outer=self
        class Layout(scratch.Layout):
            def __init__(self,*args):
                if args:require(args==(outer.catalogue_bytes,outer.base,outer.design,outer.installation_bytes,digest(outer.installation_bytes)),
                                'Exact catalogue/source installation required.')
                self.contract=json.loads(outer.catalogue_bytes);self.result=copy.deepcopy(outer.result)
                self.assets=copy.deepcopy(outer.assets);self.installation=json.loads(outer.installation_bytes)
                self.installation_sha256=digest(outer.installation_bytes)
                self.directories={'.','capture','cache','cache/cuda','tmp'}
                for n in (*self.assets,*outer.root_meta,*('capture/'+n for n in self.result)):
                    self.directories.update(str(p) for p in PurePosixPath(n).parents)
            def classify(self,name):
                scratch.safe(name)
                if name.startswith('capture/'):
                    d,plain,partial=scratch.plan.classify(name[8:],self.result)
                    return dict(d,plain='capture/'+plain,partial=partial,space='result')
                if name in ('space-proof-v2.bin.partial','emergency-space.bin') or name.startswith(('cache/','tmp/')):
                    return scratch._OriginalLayout.classify(self,name)
                partial=name.endswith('.partial');plain=name.removesuffix('.partial')
                if plain in outer.root_meta:
                    cap,mutable,allowed=outer.root_meta[plain]
                    require(not partial or allowed,'Disallowed root metadata partial.')
                    return dict(category='metadata',max_bytes=cap,plain=plain,partial=partial,mutable=mutable,partial_allowed=allowed,space='metadata')
                if name in self.assets:
                    return dict(category='code',max_bytes=self.assets[name]['bytes'],plain=name,partial=False,mutable=False,partial_allowed=False,space='asset')
                raise ValueError('Unknown whole-PGD scratch path: '+name)
        self.Layout=Layout
        # Only schema/dependency bridges change in the frozen priority codec.
        e=cold.Editor((root/'models/predictor_pgd_candidate_collection_v1.py').read_bytes())
        e.one('from models import predictor_pgd_candidate_bootstrap_v1 as boot','boot = _LOCAL_BOOT')
        e.one('from models import predictor_pgd_candidate_scratch_v1 as scratch','scratch = _LOCAL_SCRATCH')
        for before,after in (('pgd-candidate-parent-session-v1','whole-pgd-parent-session-v1'),
            ('pgd-candidate-work-receipt-v1','whole-pgd-work-receipt-v1'),
            ('pgd-candidate-evidence-stream-v1','whole-pgd-evidence-stream-v1'),
            ('Both proof and candidate roles must finish before a complete-work receipt.','Proof and all three whole-unit roles must finish before a complete-work receipt.')):
            while before.encode() in e.raw:
                start=e.raw.index(before.encode());e.raw=e.raw[:start]+after.encode()+e.raw[start+len(before.encode()):]
                e.edits.append({'start':start,'before':before,'after':after})
        codec,self.codec_derivation=e.finish();self.codec_bytes=codec
        local_scratch=NS(**vars(scratch));local_scratch.Layout=Layout
        self.codec=NS(**definitions(codec,{'_LOCAL_BOOT':boot,'_LOCAL_SCRATCH':local_scratch}))
        self.monitor=definitions(files['supervisor/workload_watch_v1.py'],{})['watch']
        self.proof_monitor=definitions(self.payload['supervisor/proof_watch_v1.py'],{})['watch']
        old=(root/'models/predictor_pgd_component_session_v1.py').read_bytes()
        node=next(n for n in ast.parse(old).body if isinstance(n,ast.ClassDef) and n.name=='PinnedCache')
        e=cold.Editor(ast.get_source_segment(old.decode(),node).encode())
        e.one('len(self.files)==37','len(self.files)=='+str(len(self.runtime)))
        cache,self.cache_derivation=e.finish()
        ns={'Path':Path,'require':require,'digest':digest,'PURE':PURE,'frozen_boot':frozen_boot,
            'builtins':builtins,'ModuleType':ModuleType,'importlib':__import__('importlib')}
        self.Cache=definitions(cache,ns)['PinnedCache']

    def control(self,data):
        require(data==self.control_bytes,'Local control bytes changed; no live controls accepted.')
        return json.loads(data),self.policy,self.control_sha

    def local_binding(self,data,csha):
        require(data==self.local_binding_bytes and csha==self.control_sha,'Only the explicit REJECTED local binding is accepted.')
        return json.loads(data)

    def installation(self,control,csha):
        require(encoded(control)==self.control_bytes and csha==self.control_sha,'Installation binding changed.')
        return self.installation_bytes,digest(self.installation_bytes)

    def live_run(self,*args,**kwargs):
        raise ValueError('Whole-unit OS execution binding is null.')


class LocalSession:
    def __init__(self,blueprint,guard,*,clock,connection_started):
        self.blueprint,self.guard,self.clock=blueprint,guard,clock
        self.origin=self.last=connection_started;self.started=False;self.collected=False;self.session_sha=None
        self.state={'kind':'whole-pgd-parent-session-v1','control_sha256':blueprint.control_sha,
            'parent_pid':os.getpid(),'connection_started':self.origin,'work_deadline':self.origin+2400,
            'connection_deadline':self.origin+3000,'phase':'new','current':None,'complete':False,
            'completed':[],'receipts':[],'evidence_origin':ORIGIN,'completion_is_simulated':True,
            'scientific_qualified':False,'training_admitted':False}
        self.check()

    def check(self,deadline=None):
        now=self.clock();end=min(self.origin+2400,deadline if deadline is not None else self.origin+2400)
        require(records.finite(now) and now>=self.last,'Continuing monotonic clock cannot move backwards.')
        if now>=end:raise TimeoutError('Original work/phase clock exhausted; no reset.')
        self.last=now;return now

    def write(self,name,row,replace=False):
        owned={'control.json','authorization.json','proposal.json','session.json','capture/results/replay.json'}
        owned.update('worker-plan-'+r+'.json' for r in ROLES);owned.update('parent-'+r+'.json' for r in ROLES)
        require(name in owned,'Parent owns only declared identity/session/plan/observation/join paths.')
        self.guard.scan(phase='capture')
        value=publish(self.guard,name,row,replace=replace,expected_sha=self.session_sha if name=='session.json' else None)
        if name=='session.json':self.session_sha=value

    def run(self,ports):
        require(not self.started,'One local session only; no retry.')
        self.started=True
        require(type(ports) is dict and set(ports)=={'origin','launch','sample','lease','verify','hard_timer','pause','gpu'}
                and ports['origin']==ORIGIN,'Explicit fake process/compute/timer ports only.')
        b=self.blueprint;self.units=[]
        try:
            self.check(self.origin+180);self.state['phase']='setup'
            self.write('control.json',json.loads(b.control_bytes));self.write('authorization.json',json.loads(b.local_binding_bytes))
            self.write('proposal.json',{'kind':ORIGIN,'approved':False,'actual_objectives':0})
            self.write('session.json',self.state)
            self.guard.scan(phase='setup',initial=True);self.guard.verify_assets()
            self.guard.proof_readiness()
            for role in ROLES:
                started=self.check();deadline=min(started+SECONDS[role],self.origin+2400)
                gpu=ports['gpu']()
                require(gpu.get('evidence_origin')==ORIGIN and gpu.get('empty_for_simulated_role') is True,
                        'Local GPU port must be explicitly simulated; live ownership is unbound.')
                plan={'kind':ORIGIN,'role':role,'method':b.method,'step_size_255':b.step,'sensitivity_banks':b.audits,
                    'control_sha256':b.control_sha,'source_manifest_sha256':digest(b.manifest_bytes),
                    'deadline':deadline,'connection_started':self.origin,'execution_binding':None}
                plan['plan_sha256']=digest(encoded(plan));self.write('worker-plan-'+role+'.json',plan)
                self.state.update(current=role,phase='running');self.write('session.json',self.state,True)
                with ports['hard_timer'](deadline,hard_deadline=self.origin+2400,phase=role):
                    child,lease=ports['launch'](role,plan,self)
                    require(child.pid not in {r['pid'] for r in self.state['receipts']},'Fresh child identity for each role required.')
                    monitor=b.proof_monitor if role=='proof' else b.monitor
                    observed=monitor(child,lease,sample=ports['sample'],lease_factory=ports['lease'],
                        gate=lambda:self.check(deadline),phase_deadline=deadline,work_deadline=self.origin+2400,
                        launch_started=started,clock=self.clock,pause=ports['pause'])
                    result=ports['verify'](role,plan,observed,self)
                    require(result.get('evidence_origin')==ORIGIN and result.get('pid')==observed['pid']
                        and result.get('start_ticks')==observed['start_ticks'] and result.get('role')==role,
                        'Explicit local fake role/source/parent receipt required.')
                    if role!='proof':
                        records.validate_work(result['work'],plan);self.units.append(result['work'])
                        require(result['work']['pid']==observed['pid']
                            and 0<=result['work']['wall_seconds']<=observed['wall_seconds']<=SECONDS[role],
                            'Own worker/parent wall identity must join.')
                    else:
                        require(result.get('proof_is_simulated') is True and result.get('actual_bytes_written')==0,
                            'Local proof port cannot assert a real storage proof.')
                    require(all(result.get(k) is False for k in ('scientific_qualified','training_admitted')),
                            'Local simulation cannot admit training/science.')
                self.write('parent-'+role+'.json',observed)
                receipt={k:v for k,v in result.items() if k!='work'}
                self.state['receipts'].append(receipt);self.state['completed'].append(role)
                self.state.update(current=None,phase='between-roles');self.write('session.json',self.state,True)
                self.check(deadline)
            self.write('capture/results/replay.json',records.join_units(*self.units))
            self.state.update(phase='await-collection',collection_pending=True)
            self.write('session.json',self.state,True);return copy.deepcopy(self.state)
        except BaseException as error:
            self.state.update(phase='failed',complete=False,error_type=type(error).__name__,error=str(error)[:1024])
            if (self.guard.root/'session.json').exists():self.write('session.json',self.state,True)
            raise

    def collect(self,out):
        require(self.started and not self.collected and self.state['phase'] in ('await-collection','failed'),
                'One ended work session only; no collection retry.')
        self.collected=True;b=self.blueprint
        collector=b.codec.Collector(self.guard,b.control_bytes,b.local_binding_bytes,
            connection_started=self.origin,clock=self.clock)
        self.header=collector.prepare();return collector.stream(out)
