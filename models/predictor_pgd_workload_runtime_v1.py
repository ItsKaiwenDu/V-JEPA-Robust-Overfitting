"""Cold target-sized workload derivation. Scientific sources are never imported here."""
import ast
import hashlib
import json
from pathlib import Path
import stat

INPUTS = 'configs/predictor_pgd_workload_runtime_inputs_v1.json'
INPUTS_SHA = '6b286c178c82ef4f93b18ec4453782c1de4b8a11979719a1b1cfc30933ce80aa'
EXECUTION_BINDING = None
ATTACK = 'attacks/pgd_workload_target_v1.py'
UNIT = 'models/predictor_pgd_workload_target_v1.py'
EVALUATION = 'models/predictor_pgd_workload_target_evaluation_v1.py'
ROUTE = 'models/predictor_forward_preserving_workload_v1.py'
ENTRY = 'models/_pgd_workload_entry_v1.py'
PROVIDER = 'models/_pgd_workload_inputs_v1.py'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inverse(raw, edits):
    for e in reversed(edits):
        a = e['start']; b = a+len(e['after'].encode())
        require(raw[a:b] == e['after'].encode(), 'Exact source inverse required.')
        raw = raw[:a]+e['before'].encode()+raw[b:]
    return raw


class Editor:
    def __init__(self, raw):
        self.original = self.raw = raw
        self.edits = []

    def one(self, before, after):
        require(self.raw.count(before.encode()) == 1, 'One cold edit required: '+before[:100])
        start = self.raw.index(before.encode())
        self.raw = self.raw[:start]+after.encode()+self.raw[start+len(before.encode()):]
        self.edits.append({'start': start, 'before': before, 'after': after})

    def node(self, name, replacement):
        n = next(n for n in ast.parse(self.raw).body if getattr(n, 'name', None) == name)
        self.one(ast.get_source_segment(self.raw.decode(), n), replacement.strip())

    def finish(self):
        require(inverse(self.raw, self.edits) == self.original, 'Whole-byte inverse required.')
        compile(self.raw, '<COLD-UNEXECUTED-WORKLOAD-SOURCE>', 'exec')
        return self.raw, {'original_sha256': digest(self.original), 'derived_sha256': digest(self.raw),
                          'edits': self.edits}


DOMAIN = '''def local_domain(clean, epsilon, step_size, steps, restarts, seed, random_start):
    if torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
        raise ValueError('First-order FP32 mode required.')
    if (not isinstance(clean, Tensor) or clean.device != torch.device('cuda:0')
            or clean.dtype != torch.float32 or tuple(clean.shape) != (1,3,16,256,256)
            or epsilon != 4/255 or step_size not in (.5/255,1/255,2/255)
            or type(steps) is not int or steps not in (20,50)
            or type(restarts) is not int or restarts != 2 or type(seed) is not int
            or not 0 <= seed < 2**63 or random_start is not True):
        raise ValueError('Full-size CUDA0 PGD20/R2 or PGD50/R2 only; no CPU fallback.')
'''

BOUNDARY = '''def local_boundary(clean, settings, video_id, sensitivity_banks):
    if (EXECUTION_BINDING is None or OBJECTIVE_TIMER is None
            or platform.system() != 'Linux' or platform.node().split('.')[0] != EXECUTION_BINDING['host']
            or torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8'):
        raise ValueError('Unbound target workload; no local or alternate runtime fallback.')
    if (type(settings) is not EvaluationSettings or settings.method not in ('primary','convergence')
            or type(video_id) is not str or not video_id or video_id.startswith('local-workload-control/')
            or type(sensitivity_banks) is not int or sensitivity_banks not in (0,2) or clean.requires_grad):
        raise ValueError('One original training-row workload with declared settings required.')
    settings.__post_init__()
    local_domain(clean,settings.epsilon_255/255,settings.step_size_255/255,
                 settings.steps,settings.restarts,seeds_for(settings,video_id)['start'],True)
    if not torch.isfinite(clean).all() or clean.min()<0 or clean.max()>1:
        raise ValueError('Finite original raw pixels required.')
'''

FACTORY = '''class TargetFactory:
    def __init__(self, role):
        if role not in ('original-unit','candidate-unit','candidate-replay'):
            raise ValueError('Exact whole-unit role required.')
        self.original = role == 'original-unit'
        self.kind = 'full-eot8-original-owned-v1' if self.original else 'full-eot8-forward-preserving-owned-v1'

    def create(self, model):
        from models import predictor_forward_preserving_workload_v1 as target
        return OriginalRoute(model) if self.original else target.TargetRoute(model,tiny=False)

    def observe(self, route):
        native = route.native
        if (route.last_task.state != 'complete' or not route.last_task.receipt['hook_removed']
                or native._task is not None or len(native.scopes)!=1 or native.refusals
                or not 0<len(native.rows)<=4096):
            raise ValueError('Exact native owned root and unchanged call cap required.')
        result = {'kind':self.kind,'native_root_hook_removed':True,'native_scopes':1,
                  'native_calls':len(native.rows),'native_refusals':0}
        if not self.original:
            scope,calls=route.prototype.scope,route.prototype.calls
            if scope['state']!='complete' or not scope['native_root_hook_removed'] or len(calls)!=384:
                raise ValueError('Complete 192 initial VJPs / 384 predictor calls required.')
            result.update(initial_predictor_VJPs=192,forward_calls=384)
        return result

    def idle(self, model):
        from models import predictor_forward_preserving_workload_v1 as target
        from models import predictor_eot8_adapter_v1 as adapter
        if (adapter._ACTIVE or target.cpu._ACTIVE or hasattr(model,target.MARKER)):
            raise ValueError('Full-size route has not restored its private bindings.')


class OriginalRoute:
    def __init__(self, model):
        from models import predictor_forward_preserving_workload_v1 as target
        # Validate the same full model/source/domain without entering a candidate route.
        checked=target.TargetRoute(model,tiny=False)
        self.model,self.native=model,checked.native
        self.entered=self.used=False;self.tokens=None;self.last_task=None

    def __enter__(self):
        if self.entered:raise ValueError('One fresh original route only.')
        self.entered=True;self.native.__enter__();self.native.phase='EOT8-before';return self

    def loss(self,value,targets,bank):
        if self.tokens is not None or self.used:raise ValueError('One original root only.')
        root=native_objective(self.model,targets,bank)(value).sum();self.tokens=(root,value);return root

    def gradient(self,root,value):
        if self.used or self.tokens is None or root is not self.tokens[0] or value is not self.tokens[1]:
            raise ValueError('One exact original loss/input token required.')
        self.used=True
        try:
            task=self.last_task=self.native.prepare_backward(root,value)
            result,=task.run();self.native.check();return result
        finally:self.tokens=None

    def __exit__(self,*error):
        self.tokens=None;return self.native.__exit__(*error)
'''

ENTRY_SOURCE = '''"""UNBOUND whole-unit entry; refuses before importing scientific dependencies."""
import math
import os
from pathlib import Path
import platform
EXECUTION_BINDING=None
_USED=False

def claim(args):
    global _USED
    if _USED:raise ValueError('Entry consumed; no retries.')
    _USED=True
    if EXECUTION_BINDING is None:raise ValueError('Whole-unit execution identity is unbound.')
    b=EXECUTION_BINDING
    keys={'kind','host','control_sha256','manifest_sha256','result_root','deadline','role'}
    if (type(b) is not dict or set(b)!=keys or b['kind']!='whole-pgd-reviewed-child-binding-v1'
            or platform.system()!='Linux' or platform.node().split('.')[0]!=b['host']
            or any(getattr(args,k,None)!=b[k] for k in keys-{'kind','result_root'})
            or args.output!=b['result_root'] or b['role'] not in ('original-unit','candidate-unit','candidate-replay')
            or not all(type(b[k]) is str and len(b[k])==64 and all(c in '0123456789abcdef' for c in b[k]) for k in ('control_sha256','manifest_sha256'))
            or type(b['deadline']) not in (int,float) or not math.isfinite(b['deadline'])
            or not Path(b['result_root']).is_absolute()):
        raise ValueError('Exact new child/source/result/host/role/deadline binding required.')
    return dict(b)

def entry(args, *, objective_timer=None):
    binding=claim(args)
    if objective_timer is None:raise ValueError('Reviewed continuing hard objective timer is unbound.')
    from models import predictor_pgd_workload_target_v1 as unit
    from models import predictor_forward_preserving_workload_v1 as route
    from models import _pgd_workload_inputs_v1 as inputs
    unit.EXECUTION_BINDING=binding;unit.OBJECTIVE_TIMER=objective_timer
    route.EXECUTION_HOST=binding['host']
    return inputs.run_prepared(args,unit)
'''

PROVIDER_SOURCE = '''"""Original packet/checkpoint/upstream preparation; invoked only after new binding."""
def run_prepared(args,unit):
    import os
    import platform
    import time
    import numpy as np
    import torch
    from models.candidate_evaluation import EvaluationSettings
    from models.execution import ExecutionGuard,configure_execution,apply_model_policy
    from models.eot_profile_packet import verify
    from models.reuse_diagnostic import upstream_gate
    from models.reuse_handoff import training_input
    from models.vjepa2 import build_vitl,load_checkpoint,sample_masks
    from models.diagnostics import state_digest
    from models import predictor_pgd_workload_runtime_records_v1 as records
    started=time.monotonic()
    if (platform.python_version()!='3.11.17' or torch.__version__!='2.8.0+cu128'
            or torch.version.cuda!='12.8' or np.__version__!='2.4.6' or not torch.cuda.is_available()):
        raise ValueError('Exact original runtime required; no installation or fallback.')
    if time.monotonic()>=args.deadline:raise TimeoutError('Continuing unit clock exhausted before model preparation.')
    configure_execution('shared-fp64-native',device='cuda');torch.set_num_threads(4);torch.manual_seed(20261029)
    torch.cuda.set_per_process_memory_fraction(7*2**30/torch.cuda.get_device_properties(0).total_memory)
    _,parent=verify(args.packet,'c95c0b21ee018dfd6c897756baf453dc7d69eb1d52b5c64d99e1ac4b4b945dbd')
    upstream_gate(args.upstream,parent)
    model=build_vitl(args.upstream,activation_checkpointing=True)
    load_checkpoint(model,args.checkpoint,'5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012')
    clean=training_input(args.packet,parent)
    apply_model_policy(model,'shared-fp64-native');model.to('cuda').eval();clean=clean.to('cuda')
    guard=ExecutionGuard(model,'cuda',target_versions=True);guard.check()
    baseline={'weights_sha256':state_digest(model),'clean_sha256':unit.tensor_hash(clean),
        'video_id':parent['training_row']['video_id'],
        'checkpoint_sha256':'5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'}
    masks=lambda seed:[p.to('cuda') for p in sample_masks(args.upstream,batch_size=1,seed=seed%(2**31-1))]
    settings=EvaluationSettings.for_method(args.method,step_size_255=args.step_size_255)
    result=unit.TargetUnit(factory=unit.TargetFactory(args.role)).run(model,clean,masks,settings,
        video_id=parent['training_row']['video_id'],sensitivity_banks=args.sensitivity_banks,journal=args.output)
    guard.check()
    plan={'role':args.role,'method':args.method,'step_size_255':args.step_size_255,
        'sensitivity_banks':args.sensitivity_banks,'control_sha256':args.control_sha256,
        'source_manifest_sha256':args.manifest_sha256,'plan_sha256':args.plan_sha256}
    row=records.wrap_work(result[0],role=args.role,plan=plan,baseline=baseline,pid=os.getpid(),
        wall_seconds=time.monotonic()-started,peak_allocator_bytes=torch.cuda.max_memory_allocated())
    return row,result[1]
'''


def derive(pool):
    files, changes = {}, {}
    def finish(name, oldname, change):
        e = Editor(pool[oldname]); change(e); files[name],changes[name]=e.finish()
        changes[name]['original_path']=oldname
    finish(ATTACK,'attacks/pgd_workload_v1.py',lambda e:e.node('local_domain',DOMAIN))
    finish(EVALUATION,'models/predictor_pgd_workload_evaluation_v1.py',lambda e:e.one(
        'from attacks.pgd_workload_v1 import','from attacks.pgd_workload_target_v1 import'))
    def route(e):
        e.one("MARKER = '_forward_preserving_target_diagnostic'", "MARKER = '_forward_preserving_whole_workload_v1'\nEXECUTION_HOST=None")
        e.one("if type(tiny) is not bool or (tiny and host in ('linux10603', 'linux10609')):","if tiny is not False:")
        e.one("if not tiny and (host != 'linux10603' or platform.system() != 'Linux'):",
              "if EXECUTION_HOST is None or host != EXECUTION_HOST or platform.system() != 'Linux':")
    finish(ROUTE,'models/predictor_forward_preserving_target_v1.py',route)
    def workload(e):
        e.one('from attacks.pgd_workload_v1 import','from attacks.pgd_workload_target_v1 import')
        e.one("IDENTITY = 'local-whole-pgd-input-gradient-workload-v1'","IDENTITY = 'target-whole-pgd-input-gradient-workload-v1'")
        e.one('EXECUTION_BINDING = None','EXECUTION_BINDING = None\nOBJECTIVE_TIMER = None')
        e.node('local_boundary',BOUNDARY)
        e.node('FrozenCPUFactory',FACTORY)
        e.one("        if type(factory) is not FrozenCPUFactory:\n            # A separate named port makes analytic evidence impossible to label as native.\n            from models.predictor_pgd_workload_controls_v1 import AnalyticFactory\n            if type(factory) is not AnalyticFactory:\n                raise ValueError('Only the concrete frozen CPU route or explicit analytic control port.')",
              "        if type(factory) is not TargetFactory:raise ValueError('Exact full-size owned route factory required.')")
        e.one('value.detach().contiguous().reshape(-1).view(torch.uint8)','value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)')
        e.one("value.device.type != 'cpu':","value.device.type not in ('cpu','cuda'):")
        e.one('value.requires_grad, tensor_hash(value)))','value.requires_grad, tensor_hash(value), value.device))')
        e.one('for value, version, shape, dtype, required, want in self.fixed:','for value, version, shape, dtype, required, want, device in self.fixed:')
        e.one("value.device.type != 'cpu' or value.requires_grad != required",'value.device != device or value.requires_grad != required')
        e.one("or value.device.type != 'cpu' or not value.requires_grad", "or value.device != torch.device('cuda:0') or not value.requires_grad")
        e.one('class LocalUnit:', 'class TargetUnit:')
        e.one('sum(p.numel() for p in model.parameters()) > 2_000_000','sum(p.numel() for p in model.parameters()) > 1_000_000_000')
        e.one("p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()", "p.device != torch.device('cuda:0') or p.dtype != torch.float32 for p in model.parameters()")
        e.one('from models.predictor_pgd_workload_evaluation_v1 import','from models.predictor_pgd_workload_target_evaluation_v1 import')
        e.one('    def idle(self, *, modes=True):', '''    def time_gate(self):
        end=EXECUTION_BINDING['deadline']
        if self.active is not None and hasattr(self.active,'deadline'):end=min(end,self.active.deadline)
        if time.monotonic()>=end:raise TimeoutError('Continuing unit/objective deadline reached.')

    def idle(self, *, modes=True):
        self.time_gate()''')
        e.one('        b.snapshot.check(idle=False)','        b.time_gate()\n        b.snapshot.check(idle=False)')
        e.one('        self.route = self.root = self.losses = None','        self.route = self.root = self.losses = None\n        self.timer=None;self.timer_entered=False')
        e.one('            self.route = b.factory.create(b.model)', '''            self.deadline=min(EXECUTION_BINDING['deadline'],time.monotonic()+120)
            self.timer=OBJECTIVE_TIMER(self.deadline,hard_deadline=EXECUTION_BINDING['deadline'])
            self.timer.__enter__();self.timer_entered=True
            self.route = b.factory.create(b.model)''')
        e.one('            self.root = self.losses = self.value = None','''            self.root = self.losses = self.value = None
            if self.timer_entered:
                self.timer_entered=False
                self.timer.__exit__(kind,error,tb)''')
        # Snapshot CUDA autocast is checked as well as CPU state, before and after.
        e.one("self.flags = (torch.is_inference_mode_enabled(), torch.is_autocast_enabled('cpu'),", "self.flags = (torch.is_inference_mode_enabled(), torch.is_autocast_enabled('cpu'), torch.is_autocast_enabled('cuda'),")
        e.one("self.flags != (torch.is_inference_mode_enabled(), torch.is_autocast_enabled('cpu'),", "self.flags != (torch.is_inference_mode_enabled(), torch.is_autocast_enabled('cpu'), torch.is_autocast_enabled('cuda'),")
    finish(UNIT,'models/predictor_pgd_workload_v1.py',workload)
    files[ENTRY]=ENTRY_SOURCE.encode();files[PROVIDER]=PROVIDER_SOURCE.encode()
    # The whole-unit monitor keeps every owned-handle/RSS/cleanup predicate;
    # only the unit deadline, cadence and observation schema differ.
    e=Editor(pool['models/predictor_pgd_split_watch_v1.py'])
    e.one('start+120','start+720')
    e.one('pause(min(.1,phase_deadline-now,work_deadline-now))',
          'pause(min(.5,phase_deadline-now,work_deadline-now))')
    e.one('pgd-split-component-parent-observation-v1','whole-pgd-parent-observation-v1')
    files['supervisor/workload_watch_v1.py'],changes['supervisor/workload_watch_v1.py']=e.finish()
    changes['supervisor/workload_watch_v1.py']['original_path']='models/predictor_pgd_split_watch_v1.py'
    return files,changes


def imports(raw):
    found, external = set(),set()
    for n in ast.walk(ast.parse(raw)):
        if isinstance(n,ast.Import):
            names=[a.name for a in n.names]
        elif isinstance(n,ast.ImportFrom):
            require(n.level==0,'Relative source imports need separate review.')
            names=[n.module+'.'+a.name for a in n.names] if n.module in ('models','attacks','datasets') else [n.module or '']
        else:continue
        for name in names:
            if name.split('.')[0] in ('models','attacks','datasets'):found.add(name)
            else:external.add(name.split('.')[0])
    return found,external


def assemble(repository):
    require(EXECUTION_BINDING is None,'Cold builder must stay unbound.')
    root=Path(repository);raw=(root/INPUTS).read_bytes()
    require(INPUTS_SHA is not None and digest(raw)==INPUTS_SHA,'Pinned workload input contract required.')
    contract=json.loads(raw);pool={}
    for name,d in contract['files'].items():
        path=root/name;require(stat.S_ISREG(path.lstat().st_mode),'Never open special input.')
        data=path.read_bytes();require(len(data)==d['bytes'] and digest(data)==d['sha256'],'Frozen input changed: '+name);pool[name]=data
    generated,changes=derive(pool);available={**pool,**generated}
    # Include frozen nonimport dependencies required by scientific source readers.
    old=ast.parse(pool['models/predictor_pgd_workload_v1.py'])
    pins=ast.literal_eval(next(n.value for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FROZEN_SOURCES' for t in n.targets)))
    queue=[ENTRY,UNIT,ROUTE,*pins];seen=set();edges={};external=set()
    while queue:
        name=queue.pop()
        if name in seen:continue
        require(name in available,'Closed import source missing: '+name)
        seen.add(name);deps,ext=imports(available[name]);external.update(ext);edges[name]=[]
        for dep in sorted(deps):
            p=dep.replace('.','/')+'.py'
            if p not in available:p=dep.replace('.','/')+'/__init__.py'
            require(p in available,'Unresolved literal project import: '+dep)
            edges[name].append(p);queue.append(p)
            queue.append(p.split('/')[0]+'/__init__.py')
    files={n:available[n] for n in sorted(seen)}
    require(not any(n in files for n in ('models/predictor_pgd_target_worker_v1.py','models/predictor_pgd_cpu_v1.py','models/predictor_pgd_workload_controls_v1.py')),'No old cohort/analytic dispatch in the target graph.')
    files['supervisor/workload_watch_v1.py']=generated['supervisor/workload_watch_v1.py']
    require(len(files)<=190 and sum(map(len,files.values()))<=2*2**20,'Original source envelope/file-count reserves required.')
    for n,data in files.items():compile(data,'<UNEXECUTED-'+n+'>','exec')
    manifest={'kind':'cold-target-whole-pgd-runtime-v1','execution_binding':None,
              'files':{n:{'bytes':len(v),'sha256':digest(v)} for n,v in files.items()},
              'imports':edges,'external_literal_roots':sorted(external),'changes':changes,
              'nonimport_frozen_pins':pins,'all_byte_inverses_verified':True,
              'scientific_arithmetic_ownership_caps_unchanged':True,'actual_scientific_execution':False,
              'training_admitted':False,'new_scope_or_handoff':False}
    return files,manifest


def write_local(repository,destination):
    files,manifest=assemble(repository);dest=Path(destination);dest.mkdir(mode=0o700)
    for name,data in files.items():
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with p.open('xb') as out:out.write(data)
        p.chmod(0o600)
    with (dest/'SOURCE-MANIFEST-v1.json').open('xb') as out:out.write(encoded(manifest))
    return manifest
