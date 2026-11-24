"""Prepare and inspect one unarmed whole-PGD packet; never authorizes or launches.

The reviewed 55 uploaded sources and supervised control are copied exactly.
Only the local source-pin list, client import and isolated CLI import are adapted.
"""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from types import SimpleNamespace

INPUTS = 'configs/predictor_pgd_workload_handoff_inputs_v3.json'
INPUTS_SHA = '7dbc1d2b89ec314f51ae27c8c9fff033c0aa8a6cfa7e2700f5f6375a4dd4d0a7'
OUTER = 'models/predictor_pgd_workload_handoff_outer_v3.py'
CLIENT = 'models/predictor_pgd_workload_handoff_client_v3.py'
CLI = 'scripts/run_predictor_pgd_workload_handoff_v3.py'
SELF = 'models/predictor_pgd_workload_handoff_v3.py'
READERS = {'bootstrap.py':'models/predictor_pgd_workload_bootstrap_v2.py', 'predictor_pgd_split_scratch_v1.py':'models/predictor_pgd_workload_scratch_v2.py', 'predictor_pgd_split_collection_v1.py':'models/predictor_pgd_workload_collection_v2.py'}
FILES = frozenset(('CONTROL-WORKLOAD.json', 'SOURCE-PAYLOAD.json', 'known-hosts', 'PACKAGE.json'))


def enc(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path, cap, want=None, *, private=False):
    p=Path(path); before=p.lstat()
    require(p.is_absolute() and p.resolve(strict=True)==p and stat.S_ISREG(before.st_mode)
            and before.st_uid==os.getuid() and before.st_nlink==1
            and stat.S_IMODE(before.st_mode) in ((0o600,) if private else (0o400,0o600,0o644))
            and type(cap) is int and 0<before.st_size<=cap, 'Ordinary bounded pinned local file required.')
    identity=lambda s:(s.st_dev,s.st_ino,s.st_uid,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        require(identity(os.fstat(f.fileno()))==identity(before), 'Held local file identity changed.')
        data=f.read(cap+1)
        require(identity(os.fstat(f.fileno()))==identity(before), 'Local file changed while reading.')
    require(identity(p.lstat())==identity(before) and len(data)==before.st_size
            and (want is None or digest(data)==want), 'Pinned local bytes/identity changed.')
    return data


def load(repository):
    root=Path(repository)
    require(root.is_absolute() and root.resolve(strict=True)==root, 'Canonical original repository required.')
    raw=read(root/INPUTS,128*1024,INPUTS_SHA); c=json.loads(raw); files={}
    for name,d in c['files'].items():
        path=PurePosixPath(name)
        require(not path.is_absolute() and path.as_posix()==name and '..' not in path.parts, 'Exact relative pinned input required.')
        data=read(root/name,d['bytes'],d['sha256']); require(len(data)==d['bytes'],'Exact input extent required.')
        files[name]=data
    return c,files


def replace(raw,before,after):
    require(raw.count(before.encode())==1, 'One exact local handoff edit required.')
    start=raw.index(before.encode()); result=raw[:start]+after.encode()+raw[start+len(before.encode()):]
    return result,{'start':start,'before':before,'after':after,'original_sha256':digest(raw),'derived_sha256':digest(result)}


def assemble(repository):
    c,f=load(repository); p=c['pins']
    supervised=f[p['supervised_control']]; b=json.loads(supervised); policy=json.loads(b['policy_text'])
    cat=json.loads(policy['scratch_contract_text']); payload=json.loads(f[p['source_payload']])
    sources={n:base64.b64decode(v,validate=True) for n,v in payload.items()}
    require(len(sources)==55 and sum(map(len,sources.values()))==896039,'Exact reviewed early-device source closure required.')
    manifest=json.loads(f[p['runtime_manifest']])
    require(manifest['execution_binding'] is None and len(manifest['files'])==47
        and sum(d['bytes'] for d in manifest['files'].values())==450548,'Exact unbound preparation runtime required.')
    for name,want in manifest['files'].items():
        raw=sources[name if name.startswith('supervisor/') else 'overlay/'+name]
        require(want=={'bytes':len(raw),'sha256':digest(raw)},'Pinned full runtime bytes changed.')
    old=f[p['outer']]; tree=ast.parse(old)
    node=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='LOCAL_SOURCES' for t in n.targets))
    renames={p['outer']:OUTER,p['client']:CLIENT,p['factory']:SELF,p['prior_inputs']:INPUTS,
        p['cli']:CLI,p['old_bootstrap']:READERS['bootstrap.py'],p['old_scratch']:READERS['predictor_pgd_split_scratch_v1.py'],
        p['old_collection']:READERS['predictor_pgd_split_collection_v1.py'],
        p['old_operator']:p['preparation_operator'],p['old_operator_inputs']:p['preparation_inputs'],
        p['old_binding_design']:p['binding_design']}
    names=[renames.get(n,n) for n in ast.literal_eval(node.value)]
    names.append(p['preparation_design'])
    require(len(names)==len(set(names))==17,'Exact seventeen local preparation source pins required.')
    edits={}; outer=old
    for key,value in [('LOCAL_SOURCES',tuple(names)),('POLICY_SHA',b['policy_sha256']),
        ('BOOT_SHA',b['supervisor_sources']['bootstrap.py']['sha256']),
        ('COLLECTOR_SHA',b['supervisor_sources']['predictor_pgd_split_collection_v1.py']['sha256']),
        ('STATIC_RUNTIME',cat['runtime_files'])]:
        node=next(n for n in ast.parse(outer).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id==key for t in n.targets))
        outer,edit=replace(outer,ast.get_source_segment(outer.decode(),node),key+' = '+repr(value))
        edits.setdefault(OUTER,[]).append(edit)
    client=f[p['client']]
    for before,after in [('predictor_pgd_workload_handoff_outer_v2 as outer','predictor_pgd_workload_handoff_outer_v3 as outer'),
        ('predictor_pgd_workload_bootstrap_v1 as boot','predictor_pgd_workload_bootstrap_v2 as boot'),
        ('predictor_pgd_workload_collection_v1 as collection','predictor_pgd_workload_collection_v2 as collection')]:
        client,edit=replace(client,before,after);edits.setdefault(CLIENT,[]).append(edit)
    cli,edit=replace(f[p['cli']],'models.predictor_pgd_workload_handoff_client_v2','models.predictor_pgd_workload_handoff_client_v3')
    edits[CLI]=[edit];generated={OUTER:outer,CLIENT:client,CLI:cli}
    for name,path in READERS.items():
        raw=sources['supervisor/'+name];edits[path]=[]
        if name=='predictor_pgd_split_collection_v1.py':
            for before,after in [('predictor_pgd_split_bootstrap_v1 as boot','predictor_pgd_workload_bootstrap_v2 as boot'),
                ('predictor_pgd_split_scratch_v1 as scratch','predictor_pgd_workload_scratch_v2 as scratch')]:
                raw,edit=replace(raw,before,after);edits[path].append(edit)
        generated[path]=raw
    for name,raw in generated.items():compile(raw,'<UNINVOKED-EARLY-DEVICE-BINDING-'+name+'>','exec')
    derivation={'kind':'whole-pgd-early-device-client-binding-v1','edits':edits,
        'local_source_names':names,'source_files':55,'source_bytes':896039,'runtime_files':47,'runtime_bytes':450548,
        'scratch_prefix':'vjepa-pgd-workload-v1-','supervised_authorization_kind':'whole-pgd-supervised-university-exception-v1',
        'preparation_operator_sources_and_control_changed':False,'scientific_or_resource_clock_gates_changed':False,
        'authorization_created':False,'new_attended_scope_or_package_created':False}
    return generated,derivation,c,f


def emit_local(repository,destination):
    generated,d,c,f=assemble(repository); root=Path(repository); dest=Path(destination)
    require(not os.path.lexists(dest) and not any(os.path.lexists(root/n) for n in generated),'Fresh handoff adapter paths required.')
    dest.mkdir(mode=0o700)
    for name,data in generated.items():
        with (root/name).open('xb') as out: out.write(data)
    with (dest/'DERIVATION.json').open('xb') as out: out.write(enc(d))
    return d


def definitions(data):
    ns={'__name__':'_UNINVOKED_WHOLE_PGD_HANDOFF_DEFINITIONS'}
    exec(compile(data,'<LOCAL-WHOLE-PGD-HANDOFF-DEFINITIONS>','exec'),ns)
    return SimpleNamespace(**ns)


def material(repository,destination):
    root=Path(repository); dest=Path(destination); generated,d,c,f=assemble(root)
    for name,data in generated.items():
        require(read(root/name,len(data),digest(data))==data,'Reviewed handoff adapter changed.')
    outer=definitions(generated[OUTER]); pins={n:digest(read(root/n,2*2**20)) for n in outer.LOCAL_SOURCES}
    supervised=f[c['pins']['supervised_control']]; b=json.loads(supervised); policy=json.loads(b['policy_text'])
    cat=json.loads(policy['scratch_contract_text'])
    control={'kind':'whole-pgd-outer-control-v1','supervised_control_text':supervised.decode(),
        'supervised_control_sha256':digest(supervised),'outer_source_sha256':pins[OUTER],
        'local_source_sha256':pins,'runtime_files':cat['runtime_files'],'repository':str(root),'local_handoff':str(dest),
        'packet_path':outer.PACKET_PATH,'known_hosts_text':outer.KNOWN_HOSTS,
        'known_hosts_sha256':digest(outer.KNOWN_HOSTS.encode()),'expected_ssh_fingerprint':outer.FINGERPRINT,
        'financial_state_unchanged':True,'authorized_at_preparation':False,'target_helper_reviewed':False}
    raw=enc(control); outer.validate_control(raw,pins[OUTER])
    payload=f[c['pins']['source_payload']]; require(enc(json.loads(payload))==payload,'Canonical exact source payload required.')
    sources=outer.decode_sources(json.loads(payload),control,b)
    require(len(sources)==55 and sum(map(len,sources.values()))==896039,'Exact reviewed uploaded closure required.')
    archive=cat['packet']['archive']
    require(read(root/control['packet_path'],archive['bytes'],archive['sha256']), 'Pinned original archive required.')
    require(len(raw)+len(payload)+65536<2*2**20,'Original source frame ceiling retained.')
    package={'kind':'whole-pgd-primary-unarmed-local-package-v3','scratch_prefix':'vjepa-pgd-workload-v1-','supervised_authorization_kind':'whole-pgd-supervised-university-exception-v1','outer_control_sha256':digest(raw),
        'supervised_control_sha256':digest(supervised),'local_source_pins':len(pins),
        'source_files':55,'source_bytes':896039,'source_envelope_bytes':len(payload),
        'outer_driver_bytes':len(generated[OUTER]),'packet':cat['packet']['archive'],
        'roles':['proof','original-unit','candidate-unit','candidate-replay'],
        'method':'primary','steps':20,'restarts':2,'attack_draws':8,'scoring_draws':4,
        'step_size_255':1.,'sensitivity_banks':0,'research_calibration_selected':False,
        'full_model_input_gradient_objectives_maximum':120,
        'input_gradients_per_unit':40,'candidate_inner_math_VJPs_maximum':15360,
        'native_component_or_closed_cohort_reruns':0,'new_FP64_reference_or_full_model_trace':False,
        'authorization_created':False,'target_or_scientific_work':False,'training_admitted':False}
    values={'CONTROL-WORKLOAD.json':raw,'SOURCE-PAYLOAD.json':payload,'known-hosts':outer.KNOWN_HOSTS.encode(),'PACKAGE.json':enc(package)}
    return outer,values,package


def private_directory(path):
    p=Path(path); s=p.lstat()
    require(p.is_absolute() and p.resolve(strict=True)==p and stat.S_ISDIR(s.st_mode)
        and s.st_uid==os.getuid() and stat.S_IMODE(s.st_mode)==0o700,'Canonical private local directory required.')
    return s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode)


def prepare_unarmed(repository,destination):
    """Four private files only. No authorization, attempt, timer, claim or SSH."""
    dest=Path(destination); parent_identity=private_directory(dest.parent)
    require(not os.path.lexists(dest),'One fresh exact package path required; preserve prior work.')
    outer,values,package=material(repository,dest)
    require(private_directory(dest.parent)==parent_identity,'Package parent changed during preparation.')
    dest.mkdir(mode=0o700)
    for name,data in values.items(): outer.private_write(dest,name,data,lambda:None)
    return review_unarmed(repository,dest)


def review_unarmed(repository,destination):
    """Read/verify exact unarmed bytes, without fabricating an authorization."""
    dest=Path(destination); identity=private_directory(dest); parent=private_directory(dest.parent)
    require({p.name for p in dest.iterdir()}==FILES,'Exact four unarmed files required; no authorization or attempt may exist.')
    outer,values,package=material(repository,dest)
    for name,data in values.items():
        require(read(dest/name,len(data),digest(data),private=True)==data,'Exact bound unarmed package required.')
    require(private_directory(dest)==identity and private_directory(dest.parent)==parent,'Unarmed package ancestry changed.')
    return dict(package,unarmed_bytes_verified=True,armed=False,actual_execution_authenticated=False)
