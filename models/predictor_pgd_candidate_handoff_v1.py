"""Prepare and inspect one unarmed candidate packet; never authorizes or launches.

The reviewed 45 uploaded sources and supervised control are copied exactly.
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

INPUTS = 'configs/predictor_pgd_candidate_handoff_inputs_v1.json'
INPUTS_SHA = '04794823394e5306e85cf3b6580f1c2585c7d188ef6fb4e827c037c1ab4d2e3a'
OUTER = 'models/predictor_pgd_candidate_handoff_outer_v1.py'
CLIENT = 'models/predictor_pgd_candidate_handoff_client_v1.py'
CLI = 'scripts/run_predictor_pgd_candidate_v1.py'
SELF = 'models/predictor_pgd_candidate_handoff_v1.py'
FILES = frozenset(('CONTROL-CANDIDATE.json', 'SOURCE-PAYLOAD.json', 'known-hosts', 'PACKAGE.json'))


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
    c,f=load(repository); p=c['pins']; old=f[p['outer']]
    tree=ast.parse(old); node=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='LOCAL_SOURCES' for t in n.targets))
    names=list(ast.literal_eval(node.value)); names[:2]=[OUTER,CLIENT]; names.extend((SELF,INPUTS,CLI))
    require(len(names)==len(set(names))==12,'Exact twelve local source pins required.')
    outer,outer_edit=replace(old,ast.get_source_segment(old.decode(),node),'LOCAL_SOURCES = '+repr(tuple(names)))
    client,client_edit=replace(f[p['client']],'predictor_pgd_candidate_outer_v1 as outer',
        'predictor_pgd_candidate_handoff_outer_v1 as outer')
    cli,cli_edit=replace(f[p['cli_scaffold']],'models.predictor_pgd_component_client_v1',
        'models.predictor_pgd_candidate_handoff_client_v1')
    generated={OUTER:outer,CLIENT:client,CLI:cli}
    for name,data in generated.items(): compile(data,'<UNINVOKED-CANDIDATE-HANDOFF-'+name+'>','exec')
    derivation={'kind':'candidate-only-local-handoff-source-v1','edits':{OUTER:outer_edit,CLIENT:client_edit,CLI:cli_edit},
        'local_source_names':names,'uploaded_sources_changed':False,'supervised_control_changed':False,
        'target_source_or_resource_clock_gates_changed':False,'authorization_created':False}
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
    ns={'__name__':'_UNINVOKED_CANDIDATE_HANDOFF_DEFINITIONS'}
    exec(compile(data,'<LOCAL-CANDIDATE-HANDOFF-DEFINITIONS>','exec'),ns)
    return SimpleNamespace(**ns)


def material(repository,destination):
    root=Path(repository); dest=Path(destination); generated,d,c,f=assemble(root)
    for name,data in generated.items():
        require(read(root/name,len(data),digest(data))==data,'Reviewed handoff adapter changed.')
    outer=definitions(generated[OUTER]); pins={n:digest(read(root/n,2*2**20)) for n in outer.LOCAL_SOURCES}
    supervised=f[c['pins']['supervised_control']]; b=json.loads(supervised); policy=json.loads(b['policy_text'])
    cat=json.loads(policy['scratch_contract_text'])
    control={'kind':'pgd-candidate-outer-control-v1','supervised_control_text':supervised.decode(),
        'supervised_control_sha256':digest(supervised),'outer_source_sha256':pins[OUTER],
        'local_source_sha256':pins,'runtime_files':cat['runtime_files'],'repository':str(root),'local_handoff':str(dest),
        'packet_path':outer.PACKET_PATH,'reference_path':outer.REFERENCE_PATH,
        'reference_manifest_sha256':outer.REFERENCE_MANIFEST_SHA,'known_hosts_text':outer.KNOWN_HOSTS,
        'known_hosts_sha256':digest(outer.KNOWN_HOSTS.encode()),'expected_ssh_fingerprint':outer.FINGERPRINT,
        'financial_state_unchanged':True,'authorized_at_preparation':False,'target_helper_reviewed':False}
    raw=enc(control); outer.validate_control(raw,pins[OUTER])
    payload=f[c['pins']['source_payload']]; require(enc(json.loads(payload))==payload,'Canonical exact source payload required.')
    sources=outer.decode_sources(json.loads(payload),control,b)
    require(len(sources)==45 and sum(map(len,sources.values()))==754567,'Exact reviewed uploaded closure required.')
    for name,key in [('packet_path','packet'),('reference_path','reference')]:
        archive=cat[key]['archive']; require(read(root/control[name],archive['bytes'],archive['sha256']), 'Pinned archive required.')
    require(len(raw)+len(payload)+65536<2*2**20,'Original source frame ceiling retained.')
    package={'kind':'candidate-only-unarmed-local-package-v1','outer_control_sha256':digest(raw),
        'supervised_control_sha256':digest(supervised),'local_source_pins':len(pins),
        'source_files':45,'source_bytes':754567,'source_envelope_bytes':len(payload),
        'outer_driver_bytes':len(generated[OUTER]),'capsule':cat['packet']['archive'],
        'native_reference':cat['reference']['archive'],'reference_manifest_sha256':outer.REFERENCE_MANIFEST_SHA,
        'roles':['proof','forward-preserving'],'new_full_model_objectives':0,'native_or_completed_cohort_repeats':0,
        'authorization_created':False,'target_or_scientific_work':False,'training_admitted':False}
    values={'CONTROL-CANDIDATE.json':raw,'SOURCE-PAYLOAD.json':payload,'known-hosts':outer.KNOWN_HOSTS.encode(),'PACKAGE.json':enc(package)}
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
