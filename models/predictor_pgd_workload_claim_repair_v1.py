"""Derive an UNBOUND client repair from sealed bytes; no grant or launch API."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat

INPUTS='configs/predictor_pgd_workload_claim_repair_v1.json'
INPUTS_SHA='8fa77ea684410b2e355d1823a509423bdd2ec84eac444d6059f0c652da765f56'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path,cap,want):
    path=Path(path);before=path.lstat()
    if (not path.is_absolute() or path.resolve(strict=True)!=path or not stat.S_ISREG(before.st_mode)
            or before.st_uid!=os.getuid() or before.st_nlink!=1 or not 0<before.st_size<=cap):
        raise ValueError('Exact owned regular source required.')
    identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as source:
        if identity(os.fstat(source.fileno()))!=identity(before):raise ValueError('Source identity changed.')
        raw=source.read(cap+1)
        if identity(os.fstat(source.fileno()))!=identity(before):raise ValueError('Source changed while reading.')
    if identity(path.lstat())!=identity(before) or len(raw)!=before.st_size or digest(raw)!=want:
        raise ValueError('Pinned source bytes changed.')
    return raw


def derive(repository):
    root=Path(repository);config=json.loads(read(root/INPUTS,65536,INPUTS_SHA));sources={}
    for name,d in config['source_files'].items():sources[name]=read(root/name,d['bytes'],d['sha256'])
    original=sources[config['client_source']];raw=original;edits=[]
    replacements=[
        ("scratch.name.removeprefix('vjepa-pgd-candidate-v1-')", "scratch.name.removeprefix('vjepa-pgd-workload-v1-')"),
        ("scratch.name != 'vjepa-pgd-candidate-v1-' + stem", "scratch.name != 'vjepa-pgd-workload-v1-' + stem"),
        ("        claimed = outer.frame(incoming, alarms.check)\n", "        claimed = outer.frame(incoming, alarms.check)\n        outer.private_write(session, 'CLAIM-OBSERVED.json', outer.encoded(claimed), alarms.check)\n"),
    ]
    for before,after in replacements:
        old=before.encode();new=after.encode()
        if raw.count(old)!=1:raise ValueError('One exact claim repair span required.')
        start=raw.index(old);prior=digest(raw);raw=raw[:start]+new+raw[start+len(old):]
        edits.append({'start':start,'before':before,'after':after,'original_sha256':prior,'derived_sha256':digest(raw)})
    ast.parse(raw)
    inverse=raw
    for edit in reversed(edits):
        start=edit['start'];after=edit['after'].encode()
        if inverse[start:start+len(after)]!=after:raise ValueError('Exact inverse span required.')
        inverse=inverse[:start]+edit['before'].encode()+inverse[start+len(after):]
    if inverse!=original:raise ValueError('All other client bytes must remain exact.')
    server=ast.parse(sources[config['server_source']])
    remote_root=next(n for n in server.body if isinstance(n,ast.FunctionDef) and n.name=='remote_root')
    prefixes=[n.value for n in ast.walk(remote_root) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('vjepa-pgd-')]
    if prefixes!=[config['server_and_repaired_client_prefix']]:raise ValueError('Pinned server namespace changed.')
    return raw,{'kind':'whole-pgd-claim-unbound-client-repair-v1','edits':edits,'original_client_sha256':digest(original),
        'derived_client_sha256':digest(raw),'old_sources_changed':False,'server_sources_changed':False,
        'scientific_or_resource_clock_gates_changed':False,'unbound':True,'new_scope_grant_or_target_work':False}


def emit_unbound(repository,destination):
    """Fresh local source artifact only; no new outer, package, CLI or live binding."""
    raw,record=derive(repository);dest=Path(destination)
    if os.path.lexists(dest):raise ValueError('Preserve prior artifacts; use a fresh destination.')
    dest.mkdir(mode=0o700)
    values={'UNBOUND-CLIENT.py':raw,'DERIVATION.json':(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()}
    for name,data in values.items():
        with os.fdopen(os.open(dest/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:out.write(data)
    return record
