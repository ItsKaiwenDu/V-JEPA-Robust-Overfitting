"""Emit separate local client/readers; no scope, package, grant or launch API."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

INPUTS = 'configs/predictor_pgd_workload_shutdown_binding_inputs_v1.json'
INPUTS_SHA = '60be53c0bc355c4ec6fc1d4a2062d0014fcecfaa4cf2b41abef3ec586f396590'
SELF = 'models/predictor_pgd_workload_shutdown_binding_v1.py'
OUTER = 'models/predictor_pgd_workload_handoff_outer_v4.py'
CLIENT = 'models/predictor_pgd_workload_handoff_client_v4.py'
CLI = 'scripts/run_predictor_pgd_workload_handoff_v4.py'
READERS = {'bootstrap.py': 'models/predictor_pgd_workload_bootstrap_v3.py',
    'predictor_pgd_split_scratch_v1.py': 'models/predictor_pgd_workload_scratch_v3.py',
    'predictor_pgd_split_collection_v1.py': 'models/predictor_pgd_workload_collection_v3.py'}
BUDGET = 'models/predictor_pgd_workload_budget_v1.py'
LOCAL_SOURCES = (OUTER, CLIENT, CLI, SELF, INPUTS, *READERS.values(),
    'configs/predictor_pgd_workload_shutdown_binding_v1.json', BUDGET,
    'configs/predictor_pgd_workload_budget_v1.json',
    'models/predictor_pgd_workload_shutdown_v1.py',
    'models/predictor_pgd_workload_shutdown_logic_v1.py',
    'configs/predictor_pgd_workload_shutdown_inputs_v1.json',
    'configs/predictor_pgd_workload_shutdown_v1.json',
    'models/predictor_pgd_workload_os_v1.py', 'models/predictor_pgd_split_plan_v1.py',
    'models/__init__.py')


def enc(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path, cap, wanted):
    p = Path(path)
    before = p.lstat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_uid, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(p.is_absolute() and p.resolve(strict=True) == p and stat.S_ISREG(before.st_mode)
        and before.st_uid == os.getuid() and before.st_nlink == 1
        and stat.S_IMODE(before.st_mode) in (0o400, 0o600, 0o644)
        and 0 < before.st_size <= cap, 'Ordinary pinned local binding input required.')
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as handle:
        require(signature(os.fstat(handle.fileno())) == signature(before), 'Held binding input changed.')
        raw = handle.read(cap+1)
        require(signature(os.fstat(handle.fileno())) == signature(before), 'Binding input changed during read.')
    require(signature(p.lstat()) == signature(before) and len(raw) == before.st_size
        and digest(raw) == wanted, 'Pinned binding bytes or identity changed.')
    return raw


def load(repository):
    root = Path(repository)
    require(root.is_absolute() and root.resolve(strict=True) == root, 'Canonical original repository required.')
    raw = read(root/INPUTS, 128*1024, INPUTS_SHA)
    cfg = json.loads(raw)
    files = {}
    for name, descriptor in cfg['files'].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and path.as_posix() == name and '..' not in path.parts, 'Exact relative pin required.')
        files[name] = read(root/name, descriptor['bytes'], descriptor['sha256'])
    return cfg, files


def replace(raw, before, after):
    require(raw.count(before.encode()) == 1, 'One exact client binding edit required.')
    start = raw.index(before.encode())
    result = raw[:start]+after.encode()+raw[start+len(before.encode()):]
    return result, {'start': start, 'before': before, 'after': after}


def assemble(repository):
    cfg, files = load(repository)
    pins = cfg['pins']
    control = json.loads(files[pins['control']])
    policy = json.loads(control['policy_text'])
    catalogue = json.loads(policy['scratch_contract_text'])
    payload = json.loads(files[pins['payload']])
    sources = {name: base64.b64decode(value, validate=True) for name, value in payload.items()}
    require(len(sources) == 55 and sum(map(len, sources.values())) == 905563, 'Exact reviewed shutdown closure required.')
    manifest = json.loads(files[pins['manifest']])
    require(manifest['execution_binding'] is None and len(manifest['files']) == 47
        and sum(row['bytes'] for row in manifest['files'].values()) == 457112, 'Exact unbound shutdown runtime required.')
    for name, wanted in manifest['files'].items():
        raw = sources[name if name.startswith('supervisor/') else 'overlay/'+name]
        require(wanted == {'bytes': len(raw), 'sha256': digest(raw)}, 'Reviewed runtime drift.')
    require(len(set(LOCAL_SOURCES)) == len(LOCAL_SOURCES) == 18, 'Exact eighteen local pins required.')
    edits = {}
    outer = files[pins['outer']]
    for key, value in [('LOCAL_SOURCES', LOCAL_SOURCES), ('POLICY_SHA', control['policy_sha256']),
        ('BOOT_SHA', control['supervisor_sources']['bootstrap.py']['sha256']),
        ('COLLECTOR_SHA', control['supervisor_sources']['predictor_pgd_split_collection_v1.py']['sha256']),
        ('STATIC_RUNTIME', catalogue['runtime_files'])]:
        node = next(n for n in ast.parse(outer).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == key for t in n.targets))
        outer, edit = replace(outer, ast.get_source_segment(outer.decode(), node), key+' = '+repr(value))
        edits.setdefault(OUTER, []).append(edit)
    client = files[pins['client']]
    for before, after in [('predictor_pgd_workload_handoff_outer_v3 as outer', 'predictor_pgd_workload_handoff_outer_v4 as outer'),
        ('predictor_pgd_workload_bootstrap_v2 as boot', 'predictor_pgd_workload_bootstrap_v3 as boot'),
        ('predictor_pgd_workload_collection_v2 as collection', 'predictor_pgd_workload_collection_v3 as collection')]:
        client, edit = replace(client, before, after)
        edits.setdefault(CLIENT, []).append(edit)
    suffix = '''

# A positive fake budget port is injected only into private test namespaces.
# The installed source always imports the closed, reviewed budget version.
from models import predictor_pgd_workload_budget_v1 as budget
_budget_unqualified_protocol = run
def run(folder):
    budget.require_execution()
    return _budget_unqualified_protocol(folder)
'''
    edits[CLIENT].append({'start': len(client), 'before': '', 'after': suffix})
    client += suffix.encode()
    cli, edit = replace(files[pins['cli']], 'models.predictor_pgd_workload_handoff_client_v3', 'models.predictor_pgd_workload_handoff_client_v4')
    edits[CLI] = [edit]
    generated = {OUTER: outer, CLIENT: client, CLI: cli}
    for name, path in READERS.items():
        raw = sources['supervisor/'+name]
        edits[path] = []
        if name == 'predictor_pgd_split_collection_v1.py':
            for before, after in [('predictor_pgd_split_bootstrap_v1 as boot', 'predictor_pgd_workload_bootstrap_v3 as boot'),
                ('predictor_pgd_split_scratch_v1 as scratch', 'predictor_pgd_workload_scratch_v3 as scratch')]:
                raw, edit = replace(raw, before, after)
                edits[path].append(edit)
        generated[path] = raw
    for name, raw in generated.items():
        compile(raw, '<UNINVOKED-SHUTDOWN-BINDING-'+name+'>', 'exec')
    derivation = {'kind': 'whole-pgd-shutdown-client-binding-v1', 'edits': edits,
        'local_source_names': list(LOCAL_SOURCES), 'source_files': 55, 'source_bytes': 905563,
        'runtime_files': 47, 'runtime_bytes': 457112, 'source_control_or_scientific_caps_changed': False,
        'budget_gate': 'unconditionally closed before any protocol clock or attempt',
        'new_scope_package_authorization_or_target_work': False}
    return generated, derivation, cfg, files


def emit_local(repository, destination):
    root = Path(repository)
    generated, derivation, _, _ = assemble(root)
    dest = Path(destination)
    require(not os.path.lexists(dest) and not any(os.path.lexists(root/name) for name in generated), 'Fresh separate binding definitions required.')
    dest.mkdir(mode=0o700)
    for name, raw in generated.items():
        with (root/name).open('xb') as handle:
            handle.write(raw)
    with (dest/'DERIVATION.json').open('xb') as handle:
        handle.write(enc(derivation))
    return derivation
