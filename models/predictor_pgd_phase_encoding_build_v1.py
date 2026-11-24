"""Cold source-only phase-encoding prototype; no operational graph or launch API."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat

INPUTS = 'configs/predictor_pgd_phase_encoding_inputs_v1.json'
INPUTS_SHA = '8378ec45463373002c2942fac45aa1b5975638ccfe7652a5cad9cf2349f25935'  # Frozen before first emission/check; no discovery fallback.
EXECUTION_BINDING = None


def enc(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, size, digest):
    before = path.lstat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_uid, s.st_gid, s.st_mode,
        s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
            or before.st_nlink != 1 or not 0 < before.st_size <= size
            or stat.S_IMODE(before.st_mode) not in (0o400, 0o600, 0o644)):
        raise ValueError('Ordinary owned pinned source input required; never open special files.')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as source:
        if signature(os.fstat(source.fileno())) != signature(before):
            raise ValueError('Held source input changed.')
        raw = source.read(size+1)
        if signature(os.fstat(source.fileno())) != signature(before):
            raise ValueError('Pinned input changed during read.')
    if signature(path.lstat()) != signature(before) or len(raw) != before.st_size or sha(raw) != digest:
        raise ValueError('Pinned source bytes or identity changed.')
    return raw


def assemble(repository):
    root = Path(repository)
    if EXECUTION_BINDING is not None or not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError('Canonical repository and unbound source-only prototype required.')
    if INPUTS_SHA is None:
        raise ValueError('Frozen prototype input identity required.')
    config_raw = read(root/INPUTS, 128*1024, INPUTS_SHA)
    config = json.loads(config_raw)
    files = {}
    for name, row in config['files'].items():
        if Path(name).is_absolute() or Path(name).as_posix() != name or '..' in Path(name).parts:
            raise ValueError('Exact relative source pin required.')
        files[name] = read(root/name, row['bytes'], row['sha256'])
    baseline = files[config['pins']['baseline']]
    codec = files[config['pins']['codec']]
    codec_tree = ast.parse(codec)
    functions = [n for n in codec_tree.body if isinstance(n, ast.FunctionDef)]
    if [n.name for n in functions] != ['_phase_json_value_end', 'insert_row_sha256']:
        raise ValueError('Exact two stdlib byte helpers required.')
    original_names = {n.name for n in ast.parse(baseline).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if original_names.intersection(n.name for n in functions):
        raise ValueError('No helper shadowing in cold candidate.')
    before = b"        row['row_sha256'] = hashlib.sha256(enc(row)).hexdigest()\n        raw = enc(row)"
    after = b"        prehash = enc(row)\n        row['row_sha256'] = hashlib.sha256(prehash).hexdigest()\n        raw = insert_row_sha256(prehash, row['row_sha256'])"
    if baseline.count(before) != 1:
        raise ValueError('One exact phase-record encoding seam required.')
    at = baseline.index(before)
    candidate = baseline[:at]+after+baseline[at+len(before):]
    helper_source = '\n\n'+'\n\n\n'.join(ast.get_source_segment(codec.decode(), node) for node in functions)+'\n'
    edits = [{'start':at,'before':before.decode(),'after':after.decode()},
             {'start':len(candidate),'before':'','after':helper_source}]
    candidate += helper_source.encode()
    compile(candidate, '<UNBOUND-PHASE-ENCODING-DEFINITIONS-ONLY>', 'exec')
    result = {'kind':'whole-pgd-phase-encoding-cold-source-derivation-v1',
        'execution_binding':None,'source_definitions_only':True,
        'existing_runtime_control_client_or_budget_changed':False,
        'new_operational_graph_package_or_grant_emitted':False,
        'baseline':{'bytes':len(baseline),'sha256':sha(baseline)},
        'candidate':{'bytes':len(candidate),'sha256':sha(candidate)},
        'codec':{'bytes':len(codec),'sha256':sha(codec)},
        'edits':edits,'input_config_sha256':sha(config_raw),'input_pins':len(files),
        'performance_or_budget_fit_qualified':False,'research_admitted':False,'training_admitted':False}
    return baseline, candidate, codec, result


def emit_local(repository, destination):
    baseline, candidate, codec, result = assemble(repository)
    dest = Path(destination)
    if os.path.lexists(dest):
        raise ValueError('Fresh source-only destination required; retain prior versions.')
    dest.mkdir(mode=0o700)
    for name, raw in [('baseline.py',baseline),('candidate.py',candidate),('codec.py',codec),('DERIVATION.json',enc(result))]:
        with (dest/name).open('xb') as target:
            os.fchmod(target.fileno(),0o600)
            target.write(raw)
    return result
