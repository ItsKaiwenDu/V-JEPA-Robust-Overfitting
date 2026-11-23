"""Cold candidate-only supervisor integration; no handoff, grant or execution.

Only pinned bytes are read. Emitted OS/client definitions remain uninvoked here.
The worker closure, owned monitor, proof recipe and scientific providers are copied
exactly; every supervisor/client edit has a complete, checked byte inverse.
"""
import ast
import base64
import hashlib
import json
from pathlib import Path
import stat

INPUTS = 'configs/predictor_pgd_candidate_operator_inputs_v2.json'
INPUTS_SHA = '54ccc4bb05f54adc45e14829221543aee4d8ab6166700b4a792ea10f2b2b36e4'
LIVE_EXECUTION_BINDING = None
ROLES = ('proof', 'forward-preserving')
SECONDS = {'proof': 60, 'forward-preserving': 120}
NAMES = ('bootstrap.py', 'predictor_pgd_split_scratch_v1.py',
         'predictor_pgd_split_coordinator_v1.py', 'predictor_pgd_split_collection_v1.py',
         'predictor_pgd_split_watch_v1.py', 'scratch_allocation_v2.py')
OUTER = 'models/predictor_pgd_candidate_outer_v1.py'
CLIENT = 'models/predictor_pgd_candidate_client_v1.py'
READERS = {'bootstrap.py': 'models/predictor_pgd_candidate_bootstrap_v1.py',
           'predictor_pgd_split_scratch_v1.py': 'models/predictor_pgd_candidate_scratch_v1.py',
           'predictor_pgd_split_collection_v1.py': 'models/predictor_pgd_candidate_collection_v1.py'}
KINDS = {
    'pgd-component-supervised-control-v1': 'pgd-candidate-supervised-control-v1',
    'pgd-component-supervised-university-exception-v1': 'pgd-candidate-supervised-university-exception-v1',
    'pgd-component-parent-session-v1': 'pgd-candidate-parent-session-v1',
    'pgd-component-child-launch-v1': 'pgd-candidate-child-launch-v1',
    'pgd-component-work-receipt-v1': 'pgd-candidate-work-receipt-v1',
    'pgd-component-evidence-stream-v1': 'pgd-candidate-evidence-stream-v1',
    'pgd-component-outer-control-v1': 'pgd-candidate-outer-control-v1',
    'pgd-component-outer-university-exception-v1': 'pgd-candidate-outer-university-exception-v1',
    'component-only-runtime-policy-v1': 'candidate-only-runtime-policy-v1',
    'component-only-whole-scratch-v1': 'candidate-only-whole-scratch-v1',
    'vjepa-pgd-component-v1-': 'vjepa-pgd-candidate-v1-',
    'All three component-only roles must finish before a complete-work receipt.':
        'Both proof and candidate roles must finish before a complete-work receipt.',
}


def enc(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def inverse(raw, edits):
    for edit in reversed(edits):
        start = edit['start']; after = edit['after'].encode()
        require(raw[start:start + len(after)] == after, 'Exact candidate supervisor inverse required.')
        raw = raw[:start] + edit['before'].encode() + raw[start + len(after):]
    return raw


class Editor:
    def __init__(self, raw):
        self.original = self.raw = raw
        self.edits = []

    def one(self, before, after):
        before = before.encode(); after = after.encode()
        require(self.raw.count(before) == 1, 'One exact integration edit required: ' + before[:100].decode())
        start = self.raw.index(before)
        self.edits.append({'start': start, 'before': before.decode(), 'after': after.decode()})
        self.raw = self.raw[:start] + after + self.raw[start + len(before):]

    def all(self, before, after):
        # One recorded span per occurrence, including embedded driver literals.
        require(before not in after, 'Integration substitutions must terminate.')
        while before.encode() in self.raw:
            start = self.raw.index(before.encode()); end = start + len(before.encode())
            self.edits.append({'start': start, 'before': before, 'after': after})
            self.raw = self.raw[:start] + after.encode() + self.raw[end:]

    def node(self, name, replacement, *, assignment=False):
        tree = ast.parse(self.raw)
        nodes = [n for n in tree.body if
                 isinstance(n, ast.Assign) and assignment and
                 any(isinstance(t, ast.Name) and t.id == name for t in n.targets) or
                 isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not assignment and n.name == name]
        require(len(nodes) == 1, 'One top-level integration node required: ' + name)
        self.one(ast.get_source_segment(self.raw.decode(), nodes[0]), replacement)

    def append(self, text):
        start = len(self.raw)
        self.edits.append({'start': start, 'before': '', 'after': text})
        self.raw += text.encode()

    def finish(self):
        for before, after in KINDS.items():
            self.all(before, after)
        require(inverse(self.raw, self.edits) == self.original, 'Complete original byte reconstruction required.')
        compile(self.raw, '<UNINVOKED-CANDIDATE-SUPERVISOR>', 'exec')
        return self.raw, {'original_sha256': digest(self.original), 'derived_sha256': digest(self.raw), 'edits': self.edits}


def assigned(raw, name):
    return ast.literal_eval(next(n.value for n in ast.parse(raw).body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))


def inputs(repository):
    root = Path(repository); raw = (root / INPUTS).read_bytes()
    require(digest(raw) == INPUTS_SHA, 'Exact candidate supervisor input contract required.')
    contract = json.loads(raw); files = {}
    for name, desc in contract['files'].items():
        path = root / name
        require(stat.S_ISREG(path.lstat().st_mode), 'Never open a special integration input.')
        value = path.read_bytes()
        require(len(value) == desc['bytes'] and digest(value) == desc['sha256'], 'Pinned integration input changed: ' + name)
        files[name] = value
    return contract, files


STATIC_CONTRACT = '''def static_contract(supervisors):
    assets = dict(STATIC['assets'])
    for name, d in supervisors.items():
        assets['supervisor/' + name] = dict(d)
    return {'kind': 'candidate-only-whole-scratch-v1', 'assets': assets,
            'runtime_files': STATIC['runtime_files'], 'result': STATIC['result'],
            'root_metadata': STATIC['root_metadata'], 'caps': STATIC['caps'],
            'packet': STATIC['packet'], 'reference': STATIC['reference']}
'''

DIRECTORIES = '''def _setup_directory_plan(layout):
    if len(layout.assets) != 105 or len(layout.result) != 17:
        raise ValueError('Exact closed candidate source/reference/result catalogue required.')
    selected = {'.', 'capture', 'capture/logs', 'capture/results', 'capture/results/components',
                'cache', 'cache/cuda', 'tmp', 'supervisor', 'configs', 'source', 'reference', 'overlay'}
    for name in layout.assets:
        selected.update(str(p) for p in PurePosixPath(name).parents)
    claims = {'forward-preserving': 'capture/results/components/forward-preserving'}
    if any(n == p or n.startswith(p + '/') for n in selected for p in claims.values()):
        raise ValueError('Candidate output directory belongs only to its worker.')
    for name in selected:
        layout.directory(name)
    return {'kind': 'candidate-only-minimal-directory-plan-v1',
            'setup_directories': sorted(selected, key=lambda n: (len(PurePosixPath(n).parts), n)),
            'worker_claim_directories': claims, 'authorized': False,
            'scientific_qualified': False, 'training_admitted': False}
'''


def assemble(repository):
    require(LIVE_EXECUTION_BINDING is None, 'This cold builder has no live binding.')
    c, f = inputs(repository); pins = c['pins']
    manifest = json.loads(f[pins['runtime_manifest']]); spec_raw = f[pins['reference_manifest']]
    spec = json.loads(spec_raw); runtime = manifest['files']
    require(len(runtime) == 39 and manifest['execution_binding'] is None, 'Reviewed null 39-module closure required.')
    workers = {name: f[c['runtime'][name]] for name in runtime}
    require(all(digest(workers[n]) == d['sha256'] and len(workers[n]) == d['bytes'] for n, d in runtime.items()), 'Candidate worker closure changed.')
    payload = {n: base64.b64decode(v, validate=True) for n, v in json.loads(f[pins['old_source_payload']]).items()}
    old = {n: payload['supervisor/' + n] for n in NAMES}
    static = assigned(f[pins['old_bootstrap']], 'STATIC')
    static['runtime_files'] = runtime
    static['assets'] = {n: d for n, d in static['assets'].items() if not n.startswith('overlay/')}
    static['assets'].update({'overlay/' + n: d for n, d in runtime.items()})
    ref_members = dict(spec['files'])
    ref_members['MANIFEST.json'] = {'bytes': len(spec_raw), 'sha256': digest(spec_raw)}
    # The archive's MANIFEST is the canonical encoding, not the pretty saved index.
    canonical_spec = enc(spec)
    ref_members['MANIFEST.json'] = {'bytes': len(canonical_spec), 'sha256': digest(canonical_spec)}
    reference = {'archive': c['files'][pins['reference_archive']], 'files': ref_members,
                 'manifest_sha256': digest(canonical_spec)}
    static['reference'] = reference
    static['assets']['native-reference.tar'] = reference['archive']
    static['assets'].update({'reference/' + n: d for n, d in ref_members.items()})
    static['result'] = {n: d for n, d in static['result'].items()
                        if n.startswith('results/components/forward-preserving/') or n in
                        ('logs/forward-preserving.log', 'results/final-receipt.json', 'results/collection-status.json')}
    static['result']['results/components/forward-preserving/accuracy.json'] = {
        'category': 'metadata', 'max_bytes': 16 * 1024, 'mutable': False, 'partial_allowed': True}
    for name in ('worker-plan-native.json', 'parent-native.json'):
        static['root_metadata'].pop(name)
    static['scope'].update(kind='candidate-only-source-interface-not-a-grant',
                           components=['forward-preserving'], native_component_reruns=0,
                           reference_manifest_sha256=reference['manifest_sha256'])
    static['source_hashes'] = dict(assigned(old[NAMES[2]], 'SOURCE_HASHES'),
                                  reference_manifest_sha256=reference['manifest_sha256'])
    static['reference_manifest_sha256'] = reference['manifest_sha256']
    require(len(static['assets']) == 98 and len(static['result']) == 17, 'Closed candidate catalogue extent changed.')
    sources = {}; derivation = {}

    def finish(name, editor):
        sources[name], derivation[name] = editor.finish()

    e = Editor(old[NAMES[0]])
    for name, value in [('ROLES', ROLES), ('SECONDS', SECONDS), ('STATIC', static)]:
        e.node(name, name + ' = ' + repr(value), assignment=True)
    e.node('static_contract', STATIC_CONTRACT.rstrip())
    e.one('len(mapping) != 43', 'len(mapping) != 45')
    e.one('Exact37component runtime and six supervisor names required.', 'Exact39candidate runtime and six supervisor names required.')
    old_check = next(n for n in ast.walk(ast.parse(e.raw)) if isinstance(n, ast.If)
                     and 'Exact immutable numerical source hashes required.' in ast.unparse(n))
    e.one(ast.get_source_segment(e.raw.decode(), old_check),
          "expected = STATIC['source_hashes'] if plan['role'] == 'forward-preserving' else {}\n"
          "    if type(plan['source_hashes']) is not dict or plan['source_hashes'] != expected:\n"
          "        raise ValueError('Exact immutable candidate source/reference hashes required.')")
    finish(NAMES[0], e)

    e = Editor(old[NAMES[1]])
    e.node('STATIC', 'STATIC = ' + repr(static), assignment=True)
    e.node('static_contract', STATIC_CONTRACT.rstrip())
    # Only catalogue classification changes. Scanner/assessment/Guard are exact.
    e.one("partial and plain == 'component-inputs.tar'", "partial and plain in ('component-inputs.tar', 'native-reference.tar')")
    e.one("plain == 'component-inputs.tar' or plain.startswith('source/')", "plain in ('component-inputs.tar', 'native-reference.tar') or plain.startswith(('source/', 'reference/'))")
    e.one("partial_allowed=plain == 'component-inputs.tar'", "partial_allowed=plain in ('component-inputs.tar', 'native-reference.tar')")
    finish(NAMES[1], e)

    e = Editor(old[NAMES[2]])
    e.node('SOURCE_HASHES', 'SOURCE_HASHES = ' + repr(static['source_hashes']), assignment=True)
    e.all("role in ('native', 'forward-preserving')", "role == 'forward-preserving'")
    e.one("plan['role'] not in ('native', 'forward-preserving')", "plan['role'] != 'forward-preserving'")
    e.one("'kind': 'component-only-reviewed-execution-binding-v1'", "'kind': 'candidate-only-reviewed-execution-binding-v1'")
    e.one("'result_root': str(root / 'capture'), 'host': HOST['hostname'], 'capsule_manifest_sha256': boot.STATIC['capsule_manifest_sha256']",
          "'result_root': str(root / 'capture'), 'reference_root': str(root / 'reference'), 'host': HOST['hostname'], 'capsule_manifest_sha256': boot.STATIC['capsule_manifest_sha256'], 'reference_manifest_sha256': boot.STATIC['reference_manifest_sha256']")
    e.one("Namespace(capsule_root=str(root / 'source'), result_root=str(root / 'capture'), control_sha256=csha",
          "Namespace(capsule_root=str(root / 'source'), result_root=str(root / 'capture'), reference_root=str(root / 'reference'), control_sha256=csha")
    e.one("'probe_index': int(role == 'forward-preserving')", "'probe_index': 1")
    e.one("        else:\n            from models import _pgd_component_io_v1 as io",
          "        else:\n            if role != 'forward-preserving' or self.sources != SOURCE_HASHES or plan['source_hashes'] != SOURCE_HASHES:\n                raise ValueError('Fresh candidate source/reference plan required.')\n            g.verify_assets(phase='capture')\n            from models import _pgd_component_io_v1 as io")
    finish(NAMES[2], e)
    e = Editor(old[NAMES[3]]); finish(NAMES[3], e)
    for name in NAMES[4:]:
        sources[name] = old[name]
        derivation[name] = {'original_sha256': digest(old[name]), 'derived_sha256': digest(old[name]), 'edits': []}
    desc = {n: {'bytes': len(v), 'sha256': digest(v)} for n, v in sources.items()}
    catalogue = {'kind': 'candidate-only-whole-scratch-v1', 'assets': dict(static['assets'],
                 **{'supervisor/' + n: d for n, d in desc.items()}), 'runtime_files': runtime,
                 'result': static['result'], 'root_metadata': static['root_metadata'],
                 'caps': static['caps'], 'packet': static['packet'], 'reference': reference}
    policy = {'kind': 'candidate-only-runtime-policy-v1', 'scratch_contract_text': enc(catalogue).decode(),
              'base_text': static['base_text'], 'design_text': static['design_text'], 'scope': static['scope']}
    control = enc({'kind': 'pgd-candidate-supervised-control-v1', 'policy_text': enc(policy).decode(),
                   'policy_sha256': digest(enc(policy)), 'supervisor_sources': desc})

    e = Editor(f[pins['old_outer']])
    local_sources = (OUTER, CLIENT, 'models/predictor_pgd_candidate_operator_v1.py', INPUTS,
                     *READERS.values(), 'models/__init__.py', 'models/predictor_pgd_split_plan_v1.py')
    for name, value in [('POLICY_SHA', digest(enc(policy))), ('BOOT_SHA', desc[NAMES[0]]['sha256']),
                        ('COLLECTOR_SHA', desc[NAMES[3]]['sha256']), ('LOCAL_SOURCES', local_sources),
                        ('STATIC_RUNTIME', runtime), ('_DIRECTORY_WORKER_CLAIMS',
                         {'forward-preserving': 'capture/results/components/forward-preserving'})]:
        e.node(name, name + ' = ' + repr(value), assignment=True)
    e.one("'kind', 'supervised_control_text'", "'reference_path', 'reference_manifest_sha256', 'kind', 'supervised_control_text'")
    e.one("c['packet_path'] != PACKET_PATH", "c['packet_path'] != PACKET_PATH or c['reference_path'] != REFERENCE_PATH or c['reference_manifest_sha256'] != REFERENCE_MANIFEST_SHA")
    e.one("len(c['runtime_files']) != 37", "len(c['runtime_files']) != 39")
    e.one("'extra_component_VJPs': 2", "'extra_component_VJPs': 1")
    e.one('len(mapping) != 43', 'len(mapping) != 45')
    e.one('All 69 exact source files required.', 'All 45 exact candidate and supervisor source files required.')
    e.node('_setup_directory_plan', DIRECTORIES.rstrip())
    # Two fixed archives on the same stdin: only the second consumes final EOF.
    eof = "    if incoming.read(1) != b'':\n        raise ValueError('Unexpected bytes after the single approved packet.')\n"
    packet_node = next(n for n in ast.parse(e.raw).body if isinstance(n, ast.FunctionDef) and n.name == 'receive_packet')
    original_packet = ast.get_source_segment(e.raw.decode(), packet_node)
    e.one(eof, '')
    receive_ref = original_packet.replace('receive_packet(', 'receive_reference(').replace('27279360', str(reference['archive']['bytes'])).replace(static['packet']['archive']['sha256'], reference['archive']['sha256']).replace('component-inputs.tar', 'native-reference.tar').replace('single approved packet', 'fixed source and reference archives').replace('unchanged input packet', 'unchanged native reference archive')
    extract_node = next(n for n in ast.parse(e.raw).body if isinstance(n, ast.FunctionDef) and n.name == 'extract_packet')
    extract_ref = ast.get_source_segment(e.raw.decode(), extract_node).replace('extract_packet(', 'extract_reference(').replace('CAPSULE_FILES', 'REFERENCE_FILES').replace('len(files) != 36', 'len(files) != 19').replace('component-inputs.tar', 'native-reference.tar').replace("/ 'source' /", "/ 'reference' /").replace('capsule', 'reference')
    e.one("    guard.verify_assets(phase='setup')\n    guard.proof_readiness()\n    check()\n    return (coordinator, scratch, guard, binding, derived)",
          "    receive_reference(incoming, root, contract['reference']['archive'], check)\n    extract_reference(root, contract['reference']['files'], check)\n    guard.verify_assets(phase='setup')\n    guard.proof_readiness()\n    check()\n    return (coordinator, scratch, guard, binding, derived)")
    e.append('\nREFERENCE_PATH = ' + repr(pins['reference_archive']) + '\nREFERENCE_MANIFEST_SHA = ' + repr(reference['manifest_sha256']) + '\nREFERENCE_FILES = ' + repr(ref_members) + '\n\n' + receive_ref + '\n\n' + extract_ref + '\n')
    outer, derivation['outer'] = e.finish()

    e = Editor(f[pins['old_client']])
    for before, after in [('predictor_pgd_component_outer_v1', 'predictor_pgd_candidate_outer_v1'),
                          ('predictor_pgd_component_bootstrap_v1', 'predictor_pgd_candidate_bootstrap_v1'),
                          ('predictor_pgd_component_collection_v1', 'predictor_pgd_candidate_collection_v1')]:
        e.one(before, after)
    e.one("    return (c, b, p, csha, raw, araw, a, mapping, code, packet)",
          "    reference = root / c['reference_path']\n    wanted_reference = contract['reference']['archive']\n    file_bytes(reference, wanted_reference['bytes'], wanted_reference['sha256'])\n    return (c, b, p, csha, raw, araw, a, mapping, code, (packet, reference))")
    e.one("                with packet.open('rb') as source:\n                    while True:\n                        block = source.read(outer.MiB)\n                        if not block:\n                            break\n                        outer.write_all(proc.stdin, block, lambda: None)",
          "                for archive in packet:\n                    with archive.open('rb') as source:\n                        while True:\n                            block = source.read(outer.MiB)\n                            if not block:\n                                break\n                            outer.write_all(proc.stdin, block, lambda: None)")
    client, derivation['client'] = e.finish()
    generated = {OUTER: outer, CLIENT: client}
    for name, target in READERS.items():
        data = sources[name]
        if name == NAMES[3]:
            re = Editor(data)
            re.one('predictor_pgd_split_bootstrap_v1 as boot', 'predictor_pgd_candidate_bootstrap_v1 as boot')
            re.one('predictor_pgd_split_scratch_v1 as scratch', 'predictor_pgd_candidate_scratch_v1 as scratch')
            data, derivation['local_collection_import_bridge'] = re.finish()
        generated[target] = data
    uploads = dict({'overlay/' + n: v for n, v in workers.items()},
                   **{'supervisor/' + n: v for n, v in sources.items()})
    require(len(uploads) == 45 and len(catalogue['assets']) == 104, 'Exact candidate upload catalogue required.')
    require(sum(map(len, uploads.values())) <= 2 * 2**20, 'Original source upload byte ceiling retained.')
    facts = {'kind': 'candidate-only-operator-cold-derivation-v1', 'edits': derivation,
             'runtime_files': 39, 'runtime_bytes': sum(map(len, workers.values())),
             'uploaded_sources': 45, 'uploaded_source_bytes': sum(map(len, uploads.values())),
             'roles': list(ROLES), 'source_hashes': static['source_hashes'],
             'worker_source_bytes_unchanged': True, 'owned_monitor_and_proof_bytes_unchanged': True,
             'all_byte_inverses_verified': True, 'execution_binding': None,
             'new_scope_handoff_authorization_or_target_work': False, 'training_admitted': False}
    return {'control': control, 'policy': policy, 'catalogue': catalogue, 'static': static,
            'source_hashes': static['source_hashes'], 'sources': sources, 'workers': workers,
            'uploads': uploads, 'generated': generated, 'derivation': facts}


def write_local(repository, destination):
    """Save review artifacts only; deliberately has no handoff/arming API."""
    result = assemble(repository); root = Path(repository); dest = Path(destination)
    require(not dest.exists() and not any((root / n).exists() for n in result['generated']), 'Fresh separate integration paths required.')
    dest.mkdir(mode=0o700)
    for name, value in result['generated'].items():
        with (root / name).open('xb') as out:
            out.write(value)
    records = {'CONTROL-UNBOUND.json': result['control'], 'CATALOGUE.json': enc(result['catalogue']),
               'DERIVATION.json': enc(result['derivation']),
               'SOURCE-PAYLOAD.json': enc({n: base64.b64encode(v).decode() for n, v in result['uploads'].items()})}
    for name, value in records.items():
        with (dest / name).open('xb') as out:
            out.write(value)
    return result
