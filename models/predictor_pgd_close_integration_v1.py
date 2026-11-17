"""Local cold assembly of the Store/owned-exit repairs with full PGD bindings.

No target API, launch, authorization, packet/checkpoint read or scientific import.
The returned 69 source files and local readers are candidates, not a handoff.
"""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
from models import predictor_pgd_host10609_v1 as literal_binding
from models import predictor_pgd_store_close_v2 as store_recipe
from models import predictor_pgd_exit_race_v1 as watch_recipe

FOLDER = 'runs/pretrained/diagnostic/predictor-pgd-host10609-v1-20261116/handoff-parent/linux10609-final-unarmed'
CONTROL = FOLDER+'/CONTROL-CANDIDATE.json'
PAYLOAD = FOLDER+'/SOURCE-PAYLOAD.json'
FROZEN = {CONTROL: '2314abaa5a3ef3b47b3af54908b90eefa0c68cb9380cefbd408106a55aff59d4',
          PAYLOAD: '904a2869fef83fd855b73d363088cd5d9badd07da7dbd21e5ba1389e42cab2f5'}
STORE = 'overlay/models/predictor_pgd_split_store_v1.py'
WATCH = 'supervisor/predictor_pgd_split_watch_v1.py'
COORDINATOR = 'supervisor/predictor_pgd_split_coordinator_v1.py'
SCRATCH = 'supervisor/predictor_pgd_split_scratch_v1.py'
BOOT = 'supervisor/bootstrap.py'
COLLECTION = 'supervisor/predictor_pgd_split_collection_v1.py'
READERS = {'bootstrap': 'models/predictor_pgd_close_bootstrap_v1.py',
           'scratch': 'models/predictor_pgd_close_scratch_v1.py',
           'collection': 'models/predictor_pgd_close_collection_v1.py'}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def descriptors(sources, prefix):
    return {n.removeprefix(prefix): {'bytes': len(v), 'sha256': digest(v)} for n, v in sources.items() if n.startswith(prefix)}


def rewrite(data, mapping):
    result, edits = literal_binding.constants(data, mapping)
    if {ast.literal_eval(e['old_literal']) for e in edits} != set(mapping):
        raise ValueError('Every dependent source pin must be replaced, and no others.')
    return result, edits


def load_frozen(repository):
    root = Path(repository)
    frozen = {n: literal_binding.read(root/n, wanted=sha) for n, sha in FROZEN.items()}
    outer = literal_binding.parsed(frozen[CONTROL])
    control = literal_binding.parsed(outer['supervised_control_text'])
    policy = literal_binding.parsed(control['policy_text'])
    if digest(control['policy_text'].encode()) != control['policy_sha256'] or digest(outer['supervised_control_text'].encode()) != outer['supervised_control_sha256']:
        raise ValueError('Exact historical source/control joins required.')
    mapping = literal_binding.parsed(frozen[PAYLOAD])
    expected = {'supervisor/'+n: v for n, v in control['supervisor_sources'].items()}
    expected.update({'overlay/'+n: v for n, v in outer['runtime_files'].items()})
    if set(mapping) != set(expected) or len(mapping) != 69:
        raise ValueError('All 69 original source descriptors required.')
    source = {}
    for name, text in mapping.items():
        data = base64.b64decode(text, validate=True)
        if len(data) != expected[name]['bytes'] or digest(data) != expected[name]['sha256']:
            raise ValueError('Consumed source bytes changed: '+name)
        source[name] = data
    return outer, control, policy, source


def assemble(repository):
    """Return frozen scientific scope plus new exact source bindings, never work."""
    outer, old_control, old_policy, original = load_frozen(repository)
    sources = dict(original)
    sources[STORE], store_meta = store_recipe.build(original[STORE])
    sources[WATCH], watch_meta = watch_recipe.build(original[WATCH])
    old_watch, new_watch = digest(original[WATCH]), digest(sources[WATCH])
    traces = {}
    sources[COORDINATOR], traces[COORDINATOR] = rewrite(original[COORDINATOR], {old_watch: new_watch})
    old_contract = literal_binding.parsed(old_policy['scratch_contract_text'])
    contract = copy.deepcopy(old_contract)
    runtime = descriptors(sources, 'overlay/')
    manifest = {'kind': 'pgd-close-runtime-manifest-v1', 'prior_runtime_manifest_sha256': old_contract['runtime_manifest_sha256'],
                'runtime_files': runtime}
    contract['runtime_files'] = runtime
    contract['runtime_manifest_sha256'] = digest(encoded(manifest))
    if contract['sources']['models/predictor_pgd_split_watch_v1.py'] != old_watch:
        raise ValueError('Original contract monitor pin required.')
    contract['sources']['models/predictor_pgd_split_watch_v1.py'] = new_watch
    contract_bytes = encoded(contract)
    sources[SCRATCH], traces[SCRATCH] = rewrite(original[SCRATCH], {old_policy['scratch_contract_sha256']: digest(contract_bytes), old_watch: new_watch})
    policy = copy.deepcopy(old_policy)
    policy['scratch_contract_text'] = contract_bytes.decode()
    policy['scratch_contract_sha256'] = digest(contract_bytes)
    policy['fixed_support']['predictor_pgd_split_watch_v1.py'] = new_watch
    policy['runtime_repair'] = {'kind': 'pgd-close-integration-v1', 'prior_policy_sha256': old_control['policy_sha256'],
        'runtime_manifest_text': encoded(manifest).decode(), 'full_store_candidate_sha256': digest(sources[STORE]),
        'monitor_candidate_sha256': new_watch, 'scientific_and_resource_scope_unchanged': True}
    policy_bytes = encoded(policy)
    sources[BOOT], traces[BOOT] = rewrite(original[BOOT], {old_control['policy_sha256']: digest(policy_bytes)})
    for name, data in sources.items():
        compile(data, '<UNEXECUTED-CLOSE-INTEGRATION-'+name+'>', 'exec')
    changed = sorted(n for n in sources if sources[n] != original[n])
    if set(changed) != {STORE, WATCH, COORDINATOR, SCRATCH, BOOT}:
        raise ValueError('Unexpected source repair surface.')
    if (policy['scope'] != old_policy['scope'] or policy['base_text'] != old_policy['base_text']
            or policy['design_text'] != old_policy['design_text'] or policy['host_change'] != old_policy['host_change']):
        raise ValueError('Original scientific/resource/host/reference/financial scope changed.')
    new_control = encoded({'kind': old_control['kind'], 'policy_text': policy_bytes.decode(),
                          'policy_sha256': digest(policy_bytes), 'supervisor_sources': descriptors(sources, 'supervisor/')})
    if len(new_control) > 256*1024 or sum(map(len, sources.values())) > 2*2**20:
        raise ValueError('Original metadata/source caps exceeded.')
    # These are local reader modules, not extra target payload files. Target
    # imports keep their original pinned generic namespace through the finder.
    readers = {READERS['bootstrap']: sources[BOOT], READERS['scratch']: sources[SCRATCH]}
    collector = original[COLLECTION]
    for old, new in ((b'predictor_pgd_split_bootstrap_v1 as boot', b'predictor_pgd_close_bootstrap_v1 as boot'),
                     (b'predictor_pgd_split_scratch_v1 as scratch', b'predictor_pgd_close_scratch_v1 as scratch')):
        if collector.count(old) != 1:
            raise ValueError('Exact local collector import bridge required.')
        collector = collector.replace(old, new, 1)
    readers[READERS['collection']] = collector
    for name, data in readers.items():
        compile(data, '<UNEXECUTED-CLOSE-READER-'+name+'>', 'exec')
    metadata = {'kind': 'pgd-close-integration-v1-cold-candidate', 'prior_outer_control_sha256': FROZEN[CONTROL],
        'supervised_control_sha256': digest(new_control), 'policy_sha256': digest(policy_bytes),
        'scratch_contract_sha256': digest(contract_bytes), 'runtime_manifest_sha256': digest(encoded(manifest)),
        'store_repair': store_meta, 'monitor_repair': watch_meta, 'source_literal_edits': traces,
        'changed_uploaded_sources': changed, 'unchanged_uploaded_sources': len(sources)-len(changed),
        'uploaded_source_files': len(sources), 'uploaded_source_bytes': sum(map(len, sources.values())),
        'runtime_files': len(runtime), 'reader_pins': {n: digest(v) for n, v in readers.items()},
        'historical_host_manifest_is_provenance_current_manifest_is_runtime_repair': True,
        'full_scientific_catalogue_retained': True, 'scientific_resource_host_and_financial_scope_unchanged': True,
        'runnable_outer_client_or_target_handoff_prepared': False, 'authorized': False,
        'scientific_qualified': False, 'training_admitted': False}
    return new_control, sources, readers, manifest, metadata


def verify_readers(repository, readers):
    """Read/hash exact local sources only; never import target or decode tensors."""
    if set(readers) != set(READERS.values()):
        raise ValueError('Exact three local reader modules required.')
    for name, data in readers.items():
        actual = literal_binding.read(Path(repository)/name, wanted=digest(data))
        if actual != data:
            raise ValueError('Local reader and server binding differ: '+name)
    return {'kind': 'pgd-close-local-reader-source-binding-v1', 'reader_files': len(readers),
            'source_identity_verified': True, 'scientific_qualified': False}
