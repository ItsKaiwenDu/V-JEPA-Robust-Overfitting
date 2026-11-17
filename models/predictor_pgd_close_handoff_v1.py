"""Cold packaging of the reviewed PGD repairs; never authorizes or launches work."""
import ast
import base64
import os
from pathlib import Path
import stat
from models import predictor_pgd_close_integration_v1 as integration
from models import predictor_pgd_host10609_v1 as literals

OLD = {'outer': 'models/predictor_pgd_split_outer_10609_v1.py',
       'client': 'models/predictor_pgd_split_outer_client_10609_v1.py',
       'cli': 'scripts/run_predictor_pgd_split_outer_10609_v1.py'}
FROZEN = {OLD['outer']: '0b82e7f724793aa13a7308b342746366cc0345fa4b3ac741e9ce5c1ecf6a0eb9',
          OLD['client']: 'f6d473688d52d0e2c330d931e5afb3d006595f3f13324b16fc4caa9ae69cb85d',
          OLD['cli']: 'adcc5e74d44985b01b13aa3c8d69b80128e6380dbe0391279960161b467846a3'}
NEW = {'outer': 'models/predictor_pgd_close_outer_v1.py',
       'client': 'models/predictor_pgd_close_client_v1.py',
       'cli': 'scripts/run_predictor_pgd_close_v1.py',
       'builder': 'models/predictor_pgd_close_handoff_v1.py'}
POLICY_SHA = 'b502797a50c09627785cc404672eab1f6815004db3f5547a713b649b205fa7f0'
BOOT_SHA = '6b5ddad6207d667abe01d6219033b58748d8d041d95d9662c3090370f90fb6c4'
COLLECTOR_SHA = '50e1deede0e0d9bf9a0f5a7ad9f1879920f4d01b74015666adcc89106b2523a1'
CONTROL_KIND = 'pgd-close-outer-control-v1'
AUTH_KIND = 'pgd-close-outer-university-exception-v1'
encoded, digest = integration.encoded, integration.digest


def assignment(raw, name):
    return next(n for n in ast.parse(raw).body if isinstance(n, ast.Assign)
                and any(isinstance(v, ast.Name) and v.id == name for v in n.targets))


def replace_value(raw, name, value):
    node = assignment(raw, name).value
    lines = raw.splitlines(keepends=True)
    begin = sum(map(len, lines[:node.lineno-1]))+node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset
    new = repr(value).encode()
    result = raw[:begin]+new+raw[end:]
    if result[:begin]+raw[begin:end]+result[begin+len(new):] != raw:
        raise ValueError('Exact assignment inverse failed.')
    return result, {'begin': begin, 'new_end': begin+len(new), 'old_value': raw[begin:end].decode(), 'new_value': new.decode()}


def replace_once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('One exact import bridge required.')
    result = raw.replace(old, new, 1)
    if result.replace(new, old, 1) != raw:
        raise ValueError('Exact import inverse failed.')
    return result


def build_local(repository):
    root = Path(repository)
    old = {n: literals.read(root/path, wanted=FROZEN[path]) for n, path in OLD.items()}
    control, source, readers, manifest, meta = integration.assemble(root)
    integration.verify_readers(root, readers)
    if meta['policy_sha256'] != POLICY_SHA or digest(source[integration.BOOT]) != BOOT_SHA or digest(source[integration.COLLECTION]) != COLLECTOR_SHA:
        raise ValueError('Exact reviewed Store/monitor candidate required.')
    replacements = {'bf4d7f961d70c6c998ef91f27b4242417f3e9caae3ee2d801bbbb79a6c1756c5': POLICY_SHA,
                    'd0b7933dc001bb603c38bf9b7aaa9488080cb3d0f2661145886a570f19a870fb': BOOT_SHA,
                    'pgd-split-outer-control-host10609-v1': CONTROL_KIND,
                    'pgd-split-outer-university-exception-host10609-v1': AUTH_KIND,
                    OLD['outer']: NEW['outer']}
    outer, trace = literals.constants(old['outer'], replacements)
    old_locals = ast.literal_eval(assignment(old['outer'], 'LOCAL_SOURCES').value)
    paths = {**{OLD[n]: NEW[n] for n in ('outer', 'client', 'cli')},
             'models/predictor_pgd_split_bootstrap_10609_v1.py': integration.READERS['bootstrap'],
             'models/predictor_pgd_split_collection_10609_v1.py': integration.READERS['collection'],
             'models/predictor_pgd_split_scratch_10609_v1.py': integration.READERS['scratch']}
    local_sources = tuple(paths.get(n, n) for n in old_locals)+(
        NEW['builder'], 'models/predictor_pgd_close_integration_v1.py',
        'models/predictor_pgd_store_close_v2.py', 'models/predictor_pgd_exit_race_v1.py',
        integration.CONTROL, integration.PAYLOAD, OLD['outer'], OLD['client'], OLD['cli'])
    if len(set(local_sources)) != len(local_sources):
        raise ValueError('Unique complete local source coverage required.')
    outer, value_trace = replace_value(outer, 'LOCAL_SOURCES', local_sources)
    outer = replace_once(outer, b'from models import predictor_pgd_host10609_v1 as binding',
                         b'from models import predictor_pgd_close_handoff_v1 as binding')
    bridges = ((b'predictor_pgd_split_outer_10609_v1 as outer', b'predictor_pgd_close_outer_v1 as outer'),
               (b'predictor_pgd_split_bootstrap_10609_v1 as boot', b'predictor_pgd_close_bootstrap_v1 as boot'),
               (b'predictor_pgd_split_collection_10609_v1 as collection', b'predictor_pgd_close_collection_v1 as collection'))
    client = old['client']
    for before, after in bridges:
        client = replace_once(client, before, after)
    cli = replace_once(old['cli'], b'from models.predictor_pgd_split_outer_client_10609_v1 import run',
                        b'from models.predictor_pgd_close_client_v1 import run')
    result = {NEW['outer']: outer, NEW['client']: client, NEW['cli']: cli}
    for name, data in result.items():
        compile(data, '<UNEXECUTED-CLOSE-HANDOFF-'+name+'>', 'exec')
    info = {'kind': 'pgd-close-handoff-local-source-derivation-v1',
            'frozen_local_source_pins': FROZEN, 'outer_literal_edits': trace, 'local_source_assignment': value_trace,
            'local_source_files': len(local_sources), 'policy_sha256': POLICY_SHA, 'bootstrap_sha256': BOOT_SHA,
            'collector_sha256': COLLECTOR_SHA, 'control_kind': CONTROL_KIND, 'authorization_kind': AUTH_KIND,
            'uploaded_source_files': len(source), 'uploaded_source_bytes': sum(map(len, source.values())),
            'integration_supervised_control_sha256': digest(control),
            'new_authorization_or_attempt_created': False, 'target_work_or_SSH_started': False,
            'scientific_qualified': False, 'training_admitted': False}
    return result, info


def prepare_package(repository, destination):
    """Create one fresh source-only LOCAL package; no packet read or grant."""
    from models import predictor_pgd_close_outer_v1 as outer
    from models import predictor_pgd_split_source_bundle_v1 as reader
    root, dest = Path(repository), Path(destination)
    parent = dest.parent.lstat()
    if (not root.is_absolute() or root.resolve(strict=True) != root or not dest.is_absolute()
            or dest.parent.resolve(strict=True) != dest.parent or os.path.lexists(dest)
            or not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700 or parent.st_uid != os.getuid()):
        raise ValueError('Fresh canonical owned private handoff required.')
    generated, derivation = build_local(root)
    for name, wanted in generated.items():
        if literals.read(root/name, wanted=digest(wanted)) != wanted:
            raise ValueError('Generated local driver changed: '+name)
    supervised, sources, readers, manifest, integration_meta = integration.assemble(root)
    integration.verify_readers(root, readers)
    outer.verify_directory_policy_sources(root, reader)
    outer.verify_refusal_policy_sources(root, reader)
    b = literals.parsed(supervised)
    policy = literals.parsed(b['policy_text'])
    pins = {name: digest(literals.read(root/name)) for name in outer.LOCAL_SOURCES}
    control = {'kind': CONTROL_KIND, 'supervised_control_text': supervised.decode(),
               'supervised_control_sha256': digest(supervised), 'outer_source_sha256': pins[NEW['outer']],
               'local_source_sha256': pins, 'runtime_files': manifest['runtime_files'], 'repository': str(root),
               'local_handoff': str(dest), 'packet_path': outer.PACKET_PATH, 'known_hosts_text': outer.KNOWN_HOSTS,
               'known_hosts_sha256': digest(outer.KNOWN_HOSTS.encode()), 'expected_ssh_fingerprint': outer.FINGERPRINT,
               'financial_state_unchanged': True, 'authorized_at_preparation': False, 'target_helper_reviewed': False}
    raw = encoded(control)
    outer.validate_control(raw, pins[NEW['outer']])
    payload = encoded({n: base64.b64encode(v).decode() for n, v in sources.items()})
    outer.decode_sources(literals.parsed(payload), control, b)
    if len(payload)+len(raw)+64*1024 > 2*2**20:
        raise ValueError('Original source/control upload frame cap exceeded.')
    result = {'kind': 'pgd-close-handoff-unarmed-package-v1', 'outer_control_sha256': digest(raw),
              'supervised_control_sha256': digest(supervised), 'source_files': len(sources),
              'source_bytes': sum(map(len, sources.values())), 'source_envelope_bytes': len(payload),
              'local_source_files': len(pins), 'derivation': derivation, 'limits': policy['scope'],
              'authorization_created': False, 'target_work_or_SSH_executed': False,
              'scientific_qualified': False, 'training_admitted': False}
    dest.mkdir(mode=0o700)
    for name, data in (('CONTROL-CANDIDATE.json', raw), ('SOURCE-PAYLOAD.json', payload),
                       ('known-hosts', outer.KNOWN_HOSTS.encode()), ('PACKAGE.json', encoded(result))):
        outer.private_write(dest, name, data, lambda: None)
    return result
