"""Full pretrained EOT8 mask serialization and unchanged storage boundaries.

These tests generate CPU masks only: no model/weights, GPU, SSH or university run.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
from models import predictor_sdpa_storage_v2 as policy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT/'configs/predictor_sdpa_handoff_v1.json'
CONTRACT_SHA = 'a0c19f26b678e095eeafbe6424b479179e76544b3d4e9e40345af2b0100b1fbd'


def put(root, name, data):
    path = root/name
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o600)
    return path


@pytest.fixture(scope='module')
def masks():
    import torch
    from models.vjepa2 import sample_masks
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == CONTRACT_SHA
    contract = json.loads(CONTRACT.read_text())
    expected = contract['approved_baseline']
    assert not torch.cuda.is_initialized()
    indices = []
    for seed, draw in zip(expected['seeds'], expected['masks'], strict=True):
        pairs = sample_masks(ROOT/'external/vjepa2', batch_size=1, seed=seed % (2**31-1))
        assert len(pairs) == len(draw) == 2
        values = []
        for pair, signatures in zip(pairs, draw, strict=True):
            for tensor, want in zip((pair.context, pair.target), signatures, strict=True):
                actual = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                          'sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                          'abs_max': float(tensor.abs().max())}
                assert actual == want
            values.append({'context': pair.context.tolist(), 'target': pair.target.tolist()})
        indices.append(values)
    assert len(indices) == 8 and not torch.cuda.is_initialized()
    return indices


def test_full_original_EOT8_masks_reproduce_observed_failure_and_new_policy_accepts(tmp_path, masks):
    data = policy.frozen.old.json_bytes(masks)
    assert len(data) == 146562
    assert sum(len(p['context'][0])+len(p['target'][0]) for d in masks for p in d) == 32768
    path = put(tmp_path, policy.MASK_PATH, data)
    with pytest.raises(ValueError, match='typed-file'):
        policy.frozen.inventory(tmp_path)
    files, _ = policy.inventory(tmp_path, hashing=True)
    assert files[policy.MASK_PATH]['bytes'] == 146562
    assert files[policy.MASK_PATH]['sha256'] == hashlib.sha256(data).hexdigest()
    assert path.read_bytes() == data


def test_worst_case_full_shape_serialization_fits_and_atomic_partial_is_counted(tmp_path, masks):
    worst = [[{k: [[2047]*len(v[0])] for k, v in p.items()} for p in d] for d in masks]
    data = policy.frozen.old.json_bytes(worst)
    assert len(data) == 164322 < policy.MASK_FILE_BYTES
    partial = put(tmp_path, policy.MASK_PATH+'.partial', data)
    assert policy.inventory(tmp_path, running=True)[0][policy.MASK_PATH+'.partial']['bytes'] == len(data)
    os.replace(partial, tmp_path/policy.MASK_PATH)
    assert policy.inventory(tmp_path)[0][policy.MASK_PATH]['bytes'] == len(data)


@pytest.mark.parametrize('suffix', ('', '.partial'))
def test_exact_mask_limit_inclusive_and_one_byte_over_refuses_with_filename(tmp_path, suffix):
    name = policy.MASK_PATH+suffix
    path = put(tmp_path, name, b' ' * policy.MASK_FILE_BYTES)
    policy.inventory(tmp_path, running=True)
    with path.open('ab') as f:
        f.write(b'!')
    with pytest.raises(ValueError, match=r'mask-indices.json.*262145 > 262144'):
        policy.inventory(tmp_path, running=True)
    out = io.BytesIO()
    with pytest.raises(ValueError, match='typed-file'):
        policy.stream(tmp_path, 'fixture-only', out)
    assert not out.getvalue()


@pytest.mark.parametrize('name', (
    'mask-indices.json', 'results/mask-indices.json', 'results/capture/sub/mask-indices.json',
    'results/components/0/mask-indices.json', 'results/capture/mask-indices.json.partial.partial',
    'results/capture/mask-indices.json.bak', 'results/capture/boundary/attention-scopes.json',
    'results/capture/boundary/attention-scopes.json.partial',
))
def test_allowance_cannot_be_borrowed_by_nearby_or_unrelated_files(tmp_path, name):
    put(tmp_path, name, b' ' * (policy.typed_file_limit(name)+1))
    with pytest.raises(ValueError, match='typed-file'):
        policy.inventory(tmp_path, running=True)
    if 'mask-indices.json' in name and not name.endswith(('.bak', '.partial.partial')):
        assert policy.typed_file_limit(name) == 65536


@pytest.mark.parametrize('name', ('/results/capture/mask-indices.json', './results/capture/mask-indices.json',
                                 'results/capture/../mask-indices.json', 'results\\capture\\mask-indices.json'))
def test_aliases_do_not_receive_an_exact_path_exception(name):
    with pytest.raises(ValueError, match='Unsafe'):
        policy.typed_file_limit(name)


def test_atomic_link_pair_keeps_both_names_and_original_accounting(tmp_path, masks):
    path = put(tmp_path, policy.MASK_PATH+'.partial', policy.frozen.old.json_bytes(masks))
    final = tmp_path/policy.MASK_PATH
    os.link(path, final)
    files, _ = policy.inventory(tmp_path, running=True)
    assert set(files) == {policy.MASK_PATH, policy.MASK_PATH+'.partial'}
    assert all(v['bytes'] == 146562 for v in files.values())
    path.unlink()
    assert set(policy.inventory(tmp_path)[0]) == {policy.MASK_PATH}


@pytest.mark.parametrize('kind', ('symlink', 'external-hardlink', 'unsafe-directory'))
def test_existing_filesystem_guards_still_refuse(tmp_path, kind):
    folder = tmp_path/'scratch'
    folder.mkdir(mode=0o700)
    if kind == 'unsafe-directory':
        folder.chmod(0o755)
        info = tmp_path.stat()
        with pytest.raises(ValueError, match='Private'):
            policy.storage(folder, {'device': info.st_dev, 'inode': info.st_ino}, parent=tmp_path)
    else:
        other = put(tmp_path, 'outside', b'outside')
        dest = folder/'foreign'
        dest.symlink_to(other) if kind == 'symlink' else os.link(other, dest)
        with pytest.raises(ValueError):
            policy.inventory(folder)


def test_masks_stay_metadata_with_unchanged_aggregate_accounting(tmp_path, masks):
    folder = tmp_path/'scratch'
    folder.mkdir(mode=0o700)
    put(folder, policy.MASK_PATH, policy.frozen.old.json_bytes(masks))
    s = tmp_path.stat()
    row = policy.storage(folder, {'device': s.st_dev, 'inode': s.st_ino}, parent=tmp_path)
    assert row['categories']['metadata'] == 146562 and row['categories']['raw'] == 0
    assert row['ceiling'] == policy.frozen.RESOURCE['capture_bytes'] == 9*policy.GiB//4
    assert policy.RESOURCE == policy.frozen.RESOURCE


@pytest.mark.parametrize('category', ('metadata', 'raw', 'cache', 'input', 'code', 'emergency'))
def test_category_ceilings_still_enforced(tmp_path, monkeypatch, category):
    folder = tmp_path/'scratch'
    folder.mkdir(mode=0o700)
    s = tmp_path.stat()
    names = {'metadata': 'other.json', 'raw': 'results/input.pt', 'cache': 'cache/entry',
             'input': 'original-packet.tar.gz', 'code': 'code/file.py', 'emergency': 'emergency-space.bin'}
    # Aggregate-only fixture, deliberately independent from individual-file caps.
    monkeypatch.setattr(policy, 'inventory', lambda *a, **k: ({names[category]: {'bytes': policy.RESOURCE[category+'_bytes']+1}}, 0))
    with pytest.raises(ValueError, match='category'):
        policy.storage(folder, {'device': s.st_dev, 'inode': s.st_ino}, parent=tmp_path)


@pytest.mark.parametrize('setup', (False, True))
def test_allocated_byte_ceiling_still_enforced(tmp_path, monkeypatch, setup):
    folder = tmp_path/'scratch'
    folder.mkdir(mode=0o700)
    s = tmp_path.stat()
    ceiling = policy.RESOURCE['setup_bytes' if setup else 'capture_bytes']
    monkeypatch.setattr(policy, 'inventory', lambda *a, **k: ({}, ceiling+1))
    with pytest.raises(ValueError, match='Allocated'):
        policy.storage(folder, {'device': s.st_dev, 'inode': s.st_ino}, setup=setup, parent=tmp_path)


@pytest.mark.parametrize('name,limit', (
    ('results/capture/worker.json.partial', 256*1024), ('results/capture/boundary/boundary.json', 128*1024),
    ('results/capture/boundary/attention-scopes.json', 64*1024), ('results/capture/boundary/attention-calls.jsonl.partial', 8*2**20),
    ('results/capture/boundary/q.pt.partial', 16*2**20), ('results/capture/capture-failure.json', 16*1024),
    ('original-packet.tar.gz.partial', 2**20), ('cache/cuda/entry', 2**20), ('emergency-space.bin', 16*2**20),
))
def test_non_mask_typed_limits_are_not_relaxed(name, limit):
    assert policy.typed_file_limit(name) == limit


def test_full_mask_collection_preserves_bytes_hashes_modes_and_ns_mtime(tmp_path, masks):
    source = tmp_path/'source'
    source.mkdir(mode=0o700)
    path = put(source, policy.MASK_PATH, policy.frozen.old.json_bytes(masks))
    put(source, 'session.json', policy.frozen.old.json_bytes({'complete': False, 'fixture_only': True}))
    os.utime(path, ns=(1700000000123456789, 1700000000123456789))
    output = io.BytesIO()
    policy.stream(source, 'local-fixture-control', output)
    output.seek(0)
    header = json.loads(output.readline())
    assert header['kind'] == policy.COLLECTION_KIND != policy.frozen.KIND
    assert header['model_restarted'] is False
    assert header['files'][policy.MASK_PATH]['mtime_ns'] == path.stat().st_mtime_ns
    with tarfile.open(fileobj=output, mode='r|') as archive:
        seen = {}
        for member in archive:
            data = archive.extractfile(member).read()
            spec = header['files'][member.name]
            assert len(data) == spec['bytes'] and hashlib.sha256(data).hexdigest() == spec['sha256']
            assert member.mode == spec['mode'] == 0o600
            seen[member.name] = data
    assert seen[policy.MASK_PATH] == path.read_bytes()
    assert header['logical_bytes'] == sum(map(len, seen.values()))


def test_collection_refuses_source_change_before_sending_changed_file(tmp_path, monkeypatch):
    path = put(tmp_path, policy.MASK_PATH, b'original')
    original = policy.inventory
    def inventory_then_change(*a, **k):
        value = original(*a, **k)
        path.write_bytes(b'changed!')
        return value
    monkeypatch.setattr(policy, 'inventory', inventory_then_change)
    with pytest.raises(ValueError, match='Collection source changed'):
        policy.stream(tmp_path, 'fixture-only', io.BytesIO())


def test_import_is_standard_library_only_and_creates_no_process_or_network(tmp_path):
    code = '''import sys
sys.path.insert(0, sys.argv[1])
def block(event,args):
 if event.startswith(('subprocess.', 'socket.', 'os.fork', 'os.exec', 'os.spawn')):raise RuntimeError(event)
 if event=='import' and args[0].split('.')[0] in ('torch','numpy','timm'):raise RuntimeError(args[0])
sys.addaudithook(block)
from models import predictor_sdpa_storage_v2
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
    p = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(ROOT)], cwd=tmp_path,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    assert p.returncode == 0, p.stderr.decode()


def test_observed_144_file_catalog_changes_only_the_exact_mask_ceiling(tmp_path, monkeypatch):
    response = ROOT/'runs/pretrained/diagnostic/sdpa-collection-incident-20261107/metadata-probe-01/response.json'
    assert hashlib.sha256(response.read_bytes()).hexdigest() == 'd4a82d9c91579a8d27ba2836e0ca5995430eec699e0c2a630ace5544503dfad4'
    rows = json.loads(response.read_text())['files']
    assert len(rows) == 144
    changed = []
    for item in rows:
        name, previous = item['file'], item['old_typed_limit_bytes']
        data = {name: {'bytes': previous}}
        monkeypatch.setattr(policy.frozen.base, 'inventory', lambda *a, **k: (data, 0))
        policy.frozen.inventory(tmp_path)
        policy.inventory(tmp_path)
        data[name]['bytes'] += 1
        with pytest.raises(ValueError, match='typed-file'):
            policy.frozen.inventory(tmp_path)
        if name == policy.MASK_PATH:
            policy.inventory(tmp_path)
            changed.append(name)
        else:
            with pytest.raises(ValueError, match='typed-file'):
                policy.inventory(tmp_path)
    assert changed == [policy.MASK_PATH]


def test_legacy_storage_and_collection_match_except_explicit_new_envelope(tmp_path):
    source = tmp_path/'source'
    source.mkdir(mode=0o700)
    for name, data in ((policy.MASK_PATH, b'[]\n'), ('results/capture/input.pt', b'saved fixture bytes'),
                       ('session.json', b'{"complete":false}\n'), ('cache/note', b'cache fixture')):
        put(source, name, data)
    s = tmp_path.stat()
    identity = {'device': s.st_dev, 'inode': s.st_ino}
    assert policy.storage(source, identity, parent=tmp_path) == policy.frozen.storage(source, identity, parent=tmp_path)
    current, previous = io.BytesIO(), io.BytesIO()
    policy.stream(source, 'fixture-control', current)
    policy.frozen.stream(source, 'fixture-control', previous)
    current.seek(0)
    previous.seek(0)
    new, old = json.loads(current.readline()), json.loads(previous.readline())
    assert new.pop('kind') == policy.COLLECTION_KIND
    assert old.pop('kind') == policy.frozen.KIND
    assert new == old and current.read() == previous.read()
