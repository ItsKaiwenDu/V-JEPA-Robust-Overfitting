"""New capsule/lineage boundary checks against the actual saved CUDA references.

No scientific worker, tensor import, reference calculation, network or storage proof.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest
from models import predictor_replay_capsule_v1 as cap

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT/'runs/pretrained/diagnostic/predictor-replay-capsule-v1-20261114/build-first'
MEASURES = json.loads((BUILD/'measurements.json').read_bytes()) if (BUILD/'measurements.json').exists() else None


@pytest.fixture(scope='module')
def copy_root(tmp_path_factory):
    assert MEASURES is not None, 'Build the capsule once before running its new boundary checks.'
    root = tmp_path_factory.mktemp('reference-boundary')/'capsule'
    shutil.copytree(BUILD/'capsule', root, copy_function=shutil.copy2)
    os.chmod(root, 0o700)
    return root


def open_reader(root):
    return cap.Capsule(root, manifest_sha256=MEASURES['manifest_sha256'])


@contextmanager
def changed(path, data=None, mode=None):
    before = path.stat(); old = path.read_bytes()
    try:
        if data is not None:
            path.write_bytes(data)
        if mode is not None:
            path.chmod(mode)
        yield
    finally:
        path.chmod(0o600); path.write_bytes(old); path.chmod(before.st_mode & 0o777)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_actual_CUDA_sources_cold_validate_without_relabeling(copy_root):
    before = dict(cap.records.load.__globals__)
    with open_reader(copy_root) as r:
        assert len(r.manifest['aliases']) == 2
        assert len([n for n in r.manifest['files'] if n.startswith('arrays/')]) == 37
        for role, index in cap.ROLES.items():
            state = r.states[role]
            assert state['index'] == index and state['control_sha256'] == cap.SOURCE_CONTROL
            assert state['execution_host'] == 'linux10603' and state['local_tiny'] is False
            assert state['complete'] is True and state['passed'] is True
            assert len(state['artifacts']) == 36
            for name in cap.PROOFS:
                actual = r._mapped_read(r._folders[role], name, 8*cap.MiB)
                original = ROOT/cap.SAVED/f'results/{index}'/name
                assert actual == original.read_bytes()
        assert r.states['fresh_process_candidate_reference']['ownership']['custom_VJPs'] == 192
        assert r.states['same_replay_original']['replay_checks'] == {}
    assert cap.records.load.__globals__ == before


def test_distinct_new_control_binds_but_does_not_authorize(copy_root):
    with open_reader(copy_root) as r:
        binding = r.execution_binding('ab'*32)
        assert binding['new_execution_control_sha256'] != binding['historical_reference_control_sha256']
        assert binding['new_worker'] == {'index': 3, 'mode': 'candidate', 'replay': 1}
        assert binding['execution_permitted'] is False and binding['authorization_required'] is True
        assert binding['old_grant_or_partial_replay_reused'] is False
        assert {v['source_index'] for v in binding['references'].values()} == {1, 2}
        for invalid in [cap.SOURCE_CONTROL, 'x'*64, 'a'*63, 'A'*64, None]:
            with pytest.raises(ValueError):
                r.execution_binding(invalid)


@pytest.mark.parametrize('role', ['partial_candidate', 'original0', 'candidate1', '', None])
def test_unknown_or_partial_role_refuses(copy_root, role):
    with open_reader(copy_root) as r:
        with pytest.raises(ValueError):
            r.load_reference(role)


def test_manifest_requires_separate_hash(copy_root):
    for sha in ['ab'*32, 'a'*63, None]:
        with pytest.raises(ValueError):
            cap.Capsule(copy_root, manifest_sha256=sha)


@pytest.mark.parametrize('relative', ['lineage/source-control.json', 'lineage/source-claim.json',
                                     'references/fresh_process_candidate_reference/success.json',
                                     'references/same_replay_original/native-calls.jsonl'])
def test_corrupt_lineage_or_marker_refuses_even_with_restored_mtime(copy_root, relative):
    path = copy_root/relative
    with changed(path, data=b'!'+path.read_bytes()[1:]):
        info = next(v for n, v in json.loads((copy_root/'manifest.json').read_bytes())['files'].items() if n == relative)
        os.utime(path, ns=(info['mtime_ns'], info['mtime_ns']))
        with pytest.raises(ValueError):
            open_reader(copy_root)


def test_complete_marker_required(copy_root):
    marker = copy_root/'references/same_replay_original/success.json'
    parked = copy_root.parent/'parked-success'
    marker.rename(parked)
    try:
        with pytest.raises(ValueError):
            open_reader(copy_root)
    finally:
        parked.rename(marker)


@pytest.mark.parametrize('name', ['failure.json', 'partial.npy', 'AUTHORIZATION.json'])
def test_extra_or_partial_files_refuse(copy_root, name):
    p = copy_root/'references/same_replay_original'/name
    try:
        p.write_bytes(b'{}'); p.chmod(0o600)
        with pytest.raises(ValueError):
            open_reader(copy_root)
    finally:
        p.unlink(missing_ok=True)


def test_hardlink_refuses_without_touching_original_sources(copy_root):
    blob = next((copy_root/'arrays').iterdir())
    link = copy_root.parent/'external-test-link'
    try:
        os.link(blob, link)
        with pytest.raises(ValueError):
            open_reader(copy_root)
    finally:
        link.unlink(missing_ok=True)
    assert blob.stat().st_nlink == 1


def test_symlink_parent_refuses(copy_root):
    folder = copy_root/'arrays'; parked = copy_root.parent/'parked-arrays'
    folder.rename(parked)
    try:
        folder.symlink_to(parked, target_is_directory=True)
        with pytest.raises((ValueError, OSError)):
            open_reader(copy_root)
    finally:
        folder.unlink(missing_ok=True); parked.rename(folder)


def test_public_file_mode_and_nsmtime_drift_refuse(copy_root):
    p = copy_root/'lineage/plan.json'
    with changed(p, mode=0o644):
        with pytest.raises(ValueError):
            open_reader(copy_root)
    before = p.stat()
    try:
        os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns+1))
        with pytest.raises(ValueError):
            open_reader(copy_root)
    finally:
        os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_relocated_root_refuses_open_reader(copy_root):
    parked = copy_root.parent/'parked-root'
    with open_reader(copy_root) as r:
        copy_root.rename(parked)
        try:
            copy_root.mkdir(mode=0o700)
            with pytest.raises(ValueError):
                r.load_reference('same_replay_original')
        finally:
            copy_root.rmdir(); parked.rename(copy_root)


def test_alias_retarget_refuses_even_if_outer_manifest_hash_rebound(copy_root):
    path = copy_root/'manifest.json'; m = json.loads(path.read_bytes())
    m['aliases']['same_replay_original']['loss.npy']['member'] = m['aliases']['same_replay_original']['input.npy']['member']
    altered = cap.encode(m)
    with changed(path, data=altered):
        with pytest.raises(ValueError):
            cap.Capsule(copy_root, manifest_sha256=cap.digest(altered))


def test_original_control_source_and_new_folder_not_interchangeable(copy_root):
    with open_reader(copy_root) as r:
        with pytest.raises(ValueError):
            r._folder(Path(cap.SAVED))
        with pytest.raises(ValueError):
            r._folder(cap._Folder(r, 'same_replay_original'))
        with pytest.raises(ValueError):
            r._mapped_read(r._folders['same_replay_original'], '../worker.json', cap.MiB)


def npy(descr='<f4', shape=(), data=b'\x00'*4, **extra):
    header = repr(dict(descr=descr, fortran_order=False, shape=shape, **extra)).encode()+b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H', len(header))+header+data


@pytest.mark.parametrize('payload', [npy('|O'), npy('>f4'), npy(shape=(2,)), npy(shape=(True,)),
                                    npy(extra=1), npy()+b'extra', b'not NPY',
                                    b'\x93NUMPY\x01\x00\xff\xff'])
def test_NPY_object_shape_endian_extent_header_refusals(payload):
    with pytest.raises((ValueError, SyntaxError)):
        cap.npy_type(payload)


def test_duplicate_json_and_NPY_keys_refuse():
    with pytest.raises(ValueError):
        cap.parse(b'{"kind":1,"kind":2}')
    h = b"{'descr':'<f4','shape':(),'shape':(),'fortran_order':False}\n"
    with pytest.raises(ValueError):
        cap.npy_type(b'\x93NUMPY\x01\x00'+struct.pack('<H', len(h))+h+b'\x00'*4)


def test_stdlib_only_cold_import_and_actual_references():
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy'}:
            raise AssertionError('Scientific imports are forbidden in the cold reader')
sys.meta_path.insert(0, Block())
from models.predictor_replay_capsule_v1 import Capsule
with Capsule(sys.argv[1], manifest_sha256=sys.argv[2]) as r:
    assert len(r.states) == 2
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print('two actual CUDA references cold-validated; zero scientific imports')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(BUILD/'capsule'), MEASURES['manifest_sha256']],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert 'zero scientific imports' in result.stdout


def test_live_reference_bytes_rechecked_before_binding(copy_root):
    with open_reader(copy_root) as r:
        member = r.manifest['aliases']['same_replay_original']['loss.npy']['member']
        path = copy_root/member
        with changed(path, data=path.read_bytes()[:-1]+b'!'):
            ns = r.manifest['files'][member]['mtime_ns']; os.utime(path, ns=(ns, ns))
            with pytest.raises(ValueError):
                r.execution_binding('cd'*32)


def test_saved_archives_are_regular_complete_byte_identical_members():
    from scripts.prepare_predictor_replay_capsule_v1 import verify_archive
    raw = (BUILD/'capsule/manifest.json').read_bytes()
    first = verify_archive(BUILD/'predictor-replay-reference-v1.tar.gz', raw, MEASURES['archive_sha256'])
    second = json.loads((BUILD.parent/'compression-9.json').read_bytes())
    selected = verify_archive(BUILD.parent/'predictor-replay-reference-level9.tar.gz', raw, second['archive_sha256'])
    assert first == selected == {'members': 55, 'member_bytes': 217613344, 'verified': True}
    assert MEASURES['archive_plus_training_packet_fits_input_cap'] is False
    # Measurements, including an over-cap result, are evidence rather than permission.
    assert second['input_cap_bytes'] == MEASURES['existing_input_cap_bytes'] == 128*2**20


@pytest.mark.parametrize('kind', ['directory', 'regular'])
def test_same_device_gate_with_explicit_foreign_device_simulation(copy_root, kind):
    from types import SimpleNamespace
    # Open/verify the actual capsule after the device guard was added. The foreign
    # device below is an explicit stat-boundary simulation, not a Linux mount test.
    with open_reader(copy_root) as r:
        p = copy_root if kind == 'directory' else copy_root/'lineage/plan.json'
        info = p.stat()
        fake = SimpleNamespace(**{key: getattr(info, key) for key in
                                 ['st_mode','st_uid','st_nlink','st_size','st_dev']})
        fake.st_dev = r.anchor.st_dev+1
        with pytest.raises(ValueError):
            r._directory(fake) if kind == 'directory' else r._regular(fake, cap.MiB)
