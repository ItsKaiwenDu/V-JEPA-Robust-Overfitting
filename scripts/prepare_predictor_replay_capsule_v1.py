"""Local-only reference capsule builder. No model, SSH, proof or authorization."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import predictor_replay_capsule_v1 as capsule

PRIOR_SEAL = 'runs/pretrained/diagnostic/scratch-inventory-observer-v1-20261114/sealed-inputs.json'


def source(root, path, expected=None):
    p = root/path
    before = p.lstat()
    if (p.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid()):
        raise ValueError('Original single-link regular source required: ' + path)
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        opened = os.fstat(f.fileno()); data = f.read(); after = os.fstat(f.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_mtime_ns)
    if any(signature(s) != signature(before) for s in (opened, after, p.lstat())):
        raise ValueError('Source changed while reading.')
    item = {'path': path, 'bytes': len(data), 'sha256': capsule.digest(data),
            'mtime_ns': before.st_mtime_ns, 'mode': stat.S_IMODE(before.st_mode)}
    if expected is not None and item != expected:
        raise ValueError('Original source byte/mode/nsmtime descriptor changed: ' + path)
    return data, item


def build(root, output, *, compresslevel=6):
    if compresslevel not in (6, 9):
        raise ValueError('Use only measured gzip levels 6 or 9.')
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Preserve any existing capsule or partial build.')
    seal = json.loads((root/PRIOR_SEAL).read_bytes())
    for path, sha in seal.items():
        with (root/path).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != sha:
                raise ValueError('Prior sealed input changed: ' + path)
    lineage = {}
    for name, (path, sha) in capsule.LINEAGE.items():
        data, item = source(root, path)
        if capsule.digest(data) != sha:
            raise ValueError('Pinned lineage source changed.')
        lineage[name] = (data, item)
    plan = capsule.parse(lineage['plan.json'][0]); pins = capsule.parse(lineage['target-pins.json'][0])
    selected, aliases = capsule.recipe(plan, pins)
    selected.update({'lineage/'+name: item for name, (_, item) in lineage.items()})
    os.umask(0o077); output.mkdir(mode=0o700, parents=True)
    folder = output/'capsule'; folder.mkdir(mode=0o700)
    files = {}
    for name, expected in sorted(selected.items()):
        data, item = source(root, expected['path'], expected)
        path = folder/name; path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.chmod(path, 0o600); os.utime(path, ns=(item['mtime_ns'], item['mtime_ns']))
        files[name] = {k: item[k] for k in ('bytes', 'sha256', 'mtime_ns')}
        files[name].update(mode=0o600, source=item)
    manifest = {'kind': capsule.KIND, 'reference_only': True, 'execution_permitted': False,
                'source_control_sha256': capsule.SOURCE_CONTROL, 'files': files,
                'aliases': aliases, 'frozen_code': capsule.FROZEN_CODE}
    raw = capsule.encode(manifest); manifest_sha = capsule.digest(raw)
    with (folder/'manifest.json').open('xb') as f:
        f.write(raw)
    with capsule.Capsule(folder, manifest_sha256=manifest_sha) as reader:
        states = {r: {'source_index': s['index'], 'mode': s['mode'], 'replay': s['replay'],
                      'control_sha256': s['control_sha256'], 'artifacts': len(s['artifacts']),
                      'ownership': s['ownership']} for r, s in reader.states.items()}
    archive = output/'predictor-replay-reference-v1.tar.gz'
    with archive.open('xb') as f, gzip.GzipFile(filename='', fileobj=f, mode='wb', compresslevel=compresslevel, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
            for name in sorted({'manifest.json'} | set(files)):
                path = folder/name; info = path.lstat()
                member = tarfile.TarInfo(name); member.mode = 0o600; member.size = info.st_size
                member.mtime = info.st_mtime_ns//10**9
                with path.open('rb') as f:
                    tar.addfile(member, f)
    with archive.open('rb') as f:
        archive_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    measurements = {'kind': 'local-replay-capsule-measurements-v1', 'archive_bytes': archive.stat().st_size,
                    'archive_sha256': archive_sha, 'manifest_sha256': manifest_sha, 'compresslevel': compresslevel,
                    'members': len(files)+1, 'member_bytes': sum(i['bytes'] for i in files.values())+len(raw),
                    'array_blobs': 37, 'array_aliases': 72, 'logical_array_alias_bytes': 404104200,
                    'unique_array_file_bytes': 214635140, 'reference_proof_record_bytes': 2666936,
                    'lineage_bytes': sum(i['bytes'] for n,i in files.items() if n.startswith('lineage/')),
                    'manifest_bytes': len(raw), 'allocation_bytes': sum(p.stat().st_blocks*512 for p in folder.rglob('*') if p.is_file()),
                    'old_training_packet_bytes': 17192095, 'existing_input_cap_bytes': 128*2**20,
                    'archive_plus_training_packet_bytes': archive.stat().st_size+17192095,
                    'archive_plus_training_packet_fits_input_cap': archive.stat().st_size+17192095 <= 128*2**20,
                    'prior_sealed_files_verified': len(seal), 'references': states,
                    'old_parent_failed': True, 'candidate_fresh_replay_open': True,
                    'new_target_execution_permitted': False, 'new_scientific_or_remote_work': False}
    verify_archive(archive, raw, archive_sha)
    with (output/'measurements.json').open('xb') as f:
        f.write(capsule.encode(measurements))
    return measurements


def verify_archive(archive, manifest_data, expected_archive_sha256):
    """Cold stream check of every regular member; no extraction or tensor loads."""
    archive = Path(archive)
    with archive.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != expected_archive_sha256:
            raise ValueError('Exact archive identity required.')
    manifest = capsule.parse(manifest_data)
    expected = dict(manifest['files'])
    expected['manifest.json'] = {'bytes': len(manifest_data), 'sha256': capsule.digest(manifest_data), 'mode': 0o600}
    seen = set()
    with tarfile.open(archive, mode='r|gz') as tar:
        for member in tar:
            name = member.name; capsule.relative(name)
            if name in seen or name not in expected or member.type != tarfile.REGTYPE or member.pax_headers:
                raise ValueError('Only exact unique regular USTAR members may be imported.')
            item = expected[name]
            if (member.size, member.mode) != (item['bytes'], 0o600):
                raise ValueError('Archive member extent or mode changed.')
            if name != 'manifest.json' and member.mtime != item['mtime_ns']//10**9:
                raise ValueError('Archive member timestamp changed.')
            with tar.extractfile(member) as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != item['sha256']:
                    raise ValueError('Archive member content changed.')
            seen.add(name)
    if seen != set(expected):
        raise ValueError('Exact archive member coverage required.')
    return {'members': len(seen), 'member_bytes': sum(v['bytes'] for v in expected.values()), 'verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compression-level', type=int, choices=(6, 9), default=6)
    args = parser.parse_args()
    print(json.dumps(build(ROOT, args.output, compresslevel=args.compression_level), sort_keys=True))
