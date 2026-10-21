"""Standard-library-only, checksummed offline handoff creation and restoration."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

UPSTREAM = '204698b45b3712590f06245fbfba32d3be539812'
CHECKPOINT = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or str(path) != name or name == '.'):
        raise ValueError('Unsafe archive/member path.')
    return name


def make_archive(destination, files):
    """Stable ordering, timestamps and ownership; no symlinks or implicit directories."""
    entries = {}
    with Path(destination).open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=3) as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for name, filename in sorted(files.items()):
                safe_name(name)
                path = Path(filename)
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Only regular input files may be packaged.')
                checksum = digest(path)
                info = tarfile.TarInfo(name)
                info.size, info.mode = path.stat().st_size, 0o644
                with path.open('rb') as handle:
                    archive.addfile(info, handle)
                if digest(path) != checksum:
                    raise ValueError('Input changed during archive creation.')
                entries[name] = {'sha256': checksum, 'bytes': info.size}
    return {'sha256': digest(destination), 'bytes': Path(destination).stat().st_size, 'files': entries}


def extract_verified(archive_path, specification, destination):
    """Validate every declared regular member; no tar.extractall or symlink following."""
    if digest(archive_path) != specification['sha256']:
        raise ValueError('Archive checksum mismatch.')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        names = [safe_name(m.name) for m in members]
        if (len(names) != len(set(names)) or set(names) != set(specification['files'])
                or any(not m.isfile() for m in members)):
            raise ValueError('Archive members differ from manifest or contain non-regular files.')
        for member in members:
            expected = specification['files'][member.name]
            if member.size != expected['bytes']:
                raise ValueError('Archive member size mismatch.')
            with archive.extractfile(member) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected['sha256']:
                    raise ValueError('Archive member checksum mismatch.')
        destination.mkdir(parents=True)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, target.open('xb') as out:
                shutil.copyfileobj(stream, out)


def build(root, output, wheelhouse, extra=()):
    root, output, wheelhouse = Path(root).resolve(), Path(output).resolve(), Path(wheelhouse).resolve()
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    names = sorted(set(n for n in tracked if n) | set(extra))
    source = {}
    for name in names:
        safe_name(name)
        path = root / name
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or path.stat().st_size > 10 * 2**20
                or path.suffix in ('.pt', '.pth', '.whl', '.pem', '.key')
                or any(p in ('.git', '.venv', '.ssh') for p in path.parts)
                or path.name == '.env' or path.name.startswith('.env.')):
            raise ValueError('Unexpected large/private/non-source input: ' + name)
        source[name] = path
    prepared = root / 'data/processed/staged-exploration-v1'
    index = json.loads((prepared / 'train/index.json').read_text())
    training = {'preparation.json': prepared / 'preparation.json', 'train/index.json': prepared / 'train/index.json'}
    if len(index['clips']) != 100:
        raise ValueError('Expected the original complete 100-clip training cache.')
    for row in index['clips']:
        if Path(row['file']).name != row['file']:
            raise ValueError('Invalid training cache filename.')
        training['train/' + row['file']] = prepared / 'train' / row['file']
    wheels = {p.name: p for p in wheelhouse.glob('*.whl')}
    if not wheels or any(p.startswith(('torch-', 'torchvision-', 'nvidia_')) for p in wheels):
        raise ValueError('Provide non-torch Linux dependency wheels; preserve the template CUDA stack.')
    upstream = root / 'external/vjepa2'
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    # Match the model adapter's relevant-tree guard. On case-insensitive macOS,
    # unrelated V-JEPA 2.1 vitG/vitg configurations collide in the checkout.
    # The bundle contains committed Git objects, never those working files.
    if revision != UPSTREAM or subprocess.check_output(
            ['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=normal',
             '--', 'src', 'app/vjepa', 'configs/train/vitl16'], text=True).strip():
        raise ValueError('Upstream scientific files must be clean and exactly pinned.')
    output.mkdir(parents=True, exist_ok=False)
    archives = {name: make_archive(output / (name + '.tar.gz'), files)
                for name, files in (('source', source), ('training', training), ('wheels', wheels))}
    subprocess.run(['git', '-C', str(upstream), 'bundle', 'create', str(output / 'upstream.bundle'), 'HEAD'], check=True)
    shutil.copyfile(root / 'models/pilot_bundle.py', output / 'restore.py')
    manifest = {'schema_version': 1, 'diagnostic_only': True, 'research_admitted': False,
                'base_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                'source_is_working_tree_snapshot': True, 'archives': archives,
                'upstream_commit': UPSTREAM, 'upstream_bundle_sha256': digest(output / 'upstream.bundle'),
                'upstream_bundle_uses_committed_objects_only': True,
                'restore_sha256': digest(output / 'restore.py'), 'checkpoint_sha256': CHECKPOINT,
                'checkpoint_bytes': 5127726842,
                'checkpoint_url': 'https://dl.fbaipublicfiles.com/vjepa2/vitl.pt',
                'packaged_training_clips': 100, 'executed_training_clips': 0,
                'held_out_pixels_included': False, 'cuda_execution_performed': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def restore(package, destination):
    package, destination = Path(package).resolve(), Path(destination).resolve()
    manifest = json.loads((package / 'manifest.json').read_text())
    if (manifest['schema_version'] != 1 or manifest['upstream_commit'] != UPSTREAM
            or manifest['checkpoint_sha256'] != CHECKPOINT
            or digest(package / 'restore.py') != manifest['restore_sha256']
            or digest(package / 'upstream.bundle') != manifest['upstream_bundle_sha256']):
        raise ValueError('Handoff identity differs from its pinned contract.')
    if destination.exists():
        raise FileExistsError(destination)
    extract_verified(package / 'source.tar.gz', manifest['archives']['source'], destination)
    extract_verified(package / 'training.tar.gz', manifest['archives']['training'],
                     destination / 'data/processed/staged-exploration-v1')
    extract_verified(package / 'wheels.tar.gz', manifest['archives']['wheels'], destination / 'offline-wheels')
    upstream = destination / 'external/vjepa2'
    subprocess.run(['git', 'clone', str(package / 'upstream.bundle'), str(upstream)], check=True)
    subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', UPSTREAM], check=True)
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM:
        raise ValueError('Restored upstream revision differs.')
    (destination / 'handoff-manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify and restore a prepared offline workflow handoff.')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    restore(args.package, args.destination)
    print('Handoff restored and verified. CUDA setup and execution have not run.')
