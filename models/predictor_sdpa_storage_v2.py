"""Local v2 storage/collection policy; this module grants no target execution.

The exact mask-index artifact needs more than v1's 64 KiB scope-record cap.
All other typed, aggregate and allocated limits retain their frozen values.
No v1 function or module global is changed.
"""
from pathlib import Path
import os
import stat
import tarfile

from models import predictor_sdpa_handoff_operator_v1 as frozen

MiB = 2**20
GiB = 2**30
MASK_PATH = 'results/capture/mask-indices.json'
MASK_FILE_BYTES = 256 * 1024
RESOURCE = dict(frozen.RESOURCE)
COLLECTION_KIND = 'native-fixed-SDPA-storage-collection-v2'


def typed_file_limit(name):
    """Only the exact native mask artifact and its atomic partial get 256 KiB."""
    frozen.old.safe_name(name)
    stem = Path(name).name.removesuffix('.partial')
    limit = 16*MiB if stem.endswith(('.pt', '.npy', '.log')) else 8*MiB if stem.endswith('.jsonl') else MiB
    if name.endswith('.tar.gz'):
        limit = 128*MiB
    if stem == 'worker.json':
        limit = 256*1024
    if stem == 'boundary.json':
        limit = 128*1024
    if stem in ('attention-scopes.json', 'mask-indices.json'):
        limit = 64*1024
    if stem.endswith(('complete.json', 'failure.json')):
        limit = 16*1024
    if name == 'space-proof-v2.bin.partial':
        limit = 3*GiB
    if name == 'emergency-space.bin':
        limit = 16*MiB
    if name.startswith(('code/', 'overlay/')):
        limit = 32*MiB
    if name in (MASK_PATH, MASK_PATH + '.partial'):
        limit = MASK_FILE_BYTES
    return limit


def inventory(folder, *, hashing=False, running=False):
    files, allocated = frozen.base.inventory(folder, hashing=hashing, max_bytes=3*GiB, running=running)
    for name, item in files.items():
        limit = typed_file_limit(name)
        if item['bytes'] > limit:
            raise ValueError(f'Native SDPA typed-file ceiling exceeded: {name}: {item["bytes"]} > {limit} bytes.')
    return files, allocated


def storage(folder, identity, *, setup=False, parent=Path('/var/tmp')):
    # Same allocated/category accounting as v1, using only the corrected inventory.
    folder = Path(folder)
    s = folder.lstat()
    par = Path(parent).stat()
    if (folder.is_symlink() or not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid()
            or stat.S_IMODE(s.st_mode) != 0o700 or s.st_dev != identity['device']
            or (par.st_dev, par.st_ino) != (identity['device'], identity['inode'])):
        raise ValueError('Private same-device scratch changed.')
    files, allocated = inventory(folder, running=True)
    for current, dirs, names in os.walk(folder, followlinks=False):
        allocated += Path(current).stat().st_blocks*512
    fs = os.statvfs(folder)
    ceiling = RESOURCE['setup_bytes' if setup else 'capture_bytes']
    if allocated > ceiling or fs.f_bavail*fs.f_frsize < GiB or fs.f_favail < 1024:
        raise ValueError('Allocated/free/inode ceiling exceeded.')
    cats = dict.fromkeys(('input', 'code', 'raw', 'metadata', 'cache', 'emergency', 'proof'), 0)
    for name, item in files.items():
        if name == 'space-proof-v2.bin.partial':
            category = 'proof'
        elif name.startswith('original-packet.tar.gz'):
            category = 'input'
        elif name.startswith(('code/', 'overlay/')):
            category = 'code'
        elif name.startswith(('tmp/', 'cache/')):
            category = 'cache'
        elif name == 'emergency-space.bin':
            category = 'emergency'
        elif name.startswith('results/') and Path(name).name.removesuffix('.partial').endswith(('.pt', '.npy')):
            category = 'raw'
        else:
            category = 'metadata'
        cats[category] += item['bytes']
    caps = {key: RESOURCE[key+'_bytes'] for key in cats if key != 'proof'}
    caps['proof'] = RESOURCE['write_proof_bytes'] if setup else 0
    if any(cats[key] > limit for key, limit in caps.items()):
        raise ValueError('Native SDPA category ceiling exceeded.')
    return {'logical_bytes': sum(item['bytes'] for item in files.values()), 'allocated_bytes': allocated,
            'categories': cats, 'ceiling': ceiling, 'files': len(files)}


def stream(folder, sha, output):
    """Separate v2 collection envelope; never impersonate a consumed v1 handoff."""
    files, _ = inventory(folder, hashing=True)
    for name, item in files.items():
        item['mode'] = stat.S_IMODE((Path(folder)/name).stat().st_mode)
    header = frozen.old.json_bytes({'kind': COLLECTION_KIND, 'phase': 'collection', 'control_sha256': sha,
        'files': files, 'logical_bytes': sum(item['bytes'] for item in files.values()), 'model_restarted': False})
    extent = sum(512+((item['bytes']+511)//512)*512 for item in files.values())+1024
    if len(header) > MiB or len(header)+((extent+10239)//10240)*10240 > RESOURCE['collection_bytes']:
        raise ValueError('Complete collection byte ceiling exceeded.')
    output.write(header)
    output.flush()
    with tarfile.open(fileobj=output, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
        for name, item in sorted(files.items()):
            path = Path(folder)/name
            if frozen.old.digest(path) != item['sha256']:
                raise ValueError('Collection source changed.')
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = item['bytes'], item['mode'], item['mtime_ns']//10**9
            with path.open('rb') as source:
                archive.addfile(member, source)
