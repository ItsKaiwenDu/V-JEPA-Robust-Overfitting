"""Same storage ceilings with an explicit frozen-capsule input category."""
from pathlib import Path
import os
import stat
import tarfile

from models import predictor_sdpa_handoff_operator_v1 as frozen

MiB = 2**20
GiB = 2**30
MASK_PATH = 'results/capture/mask-indices.json'
MASK_FILE_BYTES = 256 * 1024
from models.predictor_sdpa_continuation_contract_v6 import RESOURCE, KIND
from models import predictor_sdpa_storage_v2 as prior
RESOURCE = dict(RESOURCE)
COLLECTION_KIND = KIND


def typed_file_limit(name):
    frozen.old.safe_name(name)
    if name in ('native-input-capsule.tar.gz','native-input-capsule.tar.gz.partial'):
        return 128*MiB
    return prior.typed_file_limit(name)


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
        elif name in ('native-input-capsule.tar.gz','native-input-capsule.tar.gz.partial'):
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
    """Separate v6 collection envelope; never impersonate a consumed handoff."""
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
