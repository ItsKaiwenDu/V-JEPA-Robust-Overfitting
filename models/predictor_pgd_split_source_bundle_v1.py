"""Local source-only assembly; no authorization, upload or target work API.

Copies the 63 frozen runtime sources and six actual supervisor sources. Original
training pixels, dependencies and checkpoint are not read by this assembler.
"""
import hashlib
import os
from pathlib import Path
import stat
from models import predictor_pgd_split_bootstrap_v1 as boot
from models import predictor_pgd_split_coordinator_v1 as coordinator

FROZEN = {
 'predictor_pgd_split_bootstrap_v1.py':'4e977468a4e9fca4bda595983d0345bb6ed8c9f31ea1bcb7e8a0553b17a9b1f4',
 'predictor_pgd_split_coordinator_v1.py':'956f64be47bfae62bde77c04f44ad6f4d544895ee0fcadc9baadc06b5bcc0afb',
 'predictor_pgd_split_scratch_v1.py':'0ee1d5348867ff2579232c4441ac666aa94a757864892b7c588a5b582affc333',
 'predictor_pgd_split_watch_v1.py':'ab23159993939318b6d28946a25fe3c4be0427ed6e9732e1530cb8246d6d9cdb',
 'scratch_allocation_v2.py':'6b0503b8ab2c88a44a4d95c8fafbad52d1c0414293379406520a22f67700e874'}
RUNTIME = 'runs/pretrained/diagnostic/predictor-pgd-split-runtime-v1-20261115/UNEXECUTED-assembly-02'


def identity(s):
    return (s.st_dev,s.st_ino,s.st_uid,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def read_source(path, *, cap, sha256=None):
    path = Path(path)
    if (not path.is_absolute() or path.resolve(strict=True) != path
            or type(cap) is not int or not 0 < cap <= 2*2**20
            or sha256 is not None and not boot.sha(sha256)):
        raise ValueError('Canonical bounded local source path and digest required.')
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) not in (0o400,0o600,0o644) or not 0 < before.st_size <= cap):
        raise ValueError('Owned regular single-link bounded source required.')
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        if identity(os.fstat(fd)) != identity(before):
            raise ValueError('Held local source changed before read.')
        with os.fdopen(os.dup(fd),'rb') as stream:
            data = stream.read(cap+1)
        if (len(data) != before.st_size or identity(os.fstat(fd)) != identity(before)
                or identity(path.lstat()) != identity(before) or path.resolve(strict=True) != path
                or sha256 is not None and hashlib.sha256(data).hexdigest() != sha256):
            raise ValueError('Pinned local source bytes/identity changed.')
        return data
    finally:
        os.close(fd)


def assemble(repository, destination):
    """Prepare bytes only, in a fresh LOCAL directory; never creates a grant."""
    root,dest = Path(repository),Path(destination)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError('Canonical repository required.')
    policy = read_source(root/'configs/predictor_pgd_split_launch_v1.json',cap=256*1024,sha256=boot.POLICY_SHA)
    sources = {}
    for name in boot.SUPERVISORS.values():
        local = 'predictor_pgd_split_bootstrap_v1.py' if name=='bootstrap.py' else name
        sources[name] = read_source(root/'models'/local,cap=2*2**20,sha256=FROZEN.get(local))
    control = coordinator.make_control(policy,sources)
    c,p,csha = boot.control(control)
    ib,isha = boot.installation(c,csha)
    payload = {'supervisor/'+name:data for name,data in sources.items()}
    contract = boot.parsed(p['scratch_contract_text'])
    for name,want in contract['runtime_files'].items():
        boot.safe(name)
        data = read_source(root/RUNTIME/name,cap=want['bytes'],sha256=want['sha256'])
        if len(data) != want['bytes']:
            raise ValueError('Exact frozen runtime source extent required.')
        compile(data,'<UNEXECUTED-BUNDLE-'+name+'>','exec')
        payload['overlay/'+name] = data
    if len(payload) != 69:
        raise ValueError('Exactly 63 runtime and six actual supervisor sources required.')
    # Deliberately unarmed names. A future reviewed outer driver must bind/install
    # these under its own one-attempt mechanism; this is not a target directory.
    payload.update({'CONTROL-CANDIDATE.json':control,'INSTALLATION-CANDIDATE.json':ib,
                    'configs/base.json':p['base_text'].encode(),'configs/design.json':p['design_text'].encode()})
    if (not dest.is_absolute() or dest.parent.resolve(strict=True) != dest.parent or os.path.lexists(dest)
            or stat.S_IMODE(dest.parent.stat().st_mode)!=0o700 or dest.parent.stat().st_uid!=os.getuid()):
        raise ValueError('Fresh canonical private local output required; no overwrite.')
    dest.mkdir(mode=0o700)
    result = {}
    for name,data in sorted(payload.items()):
        path = dest/name
        q = dest
        for part in Path(name).parts[:-1]:
            q=q/part
            try:q.mkdir(mode=0o700)
            except FileExistsError:pass
            info=q.lstat()
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid()
                    or stat.S_IMODE(info.st_mode)!=0o700 or q.resolve(strict=True)!=q):
                raise ValueError('Private source assembly ancestry required.')
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as out:
            out.write(data);out.flush();os.fsync(out.fileno())
        result[name]={'bytes':len(data),'sha256':boot.digest(data)}
    return {'kind':'pgd-split-UNEXECUTED-source-bundle-v1','source_files':69,'runtime_files':63,'supervisor_files':6,
        'files':result,'control_sha256':csha,'installation_sha256':isha,'authorized':False,'target_helper_ready':False,
        'outer_attempt_setup_timer_and_guard_journals_integrated':False,'scientific_qualified':False,'training_admitted':False}
