"""Build a local, verified frozen-input capsule. No model, GPU or connection."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REVIEW='report/checks/2026-11-07-predictor-sdpa-handoff-v2-target-review.json'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def selected_inputs(root):
    root=Path(root);report=json.loads((root/REVIEW).read_text())
    if not report['native_capture_accepted'] or report['component_workers_complete']!=0:
        raise ValueError('Require the reviewed native capture and preserved startup refusal.')
    pin=report['input_pins'];pins_path=root/pin['path']
    if digest(pins_path)!=pin['sha256']:
        raise ValueError('Reviewed target input manifest changed.')
    pins=json.loads(pins_path.read_text());saved=root/report['session']/'saved'
    control=json.loads((saved/'control.json').read_text());items={}
    excluded={'input/training-clip.pt','input/prior-EOT8-derivatives.pt'}
    names={f'code/{name}':saved/'code'/name for name in control['archive_manifest']['files'] if name not in excluded}
    capture=saved/'results/capture'
    names.update({str(p.relative_to(saved)):p for p in capture.rglob('*') if p.is_file()})
    names.update({'input/parent-control.json':saved/'control.json','input/parent-session.json':saved/'session.json',
                  'input/parent-review.json':root/REVIEW})
    if len(names)!=118 or sum(n.startswith('code/') for n in names)!=97:
        raise ValueError('Exact reviewed source/native capture coverage required.')
    for name,path in sorted(names.items()):
        key=str(path.relative_to(root));s=path.lstat()
        if path.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:
            raise ValueError('Only original regular single-link inputs may be copied.')
        sha=digest(path)
        if key in pins:
            pin=pins[key]
            if (s.st_size,s.st_mtime_ns,sha)!=(pin['bytes'],pin['mtime_ns'],pin['sha256']):
                raise ValueError('Frozen capture/source input changed: '+name)
        elif key!=REVIEW:
            raise ValueError('Unexpected capsule input: '+name)
        items[name]={'source':key,'bytes':s.st_size,'sha256':sha,'mtime_ns':s.st_mtime_ns,'source_mode':stat.S_IMODE(s.st_mode),'mode':0o600}
    return items


def verify_capsule(archive,manifest):
    if digest(archive)!=manifest['sha256'] or Path(archive).stat().st_size!=manifest['bytes']:
        raise ValueError('Capsule archive identity changed.')
    seen=set()
    with tarfile.open(archive,mode='r|gz') as tar:
        for member in tar:
            name=member.name;path=PurePosixPath(name)
            if (name in seen or name not in manifest['files'] or not member.isfile() or path.is_absolute()
                    or '..' in path.parts or str(path)!=name or '\\' in name):
                raise ValueError('Unexpected/unsafe capsule member.')
            item=manifest['files'][name]
            if item['mode']!=0o600 or (member.size,member.mode,member.mtime)!=(item['bytes'],item['mode'],item['mtime_ns']//10**9):
                raise ValueError('Capsule metadata changed.')
            with tar.extractfile(member) as source:
                if hashlib.file_digest(source,'sha256').hexdigest()!=item['sha256']:
                    raise ValueError('Capsule member changed.')
            seen.add(name)
    if seen!=set(manifest['files']):
        raise ValueError('Capsule member coverage changed.')
    return True


def build(root,output):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError('Preserve any previous capsule.')
    items=selected_inputs(root);os.umask(0o077);output.mkdir(mode=0o700,parents=True)
    archive=output/'native-sdpa-capsule.tar.gz'
    with archive.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
            for name,item in items.items():
                path=root/item['source']
                if digest(path)!=item['sha256']:raise ValueError('Capsule source changed before copy.')
                member=tarfile.TarInfo(name);member.size=item['bytes'];member.mode=item['mode'];member.mtime=item['mtime_ns']//10**9
                with path.open('rb') as source:tar.addfile(member,source)
                if digest(path)!=item['sha256'] or path.stat().st_mtime_ns!=item['mtime_ns']:
                    raise ValueError('Capsule source changed during copy.')
    manifest={'kind':'frozen-native-SDPA-capsule-v1','bytes':archive.stat().st_size,'sha256':digest(archive),
              'files':items,'logical_bytes':sum(i['bytes'] for i in items.values()),
              'capture_control_sha256':'fd9481d6295880c8c97aac32ee7705fb4b5a919a76d30bb3809f0f780d598609',
              'excluded_original_training_pixel_file':True,'excluded_prior_duplicate_gradient_file':True,
              'new_model_gradient_GPU_or_target_work':False,'new_target_execution_permitted':False}
    verify_capsule(archive,manifest)
    with (output/'manifest.json').open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    m=build(ROOT,a.output)
    print(json.dumps({k:m[k] for k in ('kind','bytes','sha256','logical_bytes','new_target_execution_permitted')},sort_keys=True))
