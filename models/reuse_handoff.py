"""One-training-clip, source-pinned university diagnostic handoff; no deployment."""
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess

from models.candidate_contract import binding
from models.pilot_bundle import CHECKPOINT, UPSTREAM, digest, extract_verified, make_archive, safe_name

PACKET_SHA = 'bf2b124b9fff819cc9f1b3acbc095c208c8a4df1293cda24267b802edd5d0b8a'
HANDOFF_KIND = 'projection-reuse-university-handoff-v1'
NEW_SOURCES = (
    'models/projection_reuse.py', 'models/reuse_execution.py', 'models/reuse_handoff.py',
    'models/reuse_diagnostic.py', 'scripts/check_projection_reuse.py',
    'scripts/university_projection_reuse.sh', 'scripts/prepare_projection_reuse.py',
    'tests/test_projection_reuse.py', 'tests/test_reuse_diagnostic.py',
    'tests/test_reuse_execution.py',
)


def build(root, output):
    from models.vjepa2 import use_upstream
    root, output = Path(root).resolve(), Path(output).resolve()
    old = root/'runs/pretrained/diagnostic/candidate-eot8-handoff-v4-20261028/manifest.json'
    if digest(old) != PACKET_SHA:
        raise ValueError('Frozen source packet changed.')
    original = json.loads(old.read_text())
    if binding() != original['candidate']['candidate_binding']:
        raise ValueError('The original 61 scientific pins changed.')
    files = {}
    for name, item in original['archives']['source']['files'].items():
        p=root/name
        if digest(p) != item['sha256']:
            raise ValueError('Frozen source file changed: '+name)
        files[name]=p
    for name in NEW_SOURCES:
        p=root/safe_name(name)
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 2**20:
            raise ValueError('Missing or invalid diagnostic source: '+name)
        files[name]=p
    prepared=root/'data/processed/staged-exploration-v1/train'
    index=json.loads((prepared/'index.json').read_text())
    train=root/'data/manifests/ucf101-staged-20x5-v1/train.csv'
    if index['split']!='train' or len(index['clips'])!=100 or index['source_manifest_sha256']!=digest(train):
        raise ValueError('Original 100-clip training cache identity changed.')
    row=index['clips'][0]
    if Path(row['file']).name!=row['file']:
        raise ValueError('Invalid prepared training filename.')
    matches=[r for r in csv.DictReader(io.StringIO(train.read_text())) if r['video_id']==row['video_id']]
    if len(matches)!=1 or any(matches[0][k]!=row[k] for k in ('source_group_id','class_id','relative_path')):
        raise ValueError('Selected tensor is not the declared training row.')
    clip=prepared/row['file']
    if clip.is_symlink() or not clip.is_file():
        raise ValueError('Require the existing regular training tensor.')
    files['input/training-clip.pt']=clip
    files['input/train-index.json']=prepared/'index.json'
    upstream=use_upstream(root/'external/vjepa2')
    upstream_files={str(p.relative_to(upstream)):digest(p) for p in sorted((upstream/'src').rglob('*.py'))}
    recipe='configs/train/vitl16/pretrain-256px-16f.yaml'
    upstream_files[recipe]=digest(upstream/recipe)
    output.mkdir(parents=True,exist_ok=False)
    stage=output/'handoff';stage.mkdir()
    for name,p in files.items():
        q=stage/name;q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,q)
    metadata={'kind':HANDOFF_KIND,'schema_version':1,'diagnostic_only':True,'research_admitted':False,
        'paid_compute':False,'CUDA_execution_performed':False,'launch_authorized':False,
        'checkpoint_included':False,'held_out_pixels_included':False,'packaged_training_clips':1,
        'checkpoint_sha256':CHECKPOINT,'upstream_commit':UPSTREAM,'upstream_sha256':upstream_files,
        'parent_packet_sha256':PACKET_SHA,'parent_candidate_binding':binding(),
        'training_row':row,'training_manifest_sha256':digest(train),
        'files':{n:{'sha256':digest(stage/n),'bytes':(stage/n).stat().st_size} for n in sorted(files)},
        'proposed_limits':{'max_seconds':4800,'worker_seconds':1080,'allocator_gib':7,'workers':4,'automatic_retries':0},
        'source_is_working_tree_snapshot':True,
        'base_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()}
    (stage/'REUSE-HANDOFF.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    verify(stage,digest(stage/'REUSE-HANDOFF.json'))
    archive=make_archive(output/'vjepa-projection-reuse.tar.gz',
        {str(p.relative_to(stage)):p for p in stage.rglob('*') if p.is_file()})
    (output/'vjepa-projection-reuse.tar.gz.manifest.json').write_text(json.dumps(archive,indent=2,sort_keys=True)+'\n')
    return metadata,archive


def verify(root, expected_sha256):
    root=Path(root).resolve();path=root/'REUSE-HANDOFF.json'
    if path.is_symlink() or digest(path)!=expected_sha256:
        raise ValueError('Handoff metadata checksum mismatch.')
    m=json.loads(path.read_text())
    if (m.get('kind')!=HANDOFF_KIND or m.get('schema_version')!=1 or not m.get('diagnostic_only')
            or m.get('research_admitted') is not False or m.get('paid_compute') is not False
            or m.get('launch_authorized') is not False or m.get('packaged_training_clips')!=1
            or m.get('held_out_pixels_included') is not False or m.get('checkpoint_included') is not False
            or m.get('checkpoint_sha256')!=CHECKPOINT or m.get('upstream_commit')!=UPSTREAM):
        raise ValueError('Wrong university diagnostic handoff domain.')
    for name,item in m['files'].items():
        p=root/safe_name(name)
        if (p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root)
                or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']):
            raise ValueError('Changed handoff member: '+name)
    for name,expected in m['parent_candidate_binding']['source_sha256'].items():
        if name not in m['files'] or m['files'][name]['sha256']!=expected:
            raise ValueError('Missing or changed original scientific pin: '+name)
    for name in NEW_SOURCES:
        if name not in m['files']:raise ValueError('Missing diagnostic source: '+name)
    return m


def training_input(root, metadata):
    import hashlib
    import torch
    root=Path(root);row=metadata['training_row']
    index=json.loads((root/'input/train-index.json').read_text())
    if index['clips'][0]!=row or index['source_manifest_sha256']!=metadata['training_manifest_sha256']:
        raise ValueError('Training index differs from the declared original first clip.')
    p=root/'input/training-clip.pt'
    if digest(p)!=metadata['files']['input/training-clip.pt']['sha256']:
        raise ValueError('Training tensor file changed before loading.')
    value=torch.load(p,map_location='cpu',weights_only=True)
    pixels=value['pixels']
    if (value['metadata']!={k:v for k,v in row.items() if k!='file'}
            or pixels.dtype!=torch.float32 or tuple(pixels.shape)!=(3,16,256,256)
            or not torch.isfinite(pixels).all() or pixels.min()<0 or pixels.max()>1
            or hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest()!=row['pixel_sha256']):
        raise ValueError('Original training pixels or metadata failed verification.')
    return pixels.unsqueeze(0)
