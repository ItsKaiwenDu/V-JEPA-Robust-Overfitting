"""Extend the verified one-clip handoff; preserve every original source pin."""
import json
from pathlib import Path
import shutil
import subprocess

import torch

from models.pilot_bundle import digest,make_archive,safe_name
from models.reuse_handoff import verify as verify_parent
from models.study_artifacts import read_record

PARENT_SHA='851c77b353bd572e15ad6d31293253226c9915567b05b8f7b3772da0cbe5a490'
KIND='eot8-work-profile-handoff-v1'
SOURCES=('models/eot_profile_events.py','models/eot_profile.py','models/eot_profile_packet.py',
         'scripts/check_eot_profile.py','scripts/prepare_eot_profile.py',
         'scripts/university_eot_profile.sh','tests/test_eot_profile.py')


def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    base=root/'runs/pretrained/diagnostic/projection-reuse-university-20261029'
    prior=base/'case-download-lj8i88ut/unpacked';parent=verify_parent(base/'packet/handoff',PARENT_SHA)
    if digest(prior/'REUSE-HANDOFF.json')!=PARENT_SHA:raise ValueError('Prior handoff changed.')
    review=json.loads((root/'report/checks/2026-10-29-university-projection-reuse-review.json').read_text())
    if not review['passed'] or review['archive_sha256']!='c62df1a3f4c7f653a4afac3d3afeeee82cd42ba0af465f4f9fa6b780dd0d9a1a':
        raise ValueError('The independently reviewed original result is required.')
    worker_path=prior/'results/workers/recompute-0/worker.json';worker=read_record(worker_path)
    session=read_record(prior/'results/session.json')
    if digest(worker_path)!=session['tasks'][0]['worker_sha256'] or not worker['complete'] or not worker['passed']:
        raise ValueError('Prior reference worker changed.')
    item=worker['artifacts']['gradients'];p=worker_path.parent/item['file']
    if digest(p)!=item['sha256']:raise ValueError('Prior reference derivative file changed.')
    values=torch.load(p,map_location='cpu',weights_only=True)
    output.mkdir(parents=True,exist_ok=False);stage=output/'handoff';stage.mkdir()
    files={name:base/'packet/handoff'/name for name in parent['files']}
    files['REUSE-HANDOFF.json']=base/'packet/handoff/REUSE-HANDOFF.json'
    for name in SOURCES:files[name]=root/name
    for name,p in files.items():
        if p.is_symlink() or not p.is_file():raise ValueError('Require regular source members.')
        q=stage/safe_name(name);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    with (stage/'input/prior-EOT8-derivatives.pt').open('xb') as f:
        torch.save({name:values['recompute_'+name] for name in ('input','loss')},f)
    (stage/'input/prior-contract.json').write_text(json.dumps(worker['contract'],indent=2,sort_keys=True)+'\n')
    metadata={'kind':KIND,'schema_version':1,'diagnostic_only':True,'research_admitted':False,
        'launch_authorized':False,'paid_compute':False,'parent_handoff_sha256':PARENT_SHA,
        'existing_checkpoint_sha256':parent['checkpoint_sha256'],'upstream_commit':parent['upstream_commit'],
        'prior_reference_worker_sha256':digest(worker_path),'prior_review_sha256':digest(root/'report/checks/2026-10-29-university-projection-reuse-review.json'),
        'prior_derivatives_source_sha256':item['sha256'],'packaged_training_clips':1,'held_out_pixels_included':False,
        'checkpoint_included':False,'new_sources':list(SOURCES),'base_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'scope':'Full clean EOT8 controls, detailed first-draw/two-family work attribution; no PGD, scoring, optimizer or speed optimization.',
        'limits':{'max_seconds':1200,'worker_seconds':480,'allocator_gib':7,'workers':2,'automatic_retries':0},
        'files':{str(p.relative_to(stage)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(stage.rglob('*')) if p.is_file()}}
    (stage/'PROFILE-HANDOFF.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    verify(stage,digest(stage/'PROFILE-HANDOFF.json'))
    spec=make_archive(output/'vjepa-eot-profile.tar.gz',{str(p.relative_to(stage)):p for p in stage.rglob('*') if p.is_file()})
    (output/'vjepa-eot-profile.tar.gz.manifest.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    return metadata,spec


def verify(root,expected):
    root=Path(root).resolve();path=root/'PROFILE-HANDOFF.json'
    if path.is_symlink() or digest(path)!=expected:raise ValueError('Profile handoff checksum mismatch.')
    m=json.loads(path.read_text())
    if (m.get('kind')!=KIND or m.get('schema_version')!=1 or m.get('diagnostic_only') is not True
            or m.get('launch_authorized') is not False or m.get('research_admitted') is not False
            or m.get('paid_compute') is not False or m.get('checkpoint_included') is not False
            or m.get('packaged_training_clips')!=1 or m.get('held_out_pixels_included') is not False
            or m.get('parent_handoff_sha256')!=PARENT_SHA or m.get('new_sources')!=list(SOURCES)
            or m.get('limits')!={'max_seconds':1200,'worker_seconds':480,'allocator_gib':7,'workers':2,'automatic_retries':0}):
        raise ValueError('Wrong one-clip profile artifact domain or limits.')
    for name,item in m['files'].items():
        p=root/safe_name(name)
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file() or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
            raise ValueError('Changed profile handoff member: '+name)
    parent=verify_parent(root,PARENT_SHA)
    if any(name not in m['files'] for name in (*SOURCES,'input/prior-EOT8-derivatives.pt','input/prior-contract.json','REUSE-HANDOFF.json')):
        raise ValueError('Missing profile source or reference.')
    return m,parent
