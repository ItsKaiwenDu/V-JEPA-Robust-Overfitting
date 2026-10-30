"""Source-only attention-component handoff; no clips, checkpoints or deployment."""
import json
from pathlib import Path
import shutil
import subprocess

from models.candidate_contract import binding
from models.eot_profile_packet import verify as verify_profile
from models.pilot_bundle import digest,make_archive,safe_name
from models.study_artifacts import read_record

KIND='attention-component-handoff-v1'
PROFILE_SHA='c95c0b21ee018dfd6c897756baf453dc7d69eb1d52b5c64d99e1ac4b4b945dbd'
ROLES={'predictor':('predictor',),'encoder':('encoder-0','encoder-1')}
HOSTS={'linux10609':{'role':'predictor','uuid':'GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52'},
       'linux10603':{'role':'encoder','uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f'}}
SOURCES=('models/attention_component.py','models/attention_diagnostic.py','models/attention_packet.py',
         'scripts/check_attention_component.py','scripts/prepare_attention_component.py',
         'scripts/university_attention_component.sh','tests/test_attention_component.py','tests/test_attention_diagnostic.py')
LIMITS={'max_seconds':1200,'worker_seconds':120,'allocator_gib':4,'warmup':1,'repeats':3,'automatic_retries':0}


def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    base=root/'runs/pretrained/diagnostic/eot-work-profile-university-20261029'
    m,parent=verify_profile(base/'packet/handoff',PROFILE_SHA)
    prior=base/'case-download-j10olicf/unpacked';session=read_record(prior/'results/session.json')
    path=prior/'results/workers/0/worker.json';worker=read_record(path)
    review=root/'report/checks/2026-10-30-eot-work-profile-review.json'
    if (digest(path)!=session['tasks'][0]['worker_sha256'] or not worker['passed'] or not worker['complete']
            or not json.loads(review.read_text())['independent_scientific_review_complete']):
        raise ValueError('Require the reviewed actual profile reference.')
    if binding()!=parent['parent_candidate_binding']:raise ValueError('Original scientific sources changed.')
    output.mkdir(parents=True,exist_ok=False);stage=output/'handoff';stage.mkdir()
    files={n:base/'packet/handoff'/n for n in m['files'] if not n.startswith('input/') and n!='REUSE-HANDOFF.json'}
    for name in SOURCES:files[name]=root/name
    for name,p in files.items():
        if p.is_symlink() or not p.is_file():raise ValueError('Only regular source files are accepted.')
        target=stage/safe_name(name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    c=worker['contract']
    reference={'kind':'attention-shape-reference-v1','synthetic_operands_only':True,
        'prior_profile_manifest_sha256':PROFILE_SHA,'prior_worker_sha256':digest(path),'prior_review_sha256':digest(review),
        'video_id_for_mask_geometry_only':c['data']['video_id'],'attack_seed':c['seeds']['attack'][0],
        'first_draw_mask_signatures':c['attack_masks'][0],
        'shapes':{'predictor':{'heads':12,'head_dim':32,'tokens':2048},
                  **{f'encoder-{i}':{'heads':16,'head_dim':64,'tokens':family[0]['shape'][-1]} for i,family in enumerate(c['attack_masks'][0])}}}
    (stage/'attention-reference.json').write_text(json.dumps(reference,indent=2,sort_keys=True)+'\n')
    metadata={'kind':KIND,'schema_version':1,'diagnostic_only':True,'paid_compute':False,'research_admitted':False,
        'launch_authorized':False,'checkpoint_included':False,'pixel_tensors_included':False,
        'synthetic_operands_only':True,'original_scientific_binding':binding(),'upstream_commit':parent['upstream_commit'],
        'upstream_sha256':parent['upstream_sha256'],'new_sources':list(SOURCES),'limits':LIMITS,
        'roles':{k:list(v) for k,v in ROLES.items()},'hosts':HOSTS,
        'base_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'scope':'Standalone FP32 efficient/math SDPA with upstream synthetic post-RoPE operands; no trained activations or model execution change.',
        'files':{str(p.relative_to(stage)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(stage.rglob('*')) if p.is_file()}}
    if any(name not in metadata['files'] for name in binding()['source_sha256']):raise ValueError('Missing original scientific pin.')
    (stage/'ATTENTION-HANDOFF.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    verify(stage,digest(stage/'ATTENTION-HANDOFF.json'))
    spec=make_archive(output/'vjepa-attention-components.tar.gz',{str(p.relative_to(stage)):p for p in stage.rglob('*') if p.is_file()})
    (output/'vjepa-attention-components.tar.gz.manifest.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    return metadata,spec


def verify(root,expected):
    root=Path(root).resolve();path=root/'ATTENTION-HANDOFF.json'
    if path.is_symlink() or digest(path)!=expected:raise ValueError('Attention handoff checksum changed.')
    m=json.loads(path.read_text())
    if (m.get('kind')!=KIND or m.get('schema_version')!=1 or m.get('diagnostic_only') is not True
            or m.get('paid_compute') is not False or m.get('research_admitted') is not False
            or m.get('launch_authorized') is not False or m.get('checkpoint_included') is not False
            or m.get('pixel_tensors_included') is not False or m.get('synthetic_operands_only') is not True
            or m.get('limits')!=LIMITS or m.get('new_sources')!=list(SOURCES)
            or m.get('hosts')!=HOSTS or m.get('roles')!={k:list(v) for k,v in ROLES.items()}):
        raise ValueError('Wrong standalone attention diagnostic domain or resource bounds.')
    for name,item in m['files'].items():
        p=root/safe_name(name)
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file() or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
            raise ValueError('Changed attention handoff source: '+name)
        if p.suffix=='.pt':raise ValueError('A source-only attention packet may not contain tensor files.')
    for name,sha in m['original_scientific_binding']['source_sha256'].items():
        if name not in m['files'] or m['files'][name]['sha256']!=sha:raise ValueError('Missing/changed original science pin.')
    if any(name not in m['files'] for name in (*SOURCES,'attention-reference.json')):raise ValueError('Incomplete component source package.')
    return m,json.loads((root/'attention-reference.json').read_text())
