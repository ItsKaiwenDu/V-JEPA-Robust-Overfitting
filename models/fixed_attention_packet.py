"""Prepare a new, unarmed immutable-input diagnostic; old attempts stay frozen."""
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import torch

from models.attention_component import CASES,operands
from models.attention_fixture import freeze,load
from models.attention_packet import HOSTS as OLD_HOSTS,verify as verify_old
from models.candidate_contract import binding
from models.cuda_profile import tensor_signature
from models.pilot_bundle import digest,make_archive,safe_name
from models.study_artifacts import read_record

KIND='fixed-attention-handoff-v1'
OLD_SHA='c70c4f46f271499a2f38f16a60ec1b6c9f6c34eaedf617341b12dc82d0c944c1'
LIMITS={'max_seconds':1200,'worker_seconds':120,'allocator_gib':4,'warmup':1,'repeats':3,'automatic_retries':0}
ROLES={'encoder-0':('encoder-0',),'encoder-1':('encoder-1',)}
HOSTS={h:{'role':'encoder-0' if h=='linux10609' else 'encoder-1','uuid':v['uuid']} for h,v in OLD_HOSTS.items()}
SOURCES=('models/attention_fixture.py','models/fixed_attention_packet.py','models/fixed_attention_diagnostic.py',
         'scripts/check_fixed_attention.py','scripts/prepare_fixed_attention.py','scripts/university_fixed_attention.sh',
         'tests/test_attention_fixture.py','tests/test_fixed_attention.py','models/attention_saved_evidence.py')


def build(root,output,*,tiny=False):
    root=Path(root).resolve();base=root/'runs/pretrained/diagnostic/attention-components-university-20261030'
    old,reference=verify_old(base/'packet/handoff',OLD_SHA)
    review_path=root/'report/checks/2026-10-30-attention-component-review.json';review=json.loads(review_path.read_text())
    if not review['independent_review_complete'] or review['both_diagnostics_passed'] is not False:
        raise ValueError('Bind the actual mixed review, preserving both failed gates.')
    prior={}
    for host in HOSTS:
        claim=base/host/'ATTENDED-LAUNCH-CLAIM.json';plan=base/host/'actual-launch.json'
        for p in (claim,plan):
            if digest(p)!=review['launch_claims_unchanged'][str(p.relative_to(root))]:raise ValueError('Prior consumed attempt changed.')
        prior[host]={'claim_sha256':digest(claim),'plan_sha256':digest(plan)}
    if binding()!=old['original_scientific_binding']:raise ValueError('Original scientific pins changed.')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);stage=output/'handoff';stage.mkdir()
    paths={name:base/'packet/handoff'/name for name in old['files']}
    for name in SOURCES:paths[name]=root/name
    for name,p in paths.items():
        if p.is_symlink() or not p.is_file():raise ValueError('Regular source files required.')
        target=stage/safe_name(name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    fixtures={}
    for case in CASES:
        if not tiny and case in ('predictor','encoder-0'):
            host='linux10609' if case=='predictor' else 'linux10603'
            d=root/review['host_results'][host]['archive']['download'];folder=d/'unpacked/results/workers'/(case+'-efficient-0')
            w=read_record(folder/'worker.json')
            if not w['passed'] or not w['complete'] or digest(folder/'values.pt')!=w['artifact']['sha256']:
                raise ValueError('Prior immutable seed values changed.')
            if digest(folder/'worker.json')!=read_record(d/'unpacked/results/session.json')['tasks'][0]['worker_sha256']:
                raise ValueError('Prior worker source receipt changed.')
            saved=torch.load(folder/'values.pt',map_location='cpu',weights_only=True)
            tensors=tuple(saved[n] for n in ('q','k','v','cotangent'));shape=w['common']['shape']
            if any(tensor_signature(t)!=w['common']['operand_signatures'][n] for n,t in zip(('q','k','v','cotangent'),tensors)):
                raise ValueError('Prior logical operand identity changed.')
            origin={'kind':'reviewed-saved-synthetic-operands','host':host,'archive_sha256':review['host_results'][host]['archive']['sha256'],
                    'worker_sha256':digest(folder/'worker.json'),'values_sha256':w['artifact']['sha256'],'prior_review_sha256':digest(review_path)}
        else:
            tensors,shape=operands(root/'external/vjepa2',case,reference,tiny)
            shape=dict(shape);shape['position_signature']=tensor_signature(shape.pop('positions'))
            origin={'kind':'single-cpu-upstream-synthetic-capture','platform':platform.machine(),'torch':str(torch.__version__),
                    'numpy':np.__version__,'prior_review_sha256':digest(review_path),'generator_replay_qualified':False}
        item=freeze(stage/'fixtures'/case,case,tensors,shape,origin,tiny=tiny)
        fixtures[case]={'folder':'fixtures/'+case,**item}
        load(stage/'fixtures'/case,item['sha256'],case=case,tiny=tiny)
    metadata={'kind':KIND,'schema_version':1,'diagnostic_only':True,'tiny_fixture':tiny,'launch_authorized':False,
        'paid_compute':False,'research_admitted':False,'generator_replay_qualified':False,'runtime_generation_allowed':False,
        'synthetic_only':True,'checkpoint_included':False,'pixel_tensors_included':False,
        'prior_handoff_sha256':OLD_SHA,'prior_review_sha256':digest(review_path),'prior_consumed_attempts':prior,
        'original_scientific_binding':binding(),'upstream_commit':old['upstream_commit'],'upstream_sha256':old['upstream_sha256'],
        'roles':{r:list(c) for r,c in ROLES.items()},'hosts':HOSTS,'limits':LIMITS,'new_sources':list(SOURCES),'fixtures':fixtures,
        'base_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'scope':'New immutable-input component diagnostic only; generation replay failures and prior attempts preserved.',
        'files':{str(p.relative_to(stage)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(stage.rglob('*')) if p.is_file()}}
    (stage/'FIXED-ATTENTION-HANDOFF.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    verify(stage,digest(stage/'FIXED-ATTENTION-HANDOFF.json'))
    spec=make_archive(output/'vjepa-fixed-attention.tar.gz',{str(p.relative_to(stage)):p for p in stage.rglob('*') if p.is_file()})
    (output/'vjepa-fixed-attention.tar.gz.manifest.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    return metadata,spec


def verify(root,expected):
    root=Path(root).resolve();p=root/'FIXED-ATTENTION-HANDOFF.json'
    if p.is_symlink() or digest(p)!=expected:raise ValueError('Immutable attention manifest changed.')
    m=json.loads(p.read_text())
    if (m.get('kind')!=KIND or m.get('schema_version')!=1 or type(m.get('tiny_fixture')) is not bool
            or any(m.get(k) is not False for k in ('paid_compute','research_admitted','launch_authorized','generator_replay_qualified',
                                                   'runtime_generation_allowed','checkpoint_included','pixel_tensors_included'))
            or m.get('diagnostic_only') is not True or m.get('synthetic_only') is not True
            or m.get('new_sources')!=list(SOURCES) or m.get('limits')!=LIMITS or m.get('hosts')!=HOSTS
            or m.get('roles')!={r:list(c) for r,c in ROLES.items()} or m.get('prior_handoff_sha256')!=OLD_SHA
            or set(m.get('fixtures',{}))!=set(CASES) or set(m.get('prior_consumed_attempts',{}))!=set(HOSTS)):
        raise ValueError('Wrong immutable-input diagnostic contract.')
    if sum(v['bytes'] for v in m['files'].values())>256*2**20:raise ValueError('Fixed packet exceeds 256 MiB.')
    for name,item in m['files'].items():
        p=root/safe_name(name)
        if (p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file()
                or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']):raise ValueError('Fixed source/input changed: '+name)
        if p.suffix in ('.pt','.pth','.ckpt','.whl'):raise ValueError('Only source metadata and immutable NumPy arrays are allowed.')
    for name,sha in m['original_scientific_binding']['source_sha256'].items():
        if m['files'].get(name,{}).get('sha256')!=sha:raise ValueError('Original scientific pin changed.')
    for case,item in m['fixtures'].items():
        if item['folder']!='fixtures/'+case or item['descriptor']!='fixture.json':raise ValueError('Fixture path changed.')
        n=item['folder']+'/fixture.json'
        if m['files'].get(n,{}).get('sha256')!=item['sha256']:raise ValueError('Fixture descriptor is not bound to packet.')
    if any(n not in m['files'] for n in (*SOURCES,'attention-reference.json')):raise ValueError('Incomplete fixed-input source package.')
    return m,json.loads((root/'attention-reference.json').read_text())
