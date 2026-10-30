"""Build a bounded CPU-only reference probe from the unchanged failed fixture."""
import json
from pathlib import Path
import shutil
import subprocess

from models.attention_component import operands
from models.attention_fixture import freeze,load
from models.cuda_profile import tensor_signature
from models.fixed_attention_packet import verify as verify_fixed
from models.pilot_bundle import digest,make_archive,safe_name

KIND='cpu-oracle-handoff-v1'
FIXED_SHA='61158d9b148ad32741d4cfaa7ec2862671b4ee5e534c0b9e1b26f611ec018b37'
HOSTS=('linux10603','linux10609')
TASKS=((4,0),(1,0),(4,1),(1,1))
LIMITS={'max_seconds':600,'worker_seconds':90,'max_export_bytes':2**30,'automatic_retries':0,
        'calls_per_worker':3,'stop_after_rejected_worker':True}
SOURCES=('models/oracle_trace.py','models/cpu_oracle_packet.py','models/cpu_oracle_probe.py',
         'scripts/check_cpu_oracle.py','scripts/prepare_cpu_oracle.py','scripts/university_cpu_oracle.sh',
         'tests/test_oracle_trace.py','tests/test_cpu_oracle_probe.py')


def build(root,output,*,tiny=False):
    root=Path(root).resolve();base=root/'runs/pretrained/diagnostic/fixed-attention-university-20261030'
    parent,reference=verify_fixed(base/'packet/handoff',FIXED_SHA)
    report_path=root/'report/checks/2026-10-30-fixed-attention-review.json';report=json.loads(report_path.read_text())
    if not report['independent_review_complete'] or report['both_runs_passed'] is not False:
        raise ValueError('Require the reviewed CPU oracle failure, without waiving it.')
    history={**report['new_consumed_claims_unchanged'],**report['prior_consumed_claims_unchanged']}
    for name,sha in history.items():
        if digest(root/name)!=sha:raise ValueError('Consumed attempt history changed.')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);stage=output/'handoff';stage.mkdir()
    selected={n:base/'packet/handoff'/n for n in parent['files'] if not n.startswith('fixtures/')}
    for name in SOURCES:selected[name]=root/name
    for name,path in selected.items():
        if path.is_symlink() or not path.is_file():raise ValueError('Require regular source members.')
        target=stage/safe_name(name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    if tiny:
        tensors,shape=operands(root/'external/vjepa2','encoder-1',reference,True);shape=dict(shape)
        shape['position_signature']=tensor_signature(shape.pop('positions'))
        item=freeze(stage/'fixtures/encoder-1','encoder-1',tensors,shape,{'kind':'tiny-local-probe-fixture-only'},tiny=True)
        fixture={'folder':'fixtures/encoder-1',**item}
    else:
        fixture=parent['fixtures']['encoder-1']
        source=base/'packet/handoff'/fixture['folder'];shutil.copytree(source,stage/fixture['folder'])
        load(stage/fixture['folder'],fixture['sha256'],case='encoder-1',tiny=False)
    m={'kind':KIND,'schema_version':1,'tiny_fixture':tiny,'CPU_only':True,'GPU_work_allowed':False,'device':'cpu',
       'diagnostic_only':True,'paid_compute':False,'launch_authorized':False,'research_admitted':False,
       'old_refusals_waived':False,'runtime_generation_allowed':False,'case':'encoder-1','fixture':fixture,
       'limits':LIMITS,'tasks':[list(t) for t in TASKS],'hosts':list(HOSTS),'prior_consumed_claims':history,
       'prior_handoff_sha256':FIXED_SHA,'prior_review_sha256':digest(report_path),'new_sources':list(SOURCES),
       'original_scientific_binding':parent['original_scientific_binding'],'upstream_sha256':parent['upstream_sha256'],
       'base_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
       'scope':'CPU-only forensic reference capture; original GPU-context initialization is deliberately absent.',
       'files':{str(p.relative_to(stage)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(stage.rglob('*')) if p.is_file()}}
    (stage/'CPU-ORACLE-HANDOFF.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    verify(stage,digest(stage/'CPU-ORACLE-HANDOFF.json'))
    spec=make_archive(output/'vjepa-cpu-oracle.tar.gz',{str(p.relative_to(stage)):p for p in stage.rglob('*') if p.is_file()})
    (output/'vjepa-cpu-oracle.tar.gz.manifest.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    return m,spec


def verify(root,expected):
    root=Path(root).resolve();path=root/'CPU-ORACLE-HANDOFF.json'
    if path.is_symlink() or digest(path)!=expected:raise ValueError('CPU oracle manifest changed.')
    m=json.loads(path.read_text())
    if (m.get('kind')!=KIND or m.get('schema_version')!=1 or m.get('device')!='cpu' or m.get('case')!='encoder-1'
            or type(m.get('tiny_fixture')) is not bool or m.get('CPU_only') is not True or m.get('diagnostic_only') is not True
            or any(m.get(k) is not False for k in ('GPU_work_allowed','paid_compute','launch_authorized','research_admitted','old_refusals_waived','runtime_generation_allowed'))
            or m.get('limits')!=LIMITS or m.get('tasks')!=[list(t) for t in TASKS] or m.get('hosts')!=list(HOSTS)
            or m.get('prior_handoff_sha256')!=FIXED_SHA or m.get('new_sources')!=list(SOURCES)
            or m.get('fixture',{}).get('folder')!='fixtures/encoder-1'):
        raise ValueError('Wrong CPU-only reference diagnostic scope or bounds.')
    if sum(v['bytes'] for v in m['files'].values())>64*2**20:raise ValueError('CPU oracle packet exceeds 64 MiB.')
    for name,item in m['files'].items():
        p=root/safe_name(name)
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file() or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
            raise ValueError('CPU source/fixture changed: '+name)
    if any(m['files'].get(n,{}).get('sha256')!=sha for n,sha in m['original_scientific_binding']['source_sha256'].items()):
        raise ValueError('Original scientific source changed.')
    if any(n not in m['files'] for n in SOURCES):raise ValueError('Missing CPU probe source.')
    if m['files'].get('fixtures/encoder-1/fixture.json',{}).get('sha256')!=m['fixture']['sha256']:
        raise ValueError('CPU probe fixture is unbound.')
    return m
