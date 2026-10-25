#!/usr/bin/env python3
"""Build the frozen panel handoff with ten training tensors and no checkpoint."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def build(root,output):
    from models.matched_panel_case import PLAN_FILE_SHA256,PLAN_SHA256
    from models.matched_panel_io import RUNNER_SOURCES,load_clip,load_plan
    from models.pilot_bundle import CHECKPOINT,UPSTREAM,digest,make_archive,safe_name
    root,output=Path(root).resolve(),Path(output).resolve()
    plan=load_plan(root)
    if output.exists():raise FileExistsError(output)
    frozen=json.loads((root/'runs/pretrained/diagnostic/a40-controls-session-20261023/extracted/package/prior/extracted/handoff-manifest.json').read_text())
    pins={n:r['sha256'] for n,r in frozen['archives']['source']['files'].items() if not n.endswith('.md')}
    if len(pins)!=198 or any(digest(root/n)!=s for n,s in pins.items()):raise ValueError('Original scientific pins changed.')
    tracked=subprocess.check_output(['git','-C',str(root),'ls-files','-z'],text=True).split('\0')
    extras=(*RUNNER_SOURCES,'models/matched_panel_plan.py','scripts/prepare_matched_panel_plan.py',
        'tests/test_matched_panel_plan.py','configs/matched_panel_diagnostic_v1.json',
        'docs/matched_panel_diagnostic_plan.md','docs/matched_panel_handoff.md',
        'tests/test_matched_panel.py','tests/fixtures/matched_panel_worker.py',
        'report/checks/2026-10-24-matched-panel-predeclaration.json',
        'report/checks/2026-10-24-matched-panel-predeclaration.md',
        'report/checks/2026-10-24-matched-panel-seed-audit.json')
    files={}
    for name in sorted(set(n for n in tracked if n)|set(extras)):
        safe_name(name);path=root/name
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
                or path.stat().st_size>10*2**20 or path.suffix in ('.pt','.pth','.whl','.pem','.key')
                or any(p in ('.git','.venv','.ssh') for p in path.parts) or path.name=='.env' or path.name.startswith('.env.')):
            raise ValueError('Unexpected source/private/large input: '+name)
        files[name]=path
    prepared=root/'data/processed/staged-exploration-v1'
    for name in ('preparation.json','train/index.json'):
        files['data/processed/staged-exploration-v1/'+name]=prepared/name
    for case in plan['cases'][::2]:
        load_clip(root,plan,case['case_id'])
        name=plan['ordered_clips'][case['clip_index']]['prepared_file']
        files['data/processed/staged-exploration-v1/'+name]=prepared/name
    upstream=root/'external/vjepa2'
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM:
        raise ValueError('Wrong upstream revision.')
    if subprocess.check_output(['git','-C',str(upstream),'status','--porcelain','--untracked-files=normal',
        '--','src','app/vjepa','configs/train/vitl16'],text=True).strip():raise ValueError('Upstream relevant source changed.')
    output.mkdir(parents=True)
    subprocess.run(['git','-C',str(upstream),'bundle','create',str(output/'upstream.bundle'),'HEAD'],check=True)
    files['upstream.bundle']=output/'upstream.bundle'
    helper=(root/'models/pilot_bundle.py').read_text().split("\nif __name__ == '__main__':",1)[0]
    helper+='''
if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args()
    extract_verified(args.archive,json.loads(args.manifest.read_text()),args.destination)
    print('Panel handoff extracted and verified. No GPU job started.')
'''
    (output/'verify_extract.py').write_text(helper)
    files['verify_extract.py']=output/'verify_extract.py'
    manifest={'kind':'matched-panel-handoff-v1','plan_sha256':PLAN_SHA256,'plan_file_sha256':PLAN_FILE_SHA256,
        'diagnostic_only':True,'research_admitted':False,'checkpoint_included':False,
        'checkpoint_sha256':CHECKPOINT,'training_pixel_tensors':10,'held_out_pixels_included':False,
        'upstream_commit':UPSTREAM,'source_is_working_tree_snapshot':True,
        'original_scientific_pins_verified':198,'original_scientific_sha256':pins,
        'GPU_execution_performed':False,'first_case_independent_review_required':True,
        'files':{n:{'bytes':p.stat().st_size,'sha256':digest(p)} for n,p in files.items()}}
    (output/'PANEL-HANDOFF.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    files['PANEL-HANDOFF.json']=output/'PANEL-HANDOFF.json'
    archive=output/'vjepa-matched-panel-20261024.tar.gz'
    specification=make_archive(archive,{'matched-panel-20261024/'+n:p for n,p in files.items()})
    sidecar=output/(archive.name+'.manifest.json')
    sidecar.write_text(json.dumps(specification,indent=2,sort_keys=True)+'\n')
    (output/'SHA256SUMS').write_text(''.join(digest(p)+'  '+p.name+'\n' for p in (archive,sidecar,output/'verify_extract.py')))
    snapshot=output/'snapshot'
    snapshot.mkdir()
    for name,path in files.items():
        destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    return {'archive':str(archive),'bytes':specification['bytes'],'sha256':specification['sha256'],
            'members':len(specification['files']),'training_tensors':10,'GPU_job_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(build(ROOT,args.output),indent=2,sort_keys=True))


if __name__=='__main__':main()
