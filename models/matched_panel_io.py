"""Checksummed panel handoffs, exact training inputs and bounded case exports."""

import hashlib
import json
from pathlib import Path
import tarfile

import torch

from models.matched_panel_case import (KIND, PLAN_FILE_SHA256, PLAN_SHA256, case_spec,
                                     execution_gate, pixel_digest)
from models.matched_panel_plan import plan_digest, validate_plan
from models.pilot_bundle import CHECKPOINT, digest, extract_verified, make_archive, safe_name
from models.study_artifacts import read_record, write_record

RUNNER_SOURCES = ('models/matched_panel_case.py','models/matched_panel.py','models/matched_panel_io.py',
    'models/matched_panel_runner.py','scripts/check_matched_panel.py',
    'scripts/prepare_matched_panel_handoff.py','scripts/university_matched_panel.sh')


def load_plan(root):
    path=Path(root)/'configs/matched_panel_diagnostic_v1.json'
    if digest(path)!=PLAN_FILE_SHA256:
        raise ValueError('Frozen panel file identity changed.')
    plan=json.loads(path.read_text())
    if plan_digest(plan)!=PLAN_SHA256:
        raise ValueError('Frozen panel payload identity changed.')
    validate_plan(root,plan)
    return plan


def verify_handoff(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'PANEL-HANDOFF.json').read_text())
    if (manifest['kind']!='matched-panel-handoff-v1' or manifest['plan_sha256']!=PLAN_SHA256
            or manifest['plan_file_sha256']!=PLAN_FILE_SHA256 or manifest['training_pixel_tensors']!=10
            or manifest['checkpoint_included'] or manifest['research_admitted']):
        raise ValueError('Wrong panel handoff identity.')
    for name,expected in manifest['files'].items():
        safe_name(name)
        path=root/name
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
                or path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']):
            raise ValueError('Panel handoff file changed: '+name)
    return manifest


def execution_identity(root,plan):
    root=Path(root).resolve()
    manifest=verify_handoff(root)
    required=set(RUNNER_SOURCES)|set(plan['identity']['source_sha256'])
    if not required<=set(manifest['files']):
        raise ValueError('Handoff is missing qualified execution source files.')
    return {'kind':'matched-panel-execution-v1','domain':'university-cuda',
        'plan_sha256':plan_digest(plan),'handoff_manifest_sha256':digest(root/'PANEL-HANDOFF.json'),
        'expected_runtime':plan['identity']['reference_runtime_for_future_match'],
        'expected_weights':plan['identity']['component_weights'],
        'checkpoint_sha256':CHECKPOINT,'expected_host':plan['identity']['reference_host'],
        'expected_GPU_identity':plan['identity']['reference_GPU_identity'],
        'source_sha256':{n:manifest['files'][n]['sha256'] for n in sorted(required)},
        'upstream_sha256':plan['identity']['upstream_sha256'],'research_admitted':False}


def load_clip(root,plan,case_id):
    """One real training tensor by declared identity; no synthetic ID aliases."""
    case=case_spec(plan,case_id)
    clip=plan['ordered_clips'][case['clip_index']]
    prepared=Path(root)/'data/processed/staged-exploration-v1'
    index=json.loads((prepared/'train/index.json').read_text())
    entries=[e for e in index['clips'] if e['video_id']==clip['video_id']]
    if len(entries)!=1:
        raise ValueError('Declared clip is not unique in the training cache.')
    entry=entries[0]
    if (clip['prepared_file']!='train/'+entry['file'] or entry['pixel_sha256']!=clip['pixel_sha256']
            or any(entry[k]!=clip[k] for k in ('video_id','class_id','relative_path','source_group_id'))):
        raise ValueError('Training cache metadata differs from the frozen case.')
    safe_name(clip['prepared_file'])
    path=prepared/clip['prepared_file']
    if path.is_symlink() or not path.resolve().is_relative_to(prepared.resolve()):
        raise ValueError('Escaped pixel path.')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    pixels=payload['pixels']
    if (payload['metadata']!={k:v for k,v in entry.items() if k!='file'}
            or pixels.dtype!=torch.float32 or tuple(pixels.shape)!=(3,16,256,256)
            or not torch.isfinite(pixels).all() or pixels.min()<0 or pixels.max()>1
            or pixel_digest(pixels)!=clip['pixel_sha256']):
        raise ValueError('Declared training pixels or metadata failed verification.')
    return pixels.unsqueeze(0)


def verify_archive(path,manifest,expected_sha256):
    path=Path(path)
    if (path.is_symlink() or digest(path)!=expected_sha256 or manifest['sha256']!=expected_sha256
            or path.stat().st_size!=manifest['bytes']):
        raise ValueError('Case export checksum or size mismatch.')
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        names=[safe_name(m.name) for m in members]
        if (len(names)!=len(set(names)) or set(names)!=set(manifest['files'])
                or any(not m.isfile() for m in members)
                or sum(m.size for m in members)>256*2**20):
            raise ValueError('Invalid or oversized case export inventory.')
        for member in members:
            expected=manifest['files'][member.name]
            if member.size!=expected['bytes']:
                raise ValueError('Case export member size mismatch.')
            with archive.extractfile(member) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected['sha256']:
                    raise ValueError('Case export member checksum mismatch.')
    return len(members)


def export_case(root,directory,plan,execution,*,destination=None):
    root,directory=Path(root),Path(directory)
    files={}
    for role in ('matrix','replay'):
        for name in ('contract.json','inputs.json','inputs.pt','worker.json','observation.json',
                     *[a+s for a in 'ABCD' for s in ('.json','.pt')]):
            path=directory/role/name
            if path.is_file() and not path.is_symlink():
                if path.stat().st_size>65*2**20:
                    raise ValueError('Unexpected large panel worker artifact.')
                files[role+'/'+name]=path
    for name in ('session.json','matrix.log','replay.log','receipt.json'):
        path=directory/name
        if path.is_file() and not path.is_symlink(): files[name]=path
    for name,payload in [('plan.json',plan),('execution.json',execution)]:
        path=directory/name
        if path.exists():
            if read_record(path)!=payload:raise ValueError('Existing export identity differs.')
        else: write_record(path,payload)
        files[name]=path
    if (root/'PANEL-HANDOFF.json').is_file():files['PANEL-HANDOFF.json']=root/'PANEL-HANDOFF.json'
    for name in RUNNER_SOURCES:
        files['source/'+name]=root/name
    if sum(p.stat().st_size for p in files.values())>256*2**20:
        raise ValueError('Panel case exceeds the declared export limit; preserve partial files.')
    destination=Path(destination) if destination is not None else directory/'evidence.tar.gz'
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest=make_archive(destination,files)
    Path(str(destination)+'.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def verify_export_sources(output,execution):
    """Check captured execution files against the external trusted handoff."""
    output=Path(output)
    if digest(output/'PANEL-HANDOFF.json')!=execution['handoff_manifest_sha256']:
        raise ValueError('Exported handoff manifest differs from the trusted execution.')
    for name in RUNNER_SOURCES:
        path=output/'source'/name
        if path.is_symlink() or digest(path)!=execution['source_sha256'][name]:
            raise ValueError('Exported execution source differs from the trusted handoff: '+name)


def review_export(archive,manifest,expected_sha256,output,*,plan,execution,allow_fixture=False):
    """External reviewer uses the trusted handoff and an out-of-band tar hash."""
    from models.matched_panel_case import verify_case
    execution_gate(plan,execution,allow_fixture=allow_fixture)
    verify_archive(archive,manifest,expected_sha256)
    output=Path(output)
    extract_verified(archive,manifest,output)
    if (read_record(output/'plan.json')!=plan or read_record(output/'execution.json')!=execution):
        raise ValueError('Export does not belong to the trusted frozen handoff.')
    if execution['domain']=='university-cuda':verify_export_sources(output,execution)
    case_id=read_record(output/'session.json')['case_id']
    receipt=verify_case(output,plan=plan,case_id=case_id,execution=execution,allow_fixture=allow_fixture)
    if read_record(output/'receipt.json')!=receipt:
        raise ValueError('Exported case receipt differs from independently retained evidence.')
    return receipt
