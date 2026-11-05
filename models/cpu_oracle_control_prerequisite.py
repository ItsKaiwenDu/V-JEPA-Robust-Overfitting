"""Bind the new control to the independently accepted v3 target, without granting it."""
import hashlib
import json
from pathlib import Path
import stat

REVIEW_PATH = 'report/checks/2026-11-04-cpu-oracle-capture-v3-target-review.json'
REVIEW_SHA = '783a390b9eef87b1fe9e7d700ea97bb39677330e07d3f551652e662246514c3c'
TARGET = 'runs/pretrained/diagnostic/cpu-oracle-capture-v3-20261104/linux10603'
CLAIM_SHA = '6396015bdd55750d18ba2f44bde79d9f18becaa35ff2d1270b7c937dbd7af704'
CONTROL_SHA = '51f8c3215411d2347d58895db3cce18be6b9ac977098c9c3197217d173b509b7'
AUTH_SHA = 'e51f2925c6461ccd771f4cb10839270f189b2a77b39f0bd9fc79a4900335dd26'
RESPONSE_SHA = 'f467312d9135fd474e7b1717ba2455f4dcb90df44d473e5e623e7ed1b0fdce15'
EXPORT_SHA = '4597b30308a3ca1b61748a75643ea1716b8aa0fcbcd7d28695e47743fc51e287'
BINDING = {
    'kind':'cpu-oracle-control-prerequisite-v1', 'target_host':'linux10603',
    'target_review_sha256':REVIEW_SHA, 'target_claim_sha256':CLAIM_SHA,
    'target_control_sha256':CONTROL_SHA, 'target_authorization_sha256':AUTH_SHA,
    'target_response_sha256':RESPONSE_SHA, 'target_export_sha256':EXPORT_SHA,
    'accepted_scope':'controlled-v3-CPU-capture-only',
    'control_host_authorized':False, 'old_failures_waived':False,
    'historical_discrepancy_resolved':False, 'research_admitted':False,
}
REQUIRED_PINS = {REVIEW_PATH:REVIEW_SHA, TARGET+'/CONTROL.json':CONTROL_SHA,
                 TARGET+'/AUTHORIZATION.json':AUTH_SHA, TARGET+'/ATTEMPT.json':CLAIM_SHA}


def validate_binding(value):
    if value != BINDING or any(type(value.get(k)) is not type(v) for k,v in BINDING.items()):
        raise ValueError('Require the exact accepted v3 target prerequisite; it grants no control attempt.')


def _digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def _safe(name):
    path=Path(name)
    if not name or path.is_absolute() or '..' in path.parts or str(path)!=name or '\\' in name or name=='.':
        raise ValueError('Unsafe target evidence path.')
    return name


def _file(root,name,sha):
    path=root/_safe(name)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or _digest(path)!=sha:
        raise ValueError('Accepted target evidence changed: '+name)
    return path


def verify_target(root):
    """Rehash the accepted target and its saved files locally; no process or network."""
    root=Path(root)
    for name,sha in REQUIRED_PINS.items():_file(root,name,sha)
    review=json.loads((root/REVIEW_PATH).read_text())
    expected={'kind':'cpu-oracle-capture-v3-target-independent-review',
        'independent_review_complete':True, 'controlled_v3_target_capture_passed':True,
        'target_control_sha256':CONTROL_SHA, 'authorization_sha256':AUTH_SHA,
        'consumed_attempt_sha256':CLAIM_SHA, 'attempt_consumed':True,
        'response_sha256':RESPONSE_SHA, 'original_export_sha256':EXPORT_SHA,
        'complete_original_workers':4, 'independent_stage_comparisons':2048,
        'independent_final_reference_comparisons':48,
        'bounds':{'rtol':1e-12,'atol':1e-14}, 'old_failures_waived':False,
        'historical_numerical_discrepancy_resolved':False, 'control_host_authorized':False,
        'old_acceptance_bridge_written':False, 'research_admitted':False}
    if any(type(review.get(k)) is not type(v) or review.get(k)!=v for k,v in expected.items()):
        raise ValueError('Accepted target review scope or decisions changed.')
    claim=json.loads((root/TARGET/'ATTEMPT.json').read_text())
    if claim['local_session']!='attended-gq6y5si5' or claim['control_sha256']!=CONTROL_SHA:
        raise ValueError('Wrong consumed target attempt.')
    session=root/TARGET/claim['local_session']
    if review['local_session']!=str(session.relative_to(root)):
        raise ValueError('Wrong accepted target session.')
    response=_file(root,str((session/'response.bin').relative_to(root)),RESPONSE_SHA)
    if response.stat().st_size!=review['response_bytes']:raise ValueError('Target response extent changed.')
    with response.open('rb') as handle:
        lines=[handle.readline(2**20+1) for _ in range(2)]
    if any(len(line)>2**20 or not line.endswith(b'\n') for line in lines):
        raise ValueError('Invalid target stream headers.')
    first,header=map(json.loads,lines)
    if first['control_sha256']!=CONTROL_SHA or header['control_sha256']!=CONTROL_SHA:
        raise ValueError('Target stream control differs.')
    saved=session/'saved';files=header['files'];actual=set()
    for path in saved.rglob('*'):
        if path.is_symlink():raise ValueError('Aliased saved target file.')
        if path.is_file():actual.add(str(path.relative_to(saved)))
    if len(files)!=453 or len(files)!=review['saved_members_verified'] or actual!=set(files):
        raise ValueError('Accepted target member coverage changed.')
    for name,item in files.items():
        path=_file(saved,name,item['sha256']);info=path.stat()
        if (not stat.S_ISREG(info.st_mode) or info.st_size!=item['bytes'] or info.st_mtime_ns!=item['mtime_ns']):
            raise ValueError('Target saved member extent or modification time changed: '+name)
    if sum(item['bytes'] for item in files.values())!=review['saved_bytes_verified']:
        raise ValueError('Target saved byte coverage changed.')
    for mapping in (review['current_source_hashes'],review['independent_review_artifacts']):
        for name,sha in mapping.items():_file(root,name,sha)
    _file(saved,'results/evidence.tar.gz',EXPORT_SHA)
    return {'target_review_sha256':REVIEW_SHA,'target_claim_sha256':CLAIM_SHA,
            'saved_members_verified':len(files),'response_sha256':RESPONSE_SHA,
            'control_authorized':False,'subprocesses_started':0,'SSH_connections':0}
