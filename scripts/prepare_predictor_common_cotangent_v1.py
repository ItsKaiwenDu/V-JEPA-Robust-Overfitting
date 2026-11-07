#!/usr/bin/env python3
"""Prepare exactly one unarmed four-cell attribution handoff; never grant work."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import predictor_common_cotangent_operator_v1 as op
from models import predictor_common_cotangent_client_v1 as client


def prepare(root,folder):
    root=Path(root).resolve();folder=Path(folder).resolve()
    if folder.exists():raise FileExistsError('Preserve prior preparations/claims.')
    path=root/'configs/predictor_common_cotangent_target_v1.json';contract_text=path.read_text();contract=json.loads(contract_text)
    if op.old.digest(path)!=op.CONTRACT_SHA:raise ValueError('New controlled-VJP contract changed.')
    review=contract['prior_review']
    if op.old.digest(root/review['path'])!=review['sha256']:raise ValueError('Offline review changed.')
    prior=contract['prior_manifest'];pins_path=root/prior['path']
    if op.old.digest(pins_path)!=prior['sha256']:raise ValueError('Prior manifest changed.')
    pins=json.loads(pins_path.read_text())
    if len(pins)!=10681:raise ValueError('Prior source/scientific coverage changed.')
    for name,want in pins.items():
        if op.old.digest(root/name)!=want:raise ValueError('Prior frozen file changed: '+name)
    for name,want in contract['prior_consumed_files'].items():
        if op.old.digest(root/name)!=want:raise ValueError('Consumed predecessor source/claim changed.')
    client.ensure_previous_unarmed(root)
    proposal_text=(root/'configs/predictor_common_cotangent_v1.json').read_text();proposal=json.loads(proposal_text)
    scientific=json.loads((root/'configs/predictor_attention_eot8_proposal_v1.json').read_text())
    archive_path=scientific['reference']['existing_packet_archive']
    spec=json.loads((root/Path(archive_path).parent/'vjepa-eot-profile.tar.gz.manifest.json').read_text())
    if op.old.digest(root/archive_path)!=op.PACKET_SHA:raise ValueError('Original packet changed.')
    capsule=contract['anchors_archive'];packed=(root/capsule['path']).read_bytes()
    if len(packed)!=capsule['bytes'] or hashlib.sha256(packed).hexdigest()!=capsule['sha256']:raise ValueError('Value-only archive changed.')
    with tarfile.open(root/capsule['path']) as archive:
        members=archive.getmembers()
        if len(members)!=len(capsule['files']) or {m.name for m in members}!=set(capsule['files']):raise ValueError('Exact capsule members required.')
        for m in members:
            item=capsule['files'][m.name]
            if not m.isfile() or m.size!=item['bytes']:raise ValueError('Capsule member changed.')
            with archive.extractfile(m) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=item['sha256']:raise ValueError('Capsule value changed.')
    sparse=contract['sparse_values'];sparse_text=(root/sparse['path']).read_text()
    if len(sparse_text.encode())!=sparse['bytes'] or hashlib.sha256(sparse_text.encode()).hexdigest()!=sparse['sha256']:raise ValueError('Sparse artifact changed.')
    sources={n:op.INIT if n=='models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    local=set(op.REMOTE_NAMES)-{'models/__init__.py'}|{'models/predictor_common_cotangent_client_v1.py',
        'scripts/prepare_predictor_common_cotangent_v1.py','scripts/run_predictor_common_cotangent_v1.py',
        'tests/test_predictor_common_cotangent_v1.py','tests/test_predictor_common_cotangent_operations_v1.py'}
    key=(root/'runs/pretrained/diagnostic/attention-replay1-v1-20261105/linux10603/known-hosts').read_text()
    fingerprint='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(key.split()[2],validate=True)).digest()).decode().rstrip('=')
    if fingerprint!=op.base.SSH_FINGERPRINT:raise ValueError('Public host key changed.')
    c={'kind':op.KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':contract['expected_gpu_uuid'],'expected_driver':'580.178.04','expected_ssh_fingerprint':fingerprint,
        'contract_text':contract_text,'contract':contract,'contract_sha256':op.CONTRACT_SHA,
        'proposal_text':proposal_text,'proposal':proposal,'proposal_sha256':op.PROPOSAL_SHA,
        'prior_consumed_claim_sha256':op.PRIOR_CLAIM_SHA,'paid_compute':False,'research_admitted':False,
        'old_grants_consumed':True,'authorization_present_at_preparation':False,'automatic_retries':0,
        'repository':str(root),'local_handoff_directory':str(folder),'archive_path':archive_path,'archive_manifest':spec,
        'anchors_path':str(folder/'value-anchors.tar.gz'),'anchors_manifest':{k:v for k,v in capsule.items() if k!='path'},
        'sparse_text':sparse_text,'sparse_data':json.loads(sparse_text),'sparse_sha256':sparse['sha256'],
        'known_hosts_sha256':hashlib.sha256(key.encode()).hexdigest(),
        'local_source_sha256':{n:op.old.digest(root/n) for n in sorted(local)},
        'remote_source_sha256':{n:hashlib.sha256(data).hexdigest() for n,data in sources.items()},
        'prior_inputs_manifest':prior['path'],'prior_inputs_manifest_sha256':prior['sha256']}
    op.validate_control(c);bootstrap=client.payload(c)
    folder.mkdir(mode=0o700,parents=True)
    with (folder/'value-anchors.tar.gz').open('xb') as f:f.write(packed)
    (folder/'cotangent-values.json').write_text(sparse_text)
    op.old.write_json(folder/'CONTROL.json',c);(folder/'known-hosts').write_text(key)
    script='#!/usr/bin/env bash\nset -euo pipefail\nexec '+client.shlex.quote(str(root/'.venv/bin/python'))+' -B '+client.shlex.quote(str(root/'scripts/run_predictor_common_cotangent_v1.py'))+' --handoff '+client.shlex.quote(str(folder))+'\n'
    (folder/'run-attended.sh').write_text(script);(folder/'run-attended.sh').chmod(0o700)
    info={'kind':'unarmed-predictor-common-cotangent-payload-v1','control_sha256':op.old.digest(folder/'CONTROL.json'),
        'old_packet_bytes':17192095,'anchor_archive_bytes':len(packed),'anchor_members':5,'anchor_logical_bytes':sum(i['bytes'] for i in capsule['files'].values()),
        'sparse_bytes':len(sparse_text.encode()),'overlay_files':len(sources),'overlay_bytes':sum(map(len,sources.values())),
        'stdin_bootstrap_bytes':len(bootstrap),'prior_pins_verified':len(pins),
        'authorization_present':False,'attempt_present':False,'new_remote_or_university_work':False}
    op.old.write_json(folder/'payload-review.json',info)
    return c,info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],a.output)[1],sort_keys=True))
