#!/usr/bin/env python3
"""Create one unarmed boundary handoff with immutable value-only anchors."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import predictor_boundary_operator_v1 as op
from models import predictor_boundary_client_v1 as client


def prepare(root,folder):
    root=Path(root).resolve();folder=Path(folder).resolve()
    if folder.exists():raise FileExistsError('Preserve prior preparation and claims.')
    contract_path=root/'configs/predictor_boundary_target_v1.json';contract_text=contract_path.read_text();contract=json.loads(contract_text)
    if op.old.digest(contract_path)!=op.CONTRACT_SHA:raise ValueError('New contract changed.')
    review_path=root/'report/checks/2026-11-06-predictor-gradient-attribution-v1-offline-review.json'
    if op.old.digest(review_path)!=contract['prior_review_sha256']:raise ValueError('Offline observation proposal review changed.')
    prior=contract['prior_manifest'];pins_path=root/prior['path']
    if op.old.digest(pins_path)!=prior['sha256']:raise ValueError('Prior sealed manifest changed.')
    pins=json.loads(pins_path.read_text())
    if len(pins)!=7526:raise ValueError('Prior scientific/source coverage changed.')
    for n,want in pins.items():
        if op.old.digest(root/n)!=want:raise ValueError('Prior frozen file changed: '+n)
    for n,want in contract['prior_consumed_files'].items():
        if op.old.digest(root/n)!=want:raise ValueError('Consumed target source/claim changed.')
    client.ensure_previous_unarmed(root)
    proposal_text=(root/'configs/predictor_gradient_attribution_v1.json').read_text();proposal=json.loads(proposal_text)
    scientific=json.loads((root/'configs/predictor_attention_eot8_proposal_v1.json').read_text())
    archive_path=scientific['reference']['existing_packet_archive']
    spec=json.loads((root/Path(archive_path).parent/'vjepa-eot-profile.tar.gz.manifest.json').read_text())
    if op.old.digest(root/archive_path)!=op.PACKET_SHA:raise ValueError('Original packet changed.')
    # The new capsule contains four unchanged saved returns, not workers or a grant.
    data={'ANCHORS.json':op.old.json_bytes(op.anchor_metadata(contract))}
    for i in (0,1):
        for item in contract['anchors'][str(i)]['artifacts'].values():
            p=root/item['source_path'];raw=p.read_bytes()
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Previously reviewed value changed.')
            data[item['file']]=raw
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            for name,raw in sorted(data.items()):
                m=tarfile.TarInfo(name);m.size=len(raw);m.mode=0o400;m.mtime=0
                archive.addfile(m,io.BytesIO(raw))
    packed=buffer.getvalue()
    capsule={'bytes':len(packed),'sha256':hashlib.sha256(packed).hexdigest(),
        'files':{n:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} for n,raw in data.items()}}
    sources={n:op.INIT if n=='models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    local=set(op.REMOTE_NAMES)-{'models/__init__.py'}|{'models/predictor_boundary_client_v1.py','scripts/prepare_predictor_boundary_v1.py','scripts/run_predictor_boundary_v1.py','tests/test_predictor_boundary_v1.py','tests/test_predictor_boundary_operations_v1.py'}
    key=(root/'runs/pretrained/diagnostic/attention-replay1-v1-20261105/linux10603/known-hosts').read_text()
    fingerprint='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(key.split()[2],validate=True)).digest()).decode().rstrip('=')
    if fingerprint!=op.base.SSH_FINGERPRINT:raise ValueError('Verified public host key changed.')
    c={'kind':op.KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':contract['expected_gpu_uuid'],'expected_driver':'580.178.04','expected_ssh_fingerprint':fingerprint,
        'contract_text':contract_text,'contract':contract,'contract_sha256':op.CONTRACT_SHA,
        'proposal_text':proposal_text,'proposal':proposal,'proposal_sha256':op.PROPOSAL_SHA,
        'prior_consumed_claim_sha256':op.PRIOR_CLAIM_SHA,'paid_compute':False,'research_admitted':False,
        'old_grants_consumed':True,'authorization_present_at_preparation':False,'automatic_retries':0,
        'repository':str(root),'local_handoff_directory':str(folder),'archive_path':archive_path,'archive_manifest':spec,
        'anchors_path':str(folder/'value-anchors.tar.gz'),'anchors_manifest':capsule,
        'known_hosts_sha256':hashlib.sha256(key.encode()).hexdigest(),
        'local_source_sha256':{n:op.old.digest(root/n) for n in sorted(local)},
        'remote_source_sha256':{n:hashlib.sha256(raw).hexdigest() for n,raw in sources.items()},
        'prior_inputs_manifest':prior['path'],'prior_inputs_manifest_sha256':prior['sha256']}
    op.validate_control(c);bootstrap=client.payload(c)
    folder.mkdir(mode=0o700,parents=True)
    with (folder/'value-anchors.tar.gz').open('xb') as f:f.write(packed)
    op.old.write_json(folder/'CONTROL.json',c);(folder/'known-hosts').write_text(key)
    script='#!/usr/bin/env bash\nset -euo pipefail\nexec '+client.shlex.quote(str(root/'.venv/bin/python'))+' -B '+client.shlex.quote(str(root/'scripts/run_predictor_boundary_v1.py'))+' --handoff '+client.shlex.quote(str(folder))+'\n'
    (folder/'run-attended.sh').write_text(script);(folder/'run-attended.sh').chmod(0o700)
    info={'kind':'unarmed-predictor-boundary-payload-v1','control_sha256':op.old.digest(folder/'CONTROL.json'),
        'old_packet_bytes':17192095,'anchor_archive_bytes':len(packed),'anchor_members':len(data),'anchor_logical_bytes':sum(map(len,data.values())),
        'overlay_files':len(sources),'overlay_bytes':sum(map(len,sources.values())),'stdin_bootstrap_bytes':len(bootstrap),'prior_pins_verified':len(pins),
        'authorization_present':False,'attempt_present':False,'new_remote_or_university_work':False}
    op.old.write_json(folder/'payload-review.json',info)
    return c,info


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],args.output)[1],sort_keys=True))
