#!/usr/bin/env python3
"""Prepare an unarmed operator around the unchanged old 17 MB profile packet."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import predictor_eot8_record_operator_v2 as op
from models import predictor_eot8_record_client_v2 as client


def prepare(root,folder):
    root=Path(root).resolve();folder=Path(folder)
    if folder.exists():raise FileExistsError('Preserve previous preparation.')
    previous=root/'runs/pretrained/diagnostic/predictor-record-v2-20261106/independent-final-review/sealed-inputs.json'
    pins=json.loads(previous.read_text())
    if len(pins)!=6759:raise ValueError('Prior review coverage changed.')
    for name,want in pins.items():
        if op.old.digest(root/name)!=want:raise ValueError('Prior source/evidence changed: '+name)
    proposal_path=root/'configs/predictor_attention_eot8_proposal_v1.json';text=proposal_path.read_text();p=json.loads(text)
    if hashlib.sha256(text.encode()).hexdigest()!=op.PROPOSAL_SHA:raise ValueError('Proposal changed.')
    archive=p['reference']['existing_packet_archive'];spec=json.loads((root/Path(archive).parent/'vjepa-eot-profile.tar.gz.manifest.json').read_text())
    if op.old.digest(root/archive)!=op.PACKET_SHA:raise ValueError('Old archive changed.')
    sources={name:op.INIT if name=='models/__init__.py' else (root/name).read_bytes() for name in op.REMOTE_NAMES}
    local=set(op.REMOTE_NAMES)-{'models/__init__.py'}|{'models/predictor_eot8_record_client_v2.py','scripts/run_predictor_eot8_record_v2.py','scripts/prepare_predictor_eot8_record_v2.py','tests/test_predictor_eot8_record_target_v2.py','tests/test_predictor_eot8_record_operations_v2.py'}
    key=(root/'runs/pretrained/diagnostic/attention-replay1-v1-20261105/linux10603/known-hosts').read_text()
    raw_key=base64.b64decode(key.split()[2],validate=True)
    fingerprint='SHA256:'+base64.b64encode(hashlib.sha256(raw_key).digest()).decode().rstrip('=')
    if fingerprint!=op.base.SSH_FINGERPRINT:raise ValueError('Independently verified public host fingerprint changed.')
    c={'kind':op.KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
       'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','expected_driver':'580.178.04',
       'expected_ssh_fingerprint':fingerprint,'proposal_sha256':op.PROPOSAL_SHA,'proposal_text':text,'proposal':p,
       'paid_compute':False,'research_admitted':False,'old_grants_consumed':True,'authorization_present_at_preparation':False,'automatic_retries':0,
       'repository':str(root),'local_handoff_directory':str(folder.resolve()),'archive_path':archive,'archive_manifest':spec,'known_hosts_sha256':hashlib.sha256(key.encode()).hexdigest(),
       'local_source_sha256':{name:op.old.digest(root/name) for name in sorted(local)},
       'remote_source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},
       'prior_inputs_manifest':str(previous.relative_to(root)),'prior_inputs_manifest_sha256':op.old.digest(previous)}
    binding=(root/'configs/predictor_eot8_record_target_v2.json').read_text()
    c.update(host_binding_text=binding,host_binding=json.loads(binding),host_binding_sha256=op.HOST_BINDING_SHA,prior_consumed_control_sha256=op.PRIOR_CONSUMED_CONTROL,prior_consumed_claim_sha256=op.PRIOR_CONSUMED_CLAIM)
    client.ensure_previous_unarmed(root)
    op.validate_control(c);body=client.payload(c)
    folder.mkdir(mode=0o700,parents=True)
    op.old.write_json(folder/'CONTROL.json',c);(folder/'known-hosts').write_text(key)
    for name,extra in [('run-attended.sh','')]:
        script='#!/usr/bin/env bash\nset -euo pipefail\nexec '+client.shlex.quote(str(root/'.venv/bin/python'))+' -B '+client.shlex.quote(str(root/'scripts/run_predictor_eot8_record_v2.py'))+' --handoff '+client.shlex.quote(str(folder.resolve()))+extra+'\n'
        (folder/name).write_text(script);(folder/name).chmod(0o700)
    info={'kind':'predictor-eot8-record-unarmed-payload-v2','control_sha256':op.old.digest(folder/'CONTROL.json'),
          'old_packet_bytes':17192095,'remote_overlay_files':len(sources),'remote_overlay_bytes':sum(map(len,sources.values())),
          'stdin_bootstrap_bytes':len(body),'prior_pins_verified':len(pins),'authorization_present':False,'attempt_present':False,
          'new_remote_or_scientific_work':False,'local_CPU_tests_are_not_CUDA_qualification':True}
    op.old.write_json(folder/'payload-review.json',info)
    return c,info

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],args.output)[1],sort_keys=True))
