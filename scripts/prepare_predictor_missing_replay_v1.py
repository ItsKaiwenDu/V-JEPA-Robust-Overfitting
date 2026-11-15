#!/usr/bin/env python3
"""Prepare a concrete, unarmed one-replay handoff locally. No execution grant."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import predictor_missing_replay_operator_v1 as op
from models import predictor_missing_replay_client_v1 as client


def prepare(root,folder):
    root,folder=Path(root).resolve(),Path(folder).resolve()
    if folder.exists():raise FileExistsError('Preserve all prior preparation.')
    text=(root/op.CONTRACT).read_text();p=json.loads(text)
    if hashlib.sha256(text.encode()).hexdigest()!=op.CONTRACT_SHA:raise ValueError('Exact one-replay plan required.')
    previous=root/p['prior_inputs_manifest'];pins=json.loads(previous.read_text())
    if len(pins)!=2483:raise ValueError('Prior sealed coverage changed.')
    for n,sha in pins.items():
        if op.old.digest(root/n)!=sha:raise ValueError('Prior evidence/source changed: '+n)
    for n,sha in p['prior_consumed_files'].items():
        if op.old.digest(root/n)!=sha:raise ValueError('Consumed predecessor changed.')
    archive=p['reference']['existing_packet_archive']
    spec=json.loads((root/Path(archive).parent/'vjepa-eot-profile.tar.gz.manifest.json').read_text())
    if op.old.digest(root/archive)!=op.PACKET_SHA:raise ValueError('Original packet changed.')
    reference=root/p['reference_archive']['path'];manifest=json.loads((root/p['reference_manifest']['path']).read_text())
    if op.old.digest(reference)!=op.REFERENCE_SHA or reference.stat().st_size!=op.REFERENCE_BYTES:raise ValueError('Reference archive changed.')
    sources={n:op.INIT if n=='models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    key=(root/'runs/pretrained/diagnostic/attention-replay1-v1-20261105/linux10603/known-hosts').read_text()
    fingerprint='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(key.split()[2],validate=True)).digest()).decode().rstrip('=')
    if fingerprint!=op.base.SSH_FINGERPRINT:raise ValueError('Existing public host pin changed.')
    c={'kind':op.KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
       'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f',
       'expected_driver':'580.178.04','expected_ssh_fingerprint':fingerprint,
       'contract_sha256':op.CONTRACT_SHA,'contract_text':text,'contract':p,'resource':p['resource'],
       'paid_compute':False,'research_admitted':False,'authorization_present_at_preparation':False,'automatic_retries':0,
       'repository':str(root),'local_handoff_directory':str(folder),'archive_path':archive,'archive_manifest':spec,
       'reference_archive_path':p['reference_archive']['path'],'reference_manifest':manifest,
       'known_hosts_sha256':hashlib.sha256(key.encode()).hexdigest(),
       'local_source_sha256':{n:op.old.digest(root/n) for n in sorted(set(op.REMOTE_NAMES)-{'models/__init__.py'}|set(op.LOCAL_EXTRA))},
       'remote_source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
       'prior_inputs_manifest':str(previous.relative_to(root)),'prior_inputs_manifest_sha256':op.old.digest(previous),
       'host_binding':{'prior_consumed_files':p['prior_consumed_files']}}
    client.transport.ensure_previous_unarmed(root);op.validate_control(c);bootstrap=client.payload(c)
    target_counts=[family[1]['shape'][1] for draw in p['reference']['full_mask_tensor_identities'] for family in draw]
    new_raw=2*sum(target_counts)*1024*4+(2048*1024+3*16*256*256)*4+8*2*2048*8+36*4096
    if new_raw+214635140>p['resource']['raw_bytes']:raise ValueError('Shared historical/new raw allowance exceeded.')
    os.umask(0o077);folder.mkdir(mode=0o700,parents=True)
    op.old.write_json(folder/'CONTROL.json',c);(folder/'known-hosts').write_text(key)
    script='#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.quote(str(root/'.venv/bin/python'))+' -B '+shlex.quote(str(root/'scripts/run_predictor_missing_replay_v1.py'))+' --handoff '+shlex.quote(str(folder))+'\n'
    (folder/'run-attended.sh').write_text(script);(folder/'run-attended.sh').chmod(0o700)
    info={'kind':op.KIND,'control_sha256':op.old.digest(folder/'CONTROL.json'),'packet_bytes':17192095,
          'reference_archive_bytes':op.REFERENCE_BYTES,'compressed_inputs_bytes':17192095+op.REFERENCE_BYTES,
          'compressed_input_headroom_bytes':128*op.MiB-17192095-op.REFERENCE_BYTES,
          'overlay_files':len(sources),'overlay_bytes':sum(map(len,sources.values())),'bootstrap_bytes':len(bootstrap),
          'new_worker_raw_upper_bound_bytes':new_raw,'shared_reference_and_new_raw_upper_bound_bytes':new_raw+214635140,
          'prior_sealed_files_verified':len(pins),'authorization_present':False,'attempt_present':False,'new_remote_or_model_work':False}
    op.old.write_json(folder/'payload-review.json',info)
    return c,info

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],a.output)[1],sort_keys=True))
