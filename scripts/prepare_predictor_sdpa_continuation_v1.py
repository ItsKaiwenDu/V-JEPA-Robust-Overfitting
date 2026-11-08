"""Prepare one unarmed native SDPA handoff locally. Never authorize or launch it."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import predictor_sdpa_continuation_operator_v1 as op
from models import predictor_sdpa_continuation_client_v1 as client


def prepare(root,folder):
    root=Path(root).resolve();folder=Path(folder).resolve()
    if folder.exists():raise FileExistsError('Preserve previous prepared/claimed handoffs.')
    contract_path=root/'configs/predictor_sdpa_continuation_v1.json';text=contract_path.read_text();contract=json.loads(text)
    if op.old.digest(contract_path)!=op.CONTRACT_SHA:raise ValueError('Reviewed target contract changed.')
    for key in ('parent_consumed_claim','local_startup_review','preserved_inputs','capsule_manifest','capsule'):
        item=contract[key]
        if op.old.digest(root/item['path'])!=item['sha256']:raise ValueError('Previous reviewed binding changed: '+key)
    pins=json.loads((root/contract['preserved_inputs']['path']).read_text())
    if len(pins)!=contract['preserved_inputs']['files']:raise ValueError('Prior evidence coverage changed.')
    for name,want in pins.items():
        if op.old.digest(root/name)!=want:raise ValueError('Frozen source/evidence changed: '+name)
    client.pending_unarmed(root)
    archive=contract['capsule']['path'];manifest_text=(root/contract['capsule_manifest']['path']).read_text();spec=json.loads(manifest_text)
    if op.old.digest(root/archive)!=op.CAPSULE_SHA:raise ValueError('Frozen capsule changed.')
    sources={n:op.INIT if n=='models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    key=(root/'runs/pretrained/diagnostic/attention-replay1-v1-20261105/linux10603/known-hosts').read_text()
    local=set(op.REMOTE_NAMES)-{'models/__init__.py'}|set(op.LOCAL_EXTRA)
    c={'kind':op.KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','expected_driver':'580.178.04',
        'expected_ssh_fingerprint':op.base.SSH_FINGERPRINT,'contract_sha256':op.CONTRACT_SHA,'prior_consumed_claim_sha256':op.PRIOR_CLAIM_SHA,
        'paid_compute':False,'research_admitted':False,'authorization_present_at_preparation':False,'automatic_retries':0,'resource':op.RESOURCE,
        'contract_text':text,'contract':contract,'archive_manifest_text':manifest_text,
        'repository':str(root),'local_handoff_directory':str(folder),'archive_path':archive,'archive_manifest':spec,
        'known_hosts_sha256':hashlib.sha256(key.encode()).hexdigest(),
        'local_source_sha256':{n:op.old.digest(root/n) for n in sorted(local)},
        'remote_source_sha256':{n:hashlib.sha256(data).hexdigest() for n,data in sources.items()}}
    op.validate_control(c);bootstrap=client.payload(c)
    folder.mkdir(mode=0o700,parents=True);op.old.write_json(folder/'CONTROL.json',c);(folder/'known-hosts').write_text(key)
    script='#!/usr/bin/env bash\nset -euo pipefail\nexec caffeinate -i '+client.shlex.quote(str(root/'.venv/bin/python'))+' -B '+client.shlex.quote(str(root/'scripts/run_predictor_sdpa_continuation_v1.py'))+' --handoff '+client.shlex.quote(str(folder))+'\n'
    (folder/'run-attended.sh').write_text(script);(folder/'run-attended.sh').chmod(0o700)
    info={'kind':'unarmed-component-only-payload-review-v1','control_sha256':op.old.digest(folder/'CONTROL.json'),
        'capsule_bytes':spec['bytes'],'capsule_members':len(spec['files']),'capsule_uncompressed_bytes':spec['logical_bytes'],
        'overlay_files':len(sources),'overlay_bytes':sum(map(len,sources.values())),'stdin_bootstrap_bytes':len(bootstrap),
        'prior_pins_verified':len(pins),'full_model_backwards':0,'new_native_captures':0,'authorization_present':False,'attempt_present':False,'new_remote_or_university_work':False}
    op.old.write_json(folder/'payload-review.json',info);client.check_local(folder)
    return c,info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],a.output)[1],sort_keys=True))
